package com.sensorworldmodel.collector;

import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.SystemClock;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * The single point where inertial data enters this app.
 *
 * <p>Two suppliers, one connected at a time:
 *
 * <pre>
 *   real sensors  ──┐
 *                   ├──► ImuBus ──► CaptureService.recordSample ──► the CSV
 *   injected frames─┘
 * </pre>
 *
 * <p><b>The real sensors are stopped, not ignored.</b> The phone is in someone's
 * hand and moving; if its own stream kept flowing while a synthetic one was
 * added, the recording would contain two hands at once and the seam between
 * them would be louder than any single gesture. So switching to INJECTED calls
 * {@link SensorManager#unregisterListener} for real, and switching back
 * re-registers.
 *
 * <p>What reaches the CSV is byte-for-byte the same shape either way. The app
 * does not record which supplier a row came from, because the app is supposed
 * to be the same instrument whether a person or an agent is operating it, and a
 * column naming the condition would put the answer in the data. Provenance for
 * our own verification lives in {@link #recent()}, an in-memory ring the control
 * channel can read and nothing else ever sees.
 */
final class ImuBus {

    static final String MODE_REAL = "real";
    static final String MODE_INJECTED = "injected";

    /** Frames are six channels: ax, ay, az, gx, gy, gz. */
    static final int CHANNELS = 6;
    private static final int RING_CAPACITY = 20000;

    private static final ImuBus INSTANCE = new ImuBus();

    static ImuBus get() {
        return INSTANCE;
    }

    private ImuBus() {}

    /** One row of what was actually delivered, for verification only. */
    static final class Row {
        final long seq;
        final long timestampNs;
        final float[] values;      // six channels
        final String origin;       // "real" | "injected"
        final String bundleId;
        final int frameIndex;

        Row(long seq, long timestampNs, float[] values, String origin,
            String bundleId, int frameIndex) {
            this.seq = seq;
            this.timestampNs = timestampNs;
            this.values = values;
            this.origin = origin;
            this.bundleId = bundleId;
            this.frameIndex = frameIndex;
        }
    }

    private CaptureService service;
    private SensorManager sensorManager;
    private Handler handler;

    private volatile String mode = MODE_REAL;

    private float[][] background = new float[0][];
    private double backgroundPeriodMs = 10.0;
    private long backgroundAnchorNs = 0L;
    private boolean backgroundRunning = false;
    private Runnable backgroundTick;
    private long backgroundFramesDelivered = 0L;
    private double backgroundNextUptimeMs = 0.0;
    // The spans an action's own window occupies, so the background knows when
    // to stand aside.
    private final java.util.List<long[]> windowSpans = new java.util.ArrayList<>();

    private final Deque<Row> ring = new ArrayDeque<>();
    private long sequence = 0L;
    private long droppedRows = 0L;

    // Counters the control channel reports, so a run can say what happened
    // rather than what was intended.
    private long injectedFrames = 0L;
    private long realFrames = 0L;
    private double maxLatenessMs = 0.0;
    private double totalLatenessMs = 0.0;
    private long latenessSamples = 0L;

    synchronized void attach(CaptureService service, SensorManager sensorManager, Handler handler) {
        this.service = service;
        this.sensorManager = sensorManager;
        this.handler = handler;
    }

    synchronized void detach(CaptureService service) {
        if (this.service == service) {
            this.service = null;
            this.sensorManager = null;
            this.handler = null;
        }
    }

    String mode() {
        return mode;
    }

    boolean attached() {
        return service != null;
    }

    // -- supplier selection --------------------------------------------------

    /**
     * Connect one supplier and disconnect the other.
     *
     * @return the mode actually in force afterwards
     */
    synchronized String setMode(String requested) {
        String next = MODE_INJECTED.equals(requested) ? MODE_INJECTED : MODE_REAL;
        if (next.equals(mode)) {
            return mode;
        }
        mode = next;
        if (service == null || sensorManager == null) {
            // Nothing is capturing yet; the mode is remembered and applied when
            // the service starts, rather than silently lost.
            return mode;
        }
        if (MODE_INJECTED.equals(mode)) {
            sensorManager.unregisterListener(service);
            startBackground();
        } else {
            stopBackground();
            service.registerSensorsForBus();
        }
        return mode;
    }

    /** Whether a real sensor callback should be let through right now. */
    boolean acceptsReal() {
        return MODE_REAL.equals(mode);
    }

    // -- what plays ----------------------------------------------------------

    synchronized void setBackground(float[][] frames, double periodMs) {
        this.background = frames;
        this.backgroundPeriodMs = periodMs > 0 ? periodMs : 10.0;
        this.backgroundAnchorNs = SystemClock.elapsedRealtimeNanos();
        if (MODE_INJECTED.equals(mode)) {
            startBackground();
        }
    }

    /**
     * Keep the stream running between actions.
     *
     * <p><b>Silence is the loudest thing this system can emit.</b> Switching to
     * INJECTED unregisters the real sensors, so with nothing else supplying
     * frames the app receives absolutely nothing until the next action -- and an
     * agent spends most of its time thinking. Measured on a real run: nine gaps
     * over 200 ms and one of <b>257 seconds</b> with not a single sample, against
     * a human recording that is continuous at 100 Hz. No amount of realism
     * inside a gesture survives four minutes of a phone that was not being held.
     *
     * <p>So the background loops whenever no action window covers the moment.
     * Its content is the padding either side of a gesture, which is the same
     * hand before it reached the glass and after it left -- not synthesised
     * stillness, and not zeros.
     */
    private void startBackground() {
        if (handler == null || backgroundRunning || background.length == 0) {
            return;
        }
        backgroundRunning = true;
        final double periodMs = backgroundPeriodMs;
        // Absolute deadlines, not "wait a period after finishing".
        //
        // postDelayed re-arms *after* the callback returns, so every tick pays
        // for the previous one and the error accumulates: a 10 ms period played
        // back at 79 Hz with gaps up to 36 ms, against a recording the app asks
        // Android for at 100 Hz. Rate is exactly the kind of thing a check on a
        // whole session looks at, so the schedule advances by the period
        // regardless of how long a delivery took.
        backgroundNextUptimeMs = SystemClock.uptimeMillis();
        final Runnable tick = new Runnable() {
            @Override
            public void run() {
                if (!backgroundRunning) {
                    return;
                }
                long resumeAtUptimeMs = deliverBackground(SystemClock.elapsedRealtimeNanos());
                if (resumeAtUptimeMs > 0 && handler != null) {
                    // Inside an action's window.  Sleeping until it ends beats
                    // waking every period only to decide to stay quiet: a
                    // 1.79 s scroll window meant 179 such wake-ups competing
                    // with its own 179 frames on this one handler, and frames
                    // arrived up to 17.8 ms late against a 10 ms grid -- six of
                    // them past the point where they still counted as
                    // delivered.  Taps and swipes, with shorter windows, never
                    // showed it.
                    backgroundNextUptimeMs = resumeAtUptimeMs;
                    handler.postAtTime(this, Math.round(backgroundNextUptimeMs));
                    return;
                }
                backgroundNextUptimeMs += periodMs;
                double now = SystemClock.uptimeMillis();
                if (backgroundNextUptimeMs < now) {
                    // Fell behind -- the phone was busy. Resynchronise rather
                    // than firing a burst to catch up, which would be a rate no
                    // sensor produces.
                    backgroundNextUptimeMs = now + periodMs;
                }
                if (handler != null) {
                    handler.postAtTime(this, Math.round(backgroundNextUptimeMs));
                }
            }
        };
        backgroundTick = tick;
        handler.post(tick);
    }

    private synchronized void stopBackground() {
        backgroundRunning = false;
        if (handler != null && backgroundTick != null) {
            handler.removeCallbacks(backgroundTick);
        }
        backgroundTick = null;
    }

    /**
     * @return the uptime to resume at when an action's window covers this
     *     instant, or 0 when a background frame was delivered
     */
    private long deliverBackground(long nowNs) {
        CaptureService target;
        float[] frame;
        synchronized (this) {
            if (!MODE_INJECTED.equals(mode) || service == null || background.length == 0) {
                return 0L;
            }
            // An action's own inertia wins wherever the two would both apply;
            // doubling them would put two hands on the phone at once.
            for (long[] span : windowSpans) {
                if (nowNs >= span[0] && nowNs < span[1]) {
                    // Hand the handler back until this window is done.
                    return SystemClock.uptimeMillis() + (span[1] - nowNs) / 1_000_000L + 1L;
                }
            }
            target = service;
            long periodNs = Math.round(backgroundPeriodMs * 1_000_000.0);
            long span = (long) background.length * periodNs;
            long delta = (nowNs - backgroundAnchorNs) % span;
            if (delta < 0) {
                delta += span;
            }
            int index = (int) (delta / periodNs) % background.length;
            frame = background[index];
            backgroundFramesDelivered++;
        }
        target.recordSample(Sensor.TYPE_ACCELEROMETER, nowNs,
                SensorManager.SENSOR_STATUS_ACCURACY_HIGH, frame[0], frame[1], frame[2]);
        target.recordSample(Sensor.TYPE_GYROSCOPE, nowNs,
                SensorManager.SENSOR_STATUS_ACCURACY_HIGH, frame[3], frame[4], frame[5]);
        note(nowNs, frame, "background", "", -1);
        return 0L;
    }

    /**
     * Play one action's window, its first frame landing at {@code startElapsedNs}.
     *
     * <p>Each frame is posted for its own instant rather than the whole window
     * being written in a loop: the point of this app is that the timestamps a
     * detector sees are the instants the samples actually arrived, and a burst
     * delivered as fast as the CPU allows would have neither the planned rate
     * nor a plausible one.
     */
    synchronized int schedule(
            final float[][] frames,
            final double periodMs,
            final long startElapsedNs,
            final String bundleId) {
        if (handler == null) {
            return 0;
        }
        final long periodNs = Math.round(periodMs * 1_000_000.0);
        long offsetNs = Clocks.best(5).offsetNs();
        windowSpans.add(new long[] {startElapsedNs, startElapsedNs + frames.length * periodNs});
        // Spans that have passed cannot be re-entered and only make the check
        // above slower.
        long cutoff = SystemClock.elapsedRealtimeNanos() - 5_000_000_000L;
        java.util.Iterator<long[]> stale = windowSpans.iterator();
        while (stale.hasNext()) {
            if (stale.next()[1] < cutoff) {
                stale.remove();
            }
        }
        for (int i = 0; i < frames.length; i++) {
            final int index = i;
            final float[] frame = frames[i];
            final long dueElapsedNs = startElapsedNs + (long) i * periodNs;
            // Handler works on the uptime clock; the plan is on the sensor
            // clock. One conversion, with the offset this device measured.
            long dueUptimeMs = (dueElapsedNs - offsetNs) / 1_000_000L;
            handler.postAtTime(new Runnable() {
                @Override
                public void run() {
                    deliverInjected(frame, dueElapsedNs, bundleId, index);
                }
            }, dueUptimeMs);
        }
        return frames.length;
    }

    /** Cancel anything still queued; used between runs. */
    synchronized void clearScheduled() {
        if (handler != null) {
            handler.removeCallbacksAndMessages(null);
        }
        windowSpans.clear();
        // removeCallbacksAndMessages took the background's own tick with it.
        backgroundRunning = false;
        backgroundTick = null;
        if (MODE_INJECTED.equals(mode)) {
            startBackground();
        }
    }

    private void deliverInjected(float[] frame, long plannedElapsedNs, String bundleId, int index) {
        CaptureService target;
        synchronized (this) {
            target = service;
            if (!MODE_INJECTED.equals(mode) || target == null) {
                return;
            }
            injectedFrames++;
            double lateMs = (SystemClock.elapsedRealtimeNanos() - plannedElapsedNs) / 1e6;
            maxLatenessMs = Math.max(maxLatenessMs, lateMs);
            totalLatenessMs += lateMs;
            latenessSamples++;
        }
        // The planned instant is written, not "now": the whole method rests on
        // the touch and this frame sharing one timeline, and stamping arrival
        // time would replace that timeline with the scheduler's jitter.
        target.recordSample(Sensor.TYPE_ACCELEROMETER, plannedElapsedNs,
                SensorManager.SENSOR_STATUS_ACCURACY_HIGH, frame[0], frame[1], frame[2]);
        target.recordSample(Sensor.TYPE_GYROSCOPE, plannedElapsedNs,
                SensorManager.SENSOR_STATUS_ACCURACY_HIGH, frame[3], frame[4], frame[5]);
        note(plannedElapsedNs, frame, MODE_INJECTED, bundleId, index);
    }

    // -- the verification ring -----------------------------------------------

    synchronized void noteReal(int sensorType, long timestampNs, float x, float y, float z) {
        realFrames++;
        float[] values = new float[CHANNELS];
        if (sensorType == Sensor.TYPE_ACCELEROMETER) {
            values[0] = x; values[1] = y; values[2] = z;
        } else if (sensorType == Sensor.TYPE_GYROSCOPE) {
            values[3] = x; values[4] = y; values[5] = z;
        } else {
            return;
        }
        note(timestampNs, values, MODE_REAL, "", -1);
    }

    private synchronized void note(
            long timestampNs, float[] values, String origin, String bundleId, int frameIndex) {
        if (ring.size() >= RING_CAPACITY) {
            ring.removeFirst();
            droppedRows++;
        }
        ring.addLast(new Row(sequence++, timestampNs, values.clone(), origin, bundleId, frameIndex));
    }

    synchronized List<Row> recent() {
        return new ArrayList<>(ring);
    }

    synchronized void clearRows() {
        ring.clear();
        sequence = 0L;
        droppedRows = 0L;
        injectedFrames = 0L;
        realFrames = 0L;
        maxLatenessMs = 0.0;
        totalLatenessMs = 0.0;
        latenessSamples = 0L;
    }

    synchronized long droppedRows() {
        return droppedRows;
    }

    synchronized long injectedFrames() {
        return injectedFrames;
    }

    synchronized long realFrames() {
        return realFrames;
    }

    synchronized double maxLatenessMs() {
        return maxLatenessMs;
    }

    synchronized double meanLatenessMs() {
        return latenessSamples == 0 ? 0.0 : totalLatenessMs / latenessSamples;
    }

    synchronized int backgroundFrames() {
        return background.length;
    }

    synchronized long backgroundDelivered() {
        return backgroundFramesDelivered;
    }

    synchronized boolean backgroundRunning() {
        return backgroundRunning;
    }
}
