package com.actreal.target;

import android.os.SystemClock;

import java.util.ArrayDeque;
import java.util.concurrent.locks.LockSupport;

/**
 * Inertial data the host planned, delivered on the host's timeline.
 *
 * <p>Two things this does that a naive replayer would not:
 *
 * <p><b>It keeps a background stream running.</b> A hand holding a phone is
 * never still, so between actions the app must still receive motion. Silence
 * between bursts is a stronger tell than any single gesture's shape, and the
 * padding regions of the generated windows are exactly the material for it --
 * the same hand, before and after the gesture.
 *
 * <p><b>It schedules against the sensor clock, not a delay.</b> Each frame has
 * an absolute {@link SystemClock#elapsedRealtimeNanos()} deadline, so error
 * does not accumulate across a window and the delivered timestamps are the
 * ones the plan asked for.
 */
public final class InjectedSensorSource {

    /** Spin rather than park inside this margin, where parking overshoots. */
    private static final long SPIN_MARGIN_NS = 1_500_000L;

    private static final class Segment {
        final long startElapsedNs;
        final long periodNs;
        final float[][] frames;
        final String bundleId;
        int next;

        Segment(long startElapsedNs, long periodNs, float[][] frames, String bundleId) {
            this.startElapsedNs = startElapsedNs;
            this.periodNs = periodNs;
            this.frames = frames;
            this.bundleId = bundleId;
        }

        long deadlineNs() {
            return startElapsedNs + next * periodNs;
        }

        boolean done() {
            return next >= frames.length;
        }
    }

    private final ImuBus bus;
    private final ArrayDeque<Segment> queue = new ArrayDeque<>();
    private final Object lock = new Object();

    private Thread worker;
    private volatile boolean running;

    private volatile float[][] background = new float[0][];
    private volatile long backgroundPeriodNs = 10_000_000L;
    private int backgroundCursor;
    private long backgroundNextNs;

    private volatile long deliveredFrames;
    private volatile long maxLatenessNs;
    private volatile long sumLatenessNs;

    public InjectedSensorSource(ImuBus bus) {
        this.bus = bus;
    }

    public void setBackground(float[][] frames, double periodMs) {
        background = frames == null ? new float[0][] : frames;
        backgroundPeriodNs = Math.max(1L, Math.round(periodMs * 1_000_000.0));
    }

    public boolean hasBackground() {
        return background.length > 0;
    }

    /** Queue one action's window, to start at an absolute sensor-clock time. */
    public void schedule(long startElapsedNs, double periodMs, float[][] frames, String bundleId) {
        if (frames == null || frames.length == 0) {
            return;
        }
        long periodNs = Math.max(1L, Math.round(periodMs * 1_000_000.0));
        synchronized (lock) {
            queue.addLast(new Segment(startElapsedNs, periodNs, frames, bundleId));
            lock.notifyAll();
        }
    }

    public int pendingSegments() {
        synchronized (lock) {
            return queue.size();
        }
    }

    public long deliveredFrames() {
        return deliveredFrames;
    }

    public double maxLatenessMs() {
        return maxLatenessNs / 1e6;
    }

    public double meanLatenessMs() {
        long n = deliveredFrames;
        return n == 0 ? 0.0 : (sumLatenessNs / (double) n) / 1e6;
    }

    public void resetStats() {
        deliveredFrames = 0;
        maxLatenessNs = 0;
        sumLatenessNs = 0;
    }

    public synchronized void start() {
        if (running) {
            return;
        }
        running = true;
        backgroundNextNs = SystemClock.elapsedRealtimeNanos();
        worker = new Thread(this::loop, "actreal-injected-imu");
        worker.setPriority(Thread.MAX_PRIORITY);
        worker.start();
    }

    public synchronized void stop() {
        running = false;
        synchronized (lock) {
            lock.notifyAll();
        }
        Thread t = worker;
        if (t != null) {
            try {
                t.join(500L);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }
        worker = null;
    }

    public void clearQueue() {
        synchronized (lock) {
            queue.clear();
        }
    }

    private void loop() {
        while (running) {
            Segment segment;
            synchronized (lock) {
                segment = queue.peekFirst();
                while (segment != null && segment.done()) {
                    queue.pollFirst();
                    segment = queue.peekFirst();
                }
            }

            long now = SystemClock.elapsedRealtimeNanos();
            if (segment != null) {
                long deadline = segment.deadlineNs();
                if (deadline - now > SPIN_MARGIN_NS && hasBackground()) {
                    // Room for a background frame before the action starts.
                    if (backgroundNextNs < deadline - backgroundPeriodNs) {
                        emitBackground();
                        continue;
                    }
                }
                sleepUntil(deadline);
                int index = segment.next++;
                emit(segment.frames[index], deadline, segment.bundleId, index);
                // The action owns the timeline while it runs, so the
                // background picks up after it rather than during it.
                backgroundNextNs = deadline + segment.periodNs;
                continue;
            }

            if (hasBackground()) {
                emitBackground();
            } else {
                synchronized (lock) {
                    try {
                        lock.wait(20L);
                    } catch (InterruptedException ignored) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
            }
        }
    }

    private void emitBackground() {
        float[][] frames = background;
        if (frames.length == 0) {
            return;
        }
        long deadline = backgroundNextNs;
        long now = SystemClock.elapsedRealtimeNanos();
        if (deadline < now - 5 * backgroundPeriodNs) {
            // Fell far behind (the app was suspended); restart rather than
            // burst out a backlog of stale frames.
            deadline = now;
        }
        sleepUntil(deadline);
        float[] frame = frames[backgroundCursor];
        backgroundCursor = (backgroundCursor + 1) % frames.length;
        emit(frame, deadline, "background", -1);
        backgroundNextNs = deadline + backgroundPeriodNs;
    }

    private void emit(float[] frame, long plannedNs, String bundleId, int index) {
        long actual = SystemClock.elapsedRealtimeNanos();
        long lateness = actual - plannedNs;
        if (lateness > maxLatenessNs) {
            maxLatenessNs = lateness;
        }
        sumLatenessNs += Math.abs(lateness);
        deliveredFrames++;
        // The timestamp handed to the app is the planned one: the plan is the
        // ground truth the touch was aligned against, and reporting the actual
        // wake-up instead would leak the scheduler's jitter into the data.
        bus.publish(
                new ImuSample(
                        plannedNs,
                        frame[0],
                        frame[1],
                        frame[2],
                        frame[3],
                        frame[4],
                        frame[5],
                        ImuSample.ORIGIN_INJECTED,
                        bundleId,
                        index));
    }

    private void sleepUntil(long deadlineNs) {
        while (running) {
            long remaining = deadlineNs - SystemClock.elapsedRealtimeNanos();
            if (remaining <= 0) {
                return;
            }
            if (remaining > SPIN_MARGIN_NS) {
                LockSupport.parkNanos(remaining - SPIN_MARGIN_NS);
            } else {
                Thread.onSpinWait();
            }
        }
    }
}
