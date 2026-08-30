package com.actreal.target;

import android.app.Activity;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.InputDevice;
import android.view.MotionEvent;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.LockSupport;

/**
 * Touch delivered inside the app, for when the input pipeline is closed to us.
 *
 * <p>The preferred route is the real one: something outside the app writes to
 * an input device and Android dispatches the result, which is what a detector
 * in a third-party app would see. That route needs {@code uinput} or root, and
 * whether a stock Pixel grants either is a question for the probe. This class
 * is the route that always works, because the target app is ours: it builds
 * the MotionEvents itself and dispatches them into its own view hierarchy.
 *
 * <p>What it gives up is the input pipeline's own metadata -- device id,
 * source flags, the driver's batching -- so a recording made this way is
 * marked as such and never passed off as the other kind. What it keeps is the
 * part that matters for alignment: every event carries the timestamp the plan
 * asked for, not the moment the thread happened to wake.
 */
public final class TouchInjector {

    private static final long SPIN_MARGIN_NS = 1_500_000L;

    public static final class Point {
        public final double tMs;
        public final float x;
        public final float y;
        public final float pressure;
        public final float size;
        public final int pointerId;
        public final String action;

        public Point(
                double tMs,
                float x,
                float y,
                float pressure,
                float size,
                int pointerId,
                String action) {
            this.tMs = tMs;
            this.x = x;
            this.y = y;
            this.pressure = pressure;
            this.size = size;
            this.pointerId = pointerId;
            this.action = action;
        }
    }

    private final Activity activity;
    private final Handler main = new Handler(Looper.getMainLooper());

    private Thread worker;
    private volatile boolean running;
    private volatile long dispatched;
    private volatile long maxLatenessNs;
    private volatile String lastError = "";

    public TouchInjector(Activity activity) {
        this.activity = activity;
    }

    public long dispatchedCount() {
        return dispatched;
    }

    public double maxLatenessMs() {
        return maxLatenessNs / 1e6;
    }

    public String lastError() {
        return lastError;
    }

    public void resetStats() {
        dispatched = 0;
        maxLatenessNs = 0;
        lastError = "";
    }

    public synchronized void cancel() {
        running = false;
        Thread t = worker;
        if (t != null) {
            t.interrupt();
        }
        worker = null;
    }

    /**
     * Play a gesture, with every event stamped at its planned uptime.
     *
     * @param startUptimeMs when the first point is due, on the MotionEvent clock
     */
    public synchronized void play(long startUptimeMs, List<Point> points) {
        if (points.isEmpty()) {
            return;
        }
        cancel();
        List<Point> copy = new ArrayList<>(points);
        running = true;
        worker =
                new Thread(
                        () -> {
                            try {
                                run(startUptimeMs, copy);
                            } catch (RuntimeException error) {
                                lastError = error.getClass().getSimpleName() + ": "
                                        + error.getMessage();
                            }
                        },
                        "actreal-touch");
        worker.setPriority(Thread.MAX_PRIORITY);
        worker.start();
    }

    private void run(long startUptimeMs, List<Point> points) {
        long downTimeMs = startUptimeMs + Math.round(points.get(0).tMs);
        for (Point point : points) {
            if (!running) {
                return;
            }
            long eventUptimeMs = startUptimeMs + Math.round(point.tMs);
            if (isDown(point.action)) {
                // Each contact starts a new pointer lifecycle, and a keystroke
                // is a run of them, so the down time is re-taken per press
                // rather than fixed once for the whole plan.
                downTimeMs = eventUptimeMs;
            }
            sleepUntilUptime(eventUptimeMs);
            dispatch(downTimeMs, eventUptimeMs, point);
        }
        running = false;
    }

    private static boolean isDown(String action) {
        return "DOWN".equals(action) || "ACTION_DOWN".equals(action);
    }

    private static int actionCode(String action) {
        switch (action) {
            case "DOWN":
            case "ACTION_DOWN":
                return MotionEvent.ACTION_DOWN;
            case "UP":
            case "ACTION_UP":
                return MotionEvent.ACTION_UP;
            case "CANCEL":
            case "ACTION_CANCEL":
                return MotionEvent.ACTION_CANCEL;
            default:
                return MotionEvent.ACTION_MOVE;
        }
    }

    private void dispatch(long downTimeMs, long eventUptimeMs, Point point) {
        MotionEvent.PointerProperties props = new MotionEvent.PointerProperties();
        props.id = point.pointerId;
        props.toolType = MotionEvent.TOOL_TYPE_FINGER;

        MotionEvent.PointerCoords coords = new MotionEvent.PointerCoords();
        coords.x = point.x;
        coords.y = point.y;
        coords.pressure = point.pressure;
        coords.size = point.size;
        coords.setAxisValue(MotionEvent.AXIS_TOUCH_MAJOR, point.size);

        MotionEvent event =
                MotionEvent.obtain(
                        downTimeMs,
                        eventUptimeMs,
                        actionCode(point.action),
                        1,
                        new MotionEvent.PointerProperties[] {props},
                        new MotionEvent.PointerCoords[] {coords},
                        0,
                        0,
                        1.0f,
                        1.0f,
                        0,
                        0,
                        InputDevice.SOURCE_TOUCHSCREEN,
                        0);

        long plannedNs = eventUptimeMs * 1_000_000L;
        long lateness = SystemClock.uptimeMillis() * 1_000_000L - plannedNs;
        if (lateness > maxLatenessNs) {
            maxLatenessNs = lateness;
        }
        dispatched++;
        main.post(
                () -> {
                    try {
                        activity.dispatchTouchEvent(event);
                    } finally {
                        event.recycle();
                    }
                });
    }

    private void sleepUntilUptime(long deadlineMs) {
        while (running) {
            long remainingNs = (deadlineMs - SystemClock.uptimeMillis()) * 1_000_000L;
            if (remainingNs <= 0) {
                return;
            }
            if (remainingNs > SPIN_MARGIN_NS) {
                LockSupport.parkNanos(remainingNs - SPIN_MARGIN_NS);
            } else {
                Thread.onSpinWait();
            }
        }
    }
}
