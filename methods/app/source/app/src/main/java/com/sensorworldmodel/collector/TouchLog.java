package com.sensorworldmodel.collector;

import android.view.MotionEvent;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * The MotionEvents this app received, kept in memory for the control channel.
 *
 * <p>The CSV already has them; this exists because the CSV is the study's
 * output and the control channel is how a run is checked while it happens.
 * Without it there is no way to ask this app whether a synthesised touch
 * actually arrived -- only whether the file grew afterwards.
 *
 * <p><b>Historical samples are kept.</b> Android coalesces rapid MOVEs into one
 * MotionEvent carrying earlier samples as history, and it also resamples toward
 * the display's frame boundaries, inventing samples nobody sent. Dropping the
 * historical ones would report a thirteen-point scroll that arrived whole as
 * five points delivered; keeping them, and marking which is which, lets the
 * host tell our samples from the framework's.
 *
 * <p>Two timestamps per row, because Android has two clocks and they are not
 * interchangeable: {@code getEventTimeNanos} is on the same clock as
 * {@code uptimeMillis}, while a SensorEvent is on {@code elapsedRealtimeNanos}.
 * Both are recorded so the host never has to guess which it is holding.
 */
final class TouchLog {

    private static final int CAPACITY = 20000;
    private static final TouchLog INSTANCE = new TouchLog();

    static TouchLog get() {
        return INSTANCE;
    }

    private TouchLog() {}

    static final class Row {
        final long seq;
        final long uptimeNs;
        final long elapsedNs;
        final String action;
        final boolean historical;
        final int pointerCount;
        final int pointerId;
        final float x;
        final float y;
        final float pressure;
        final float size;
        final int deviceId;
        final int source;

        Row(long seq, long uptimeNs, long elapsedNs, String action, boolean historical,
            int pointerCount, int pointerId, float x, float y, float pressure, float size,
            int deviceId, int source) {
            this.seq = seq;
            this.uptimeNs = uptimeNs;
            this.elapsedNs = elapsedNs;
            this.action = action;
            this.historical = historical;
            this.pointerCount = pointerCount;
            this.pointerId = pointerId;
            this.x = x;
            this.y = y;
            this.pressure = pressure;
            this.size = size;
            this.deviceId = deviceId;
            this.source = source;
        }
    }

    private final Deque<Row> ring = new ArrayDeque<>();
    private long sequence = 0L;
    private long dropped = 0L;

    synchronized void note(
            MotionEvent event,
            int pointer,
            long uptimeNs,
            long elapsedNs,
            float x,
            float y,
            float pressure,
            float size,
            String action,
            boolean historical) {
        if (ring.size() >= CAPACITY) {
            ring.removeFirst();
            dropped++;
        }
        ring.addLast(new Row(
                sequence++,
                uptimeNs,
                elapsedNs,
                action,
                historical,
                event.getPointerCount(),
                event.getPointerId(pointer),
                x,
                y,
                pressure,
                size,
                // The provenance an in-app dispatch cannot forge, and the whole
                // reason the touch half goes through the input pipeline: a real
                // device id and a real source.
                event.getDeviceId(),
                event.getSource()));
    }

    synchronized List<Row> recent() {
        return new ArrayList<>(ring);
    }

    synchronized void clear() {
        ring.clear();
        sequence = 0L;
        dropped = 0L;
    }

    synchronized long dropped() {
        return dropped;
    }
}
