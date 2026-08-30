package com.actreal.target;

import android.os.SystemClock;

/**
 * The two clocks an action is observed on, and the offset between them.
 *
 * <p>A MotionEvent is stamped with {@link SystemClock#uptimeMillis()}; a
 * SensorEvent is stamped with {@link SystemClock#elapsedRealtimeNanos()}. They
 * differ by however long the device has spent suspended, so a touch and an IMU
 * sample that happened together carry timestamps that do not look together
 * until one is converted. Every plan that crosses the two streams is expressed
 * on uptime, and converted here exactly once.
 */
public final class Clocks {

    private Clocks() {
    }

    /** Sample both clocks as close together as the platform allows. */
    public static long[] sampleBoth() {
        long before = SystemClock.elapsedRealtimeNanos();
        long uptimeMs = SystemClock.uptimeMillis();
        long after = SystemClock.elapsedRealtimeNanos();
        long elapsedNs = before / 2 + after / 2;
        // The read window is reported so the host can tell a real offset drift
        // from the noise of taking the two readings.
        return new long[] {uptimeMs, elapsedNs, after - before};
    }

    /**
     * Offset in nanoseconds such that {@code elapsedNs = uptimeMs * 1e6 + offset}.
     */
    public static long uptimeToElapsedOffsetNs() {
        long[] both = sampleBoth();
        return both[1] - both[0] * 1_000_000L;
    }

    public static long uptimeMsToElapsedNs(long uptimeMs, long offsetNs) {
        return uptimeMs * 1_000_000L + offsetNs;
    }

    public static long elapsedNsToUptimeMs(long elapsedNs, long offsetNs) {
        return (elapsedNs - offsetNs) / 1_000_000L;
    }
}
