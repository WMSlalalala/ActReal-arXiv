package com.sensorworldmodel.collector;

import android.os.SystemClock;

/**
 * The offset between the two Android clocks, measured rather than assumed.
 *
 * <p>A MotionEvent is stamped with {@link SystemClock#uptimeMillis} and a
 * SensorEvent with {@link SystemClock#elapsedRealtimeNanos}. The first stops
 * while the device is suspended and the second does not, so the difference
 * between them is however long this phone has slept since it booted: it differs
 * per device, per boot, and grows during a session. On the Pixel 10 used for
 * this study it was over 1,062,622 seconds -- twelve days of accumulated sleep.
 *
 * <p>Nothing can read both clocks at the same instant, so this reads one, then
 * the other, then the first again, and reports how wide that window was. The
 * host keeps whichever sample was taken through the narrowest window, the same
 * way a time protocol does, instead of averaging the noise in.
 */
final class Clocks {

    private Clocks() {}

    /** One reading of both clocks, plus the width of the read. */
    static final class Sample {
        final long uptimeMs;
        final long elapsedNs;
        final long readWindowNs;

        Sample(long uptimeMs, long elapsedNs, long readWindowNs) {
            this.uptimeMs = uptimeMs;
            this.elapsedNs = elapsedNs;
            this.readWindowNs = readWindowNs;
        }

        /** elapsed - uptime: the quantity that differs between the clocks. */
        long offsetNs() {
            return elapsedNs - uptimeMs * 1_000_000L;
        }
    }

    static Sample read() {
        long before = SystemClock.elapsedRealtimeNanos();
        long uptimeMs = SystemClock.uptimeMillis();
        long after = SystemClock.elapsedRealtimeNanos();
        return new Sample(uptimeMs, (before + after) / 2L, after - before);
    }

    /** Keep the reading taken through the narrowest window. */
    static Sample best(int samples) {
        Sample best = read();
        for (int i = 1; i < samples; i++) {
            Sample candidate = read();
            if (candidate.readWindowNs < best.readWindowNs) {
                best = candidate;
            }
        }
        return best;
    }

    /** Turn a touch-clock instant into a sensor-clock one, on this device. */
    static long elapsedNsAtUptimeMs(long uptimeMs) {
        return uptimeMs * 1_000_000L + best(5).offsetNs();
    }
}
