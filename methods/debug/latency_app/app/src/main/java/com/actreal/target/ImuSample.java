package com.actreal.target;

/** One inertial frame: accelerometer and gyroscope on a single timestamp. */
public final class ImuSample {

    /** Where the sample came from, recorded with every row so nothing is ambiguous. */
    public static final String ORIGIN_REAL = "real";
    public static final String ORIGIN_INJECTED = "injected";

    public final long timestampNs;
    public final float ax;
    public final float ay;
    public final float az;
    public final float gx;
    public final float gy;
    public final float gz;
    public final String origin;
    public final String bundleId;
    public final int frameIndex;

    public ImuSample(
            long timestampNs,
            float ax,
            float ay,
            float az,
            float gx,
            float gy,
            float gz,
            String origin,
            String bundleId,
            int frameIndex) {
        this.timestampNs = timestampNs;
        this.ax = ax;
        this.ay = ay;
        this.az = az;
        this.gx = gx;
        this.gy = gy;
        this.gz = gz;
        this.origin = origin;
        this.bundleId = bundleId == null ? "" : bundleId;
        this.frameIndex = frameIndex;
    }
}
