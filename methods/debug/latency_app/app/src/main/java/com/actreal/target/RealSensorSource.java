package com.actreal.target;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.HandlerThread;

/**
 * The phone's own accelerometer and gyroscope, paired into IMU frames.
 *
 * <p>The two sensors are separate streams with separate callbacks, so a frame
 * is emitted when the accelerometer reports and the most recent gyroscope
 * reading is attached to it. That is the same pairing the offline data was
 * built with; interpolating between gyroscope samples instead would invent
 * readings the device never produced.
 */
public final class RealSensorSource implements SensorEventListener {

    private final SensorManager sensorManager;
    private final ImuBus bus;
    private final HandlerThread thread;
    private final Handler handler;

    private volatile float gx;
    private volatile float gy;
    private volatile float gz;
    private volatile boolean haveGyro;
    private volatile boolean running;

    private long accelSamples;
    private long gyroSamples;
    private long firstTimestampNs;
    private long lastTimestampNs;

    public RealSensorSource(Context context, ImuBus bus) {
        this.sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        this.bus = bus;
        this.thread = new HandlerThread("actreal-real-sensors");
        this.thread.start();
        this.handler = new Handler(thread.getLooper());
    }

    public synchronized void start() {
        if (running) {
            return;
        }
        Sensor accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        Sensor gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        if (accel == null || gyro == null) {
            throw new IllegalStateException("accelerometer or gyroscope missing");
        }
        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_FASTEST, 0, handler);
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_FASTEST, 0, handler);
        running = true;
    }

    public synchronized void stop() {
        if (!running) {
            return;
        }
        sensorManager.unregisterListener(this);
        running = false;
    }

    public synchronized void release() {
        stop();
        thread.quitSafely();
    }

    public boolean isRunning() {
        return running;
    }

    /** Measured delivery rate of the accelerometer, in hertz. */
    public double measuredRateHz() {
        long span = lastTimestampNs - firstTimestampNs;
        if (accelSamples < 2 || span <= 0) {
            return 0.0;
        }
        return (accelSamples - 1) * 1e9 / span;
    }

    public long accelSampleCount() {
        return accelSamples;
    }

    public long gyroSampleCount() {
        return gyroSamples;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        int type = event.sensor.getType();
        if (type == Sensor.TYPE_GYROSCOPE) {
            gx = event.values[0];
            gy = event.values[1];
            gz = event.values[2];
            haveGyro = true;
            gyroSamples++;
            return;
        }
        if (type != Sensor.TYPE_ACCELEROMETER) {
            return;
        }
        accelSamples++;
        if (firstTimestampNs == 0L) {
            firstTimestampNs = event.timestamp;
        }
        lastTimestampNs = event.timestamp;
        if (!haveGyro) {
            // Until the gyroscope has reported once there is no frame to make.
            return;
        }
        bus.publish(
                new ImuSample(
                        event.timestamp,
                        event.values[0],
                        event.values[1],
                        event.values[2],
                        gx,
                        gy,
                        gz,
                        ImuSample.ORIGIN_REAL,
                        "",
                        -1));
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Recorded per sample by the log, not needed here.
    }
}
