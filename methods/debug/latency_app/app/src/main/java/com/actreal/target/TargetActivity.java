package com.actreal.target;

import android.app.Activity;
import android.graphics.RectF;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.view.MotionEvent;
import android.view.WindowManager;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.Locale;

/**
 * The app under test: it receives touch through Android and inertia through the bus.
 *
 * <p>The two halves of an action arrive by different routes -- MotionEvents
 * come up the normal input pipeline from whatever injected them, IMU frames
 * come from {@link ImuBus} -- and this activity is where they are both
 * observed, exactly as an app with a behavioural detector in it would observe
 * them.
 */
public final class TargetActivity extends Activity implements ControlServer.Host {

    private static final int MAX_ROWS = 400_000;

    private TargetView view;
    private Recorder recorder;
    private ImuBus bus;
    private RealSensorSource realSource;
    private InjectedSensorSource injectedSource;
    private TouchInjector touchInjector;
    private ControlServer control;

    private final Handler ui = new Handler(Looper.getMainLooper());
    private final Runnable statusTick = this::refreshStatus;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        view = new TargetView(this);
        setContentView(view);

        recorder = new Recorder(MAX_ROWS);
        bus = new ImuBus();
        bus.addListener(recorder);

        realSource = new RealSensorSource(this, bus);
        injectedSource = new InjectedSensorSource(bus);
        touchInjector = new TouchInjector(this);

        control = new ControlServer(ControlServer.DEFAULT_PORT, this);
        control.start();
    }

    @Override
    protected void onResume() {
        super.onResume();
        recorder.refreshClockOffset();
        realSource.start();
        injectedSource.start();
        ui.post(statusTick);
    }

    @Override
    protected void onPause() {
        ui.removeCallbacks(statusTick);
        touchInjector.cancel();
        injectedSource.stop();
        realSource.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        control.stop();
        realSource.release();
        super.onDestroy();
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        // Observed without being consumed, so the event still reaches the view
        // hierarchy exactly as it would in any other app.
        recorder.observe(event);
        view.addTouch(event.getX(), event.getY(), event.getPressure());
        return super.dispatchTouchEvent(event);
    }

    private void refreshStatus() {
        RectF usable = view.usableRect();
        String mode = bus.mode() == ImuBus.MODE_INJECTED ? "INJECTED" : "real";
        view.setStatus(
                String.format(
                        Locale.US,
                        "ActReal target  port %d  conn %d\n"
                                + "%dx%d  usable y %.0f..%.0f\n"
                                + "imu mode %s   delivered %d\n"
                                + "touch rows %d   imu rows %d\n"
                                + "injected %d frames  late max %.2f ms  mean %.2f ms\n"
                                + "queued %d  real accel %d gyro %d @ %.1f Hz",
                        control.port(),
                        control.connectionCount(),
                        view.getWidth(),
                        view.getHeight(),
                        usable.top,
                        usable.bottom,
                        mode,
                        bus.deliveredCount(),
                        recorder.touchRowCount(),
                        recorder.imuRowCount(),
                        injectedSource.deliveredFrames(),
                        injectedSource.maxLatenessMs(),
                        injectedSource.meanLatenessMs(),
                        injectedSource.pendingSegments(),
                        realSource.accelSampleCount(),
                        realSource.gyroSampleCount(),
                        realSource.measuredRateHz()));
        ui.postDelayed(statusTick, 250L);
    }

    // --- ControlServer.Host ---------------------------------------------------

    @Override
    public JSONObject describe() throws JSONException {
        JSONObject out = new JSONObject();
        DisplayMetrics metrics = getResources().getDisplayMetrics();
        out.put("model", Build.MODEL);
        out.put("manufacturer", Build.MANUFACTURER);
        out.put("android_release", Build.VERSION.RELEASE);
        out.put("sdk_int", Build.VERSION.SDK_INT);
        out.put("display_w", view.getWidth());
        out.put("display_h", view.getHeight());
        out.put("metrics_w", metrics.widthPixels);
        out.put("metrics_h", metrics.heightPixels);
        out.put("density_dpi", metrics.densityDpi);

        RectF usable = view.usableRect();
        JSONObject rect = new JSONObject();
        rect.put("left", usable.left);
        rect.put("top", usable.top);
        rect.put("right", usable.right);
        rect.put("bottom", usable.bottom);
        out.put("usable_rect", rect);
        out.put("source_w", TargetView.SOURCE_W);
        out.put("source_h", TargetView.SOURCE_H);

        SensorManager sm = (SensorManager) getSystemService(SENSOR_SERVICE);
        out.put("accel", describeSensor(sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)));
        out.put("gyro", describeSensor(sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)));
        out.put("real_rate_hz", realSource.measuredRateHz());
        return out;
    }

    private static JSONObject describeSensor(Sensor sensor) throws JSONException {
        JSONObject out = new JSONObject();
        if (sensor == null) {
            out.put("present", false);
            return out;
        }
        out.put("present", true);
        out.put("name", sensor.getName());
        out.put("vendor", sensor.getVendor());
        out.put("resolution", sensor.getResolution());
        out.put("max_range", sensor.getMaximumRange());
        out.put("min_delay_us", sensor.getMinDelay());
        out.put("max_delay_us", sensor.getMaxDelay());
        return out;
    }

    @Override
    public void setImuMode(int mode) {
        bus.setMode(mode);
        if (mode == ImuBus.MODE_INJECTED) {
            // The real sensors are stopped, not merely ignored: leaving them
            // registered would keep the phone's own motion arriving alongside
            // the planned stream, and the two hands would be spliced together.
            realSource.stop();
        } else {
            injectedSource.clearQueue();
            realSource.start();
        }
    }

    @Override
    public int imuMode() {
        return bus.mode();
    }

    @Override
    public void scheduleImu(
            long startElapsedNs, double periodMs, float[][] frames, String bundleId) {
        injectedSource.schedule(startElapsedNs, periodMs, frames, bundleId);
    }

    @Override
    public void setBackground(float[][] frames, double periodMs) {
        injectedSource.setBackground(frames, periodMs);
    }

    @Override
    public void playTouch(long startUptimeMs, java.util.List<TouchInjector.Point> points) {
        touchInjector.play(startUptimeMs, points);
    }

    @Override
    public void clearRecordings() {
        recorder.clear();
        injectedSource.resetStats();
        touchInjector.resetStats();
        ui.post(view::clearTrail);
    }

    @Override
    public void clearScheduled() {
        injectedSource.clearQueue();
    }

    @Override
    public JSONObject dump(boolean includeRows) throws JSONException {
        return recorder.dump(includeRows);
    }

    @Override
    public JSONObject stats() throws JSONException {
        JSONObject out = new JSONObject();
        out.put("imu_mode", bus.mode() == ImuBus.MODE_INJECTED ? "injected" : "real");
        out.put("bus_delivered", bus.deliveredCount());
        out.put("injected_frames", injectedSource.deliveredFrames());
        out.put("injected_max_lateness_ms", injectedSource.maxLatenessMs());
        out.put("injected_mean_lateness_ms", injectedSource.meanLatenessMs());
        out.put("injected_pending_segments", injectedSource.pendingSegments());
        out.put("background_frames", injectedSource.hasBackground());
        out.put("real_accel_samples", realSource.accelSampleCount());
        out.put("real_gyro_samples", realSource.gyroSampleCount());
        out.put("real_rate_hz", realSource.measuredRateHz());
        out.put("touch_rows", recorder.touchRowCount());
        out.put("imu_rows", recorder.imuRowCount());
        out.put("touch_dropped", recorder.touchDropped());
        out.put("imu_dropped", recorder.imuDropped());
        out.put("touch_dispatched", touchInjector.dispatchedCount());
        out.put("touch_max_lateness_ms", touchInjector.maxLatenessMs());
        out.put("touch_error", touchInjector.lastError());
        out.put("control_error", control.lastError());
        return out;
    }
}
