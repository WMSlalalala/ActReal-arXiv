package com.sensorworldmodel.collector;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.SystemClock;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

public final class CaptureService extends Service implements SensorEventListener {
    public static final String ACTION_START = "collector.START";
    public static final String ACTION_STOP = "collector.STOP";
    public static final String ACTION_PHASE = "collector.PHASE";
    public static final String EXTRA_TASK = "task";
    public static final String EXTRA_PHASE = "phase";
    public static final String EXTRA_POSTURE = "posture";
    public static final String EXTRA_RUN_ID = "run_id";

    private static final String CHANNEL_ID = "sensor_study_capture";
    private static final int NOTIFICATION_ID = 7401;
    private static final int REQUEST_PERIOD_US = 10_000;
    private static final long WAKE_LOCK_TIMEOUT_MS = 2L * 60L * 60L * 1000L;

    private SensorManager sensorManager;
    private ControlServer controlServer;
    private HandlerThread sensorThread;
    private Handler sensorHandler;
    private PowerManager.WakeLock wakeLock;
    private FileOutputStream imuStream;
    private BufferedWriter imuWriter;
    private int rowsSinceFlush = 0;
    private String currentTask = "";
    private String currentPhase = "";
    private String currentPosture = "";
    private String currentRunId = "";
    private long captureStartElapsedNs;
    private long accelerometerRows;
    private long gyroscopeRows;
    private long rotationVectorRows;
    private long accelerometerFirstNs = -1L;
    private long gyroscopeFirstNs = -1L;
    private long accelerometerLastNs = -1L;
    private long gyroscopeLastNs = -1L;
    private long accelerometerMaxGapNs;
    private long gyroscopeMaxGapNs;
    private long sensorWriteErrors;
    private static volatile boolean captureActiveInProcess;
    private static volatile boolean accelerometerSeenInProcess;
    private static volatile boolean gyroscopeSeenInProcess;
    private static volatile String activeRunIdInProcess = "";
    private static volatile boolean captureFinalizingInProcess;
    private volatile boolean stopping;

    public static boolean isCaptureActiveInProcess() {
        return captureActiveInProcess;
    }

    public static boolean isCaptureReadyInProcess(String runId) {
        return captureActiveInProcess
                && runId != null
                && runId.equals(activeRunIdInProcess)
                && accelerometerSeenInProcess
                && gyroscopeSeenInProcess;
    }

    public static boolean isCaptureRunActiveInProcess(String runId) {
        return captureActiveInProcess
                && runId != null
                && runId.equals(activeRunIdInProcess);
    }

    public static boolean isCaptureFinalizingInProcess() {
        return captureFinalizingInProcess;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        createNotificationChannel();
        sensorThread = new HandlerThread("study-sensors");
        sensorThread.start();
        sensorHandler = new Handler(sensorThread.getLooper());
        // Injected frames are posted onto the same handler the real sensors
        // are delivered on, so both suppliers reach the writer on one thread
        // and the recording cannot interleave them mid-row.
        ImuBus.get().attach(this, sensorManager, sensorHandler);
        controlServer = new ControlServer(this);
        controlServer.start();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent == null ? ACTION_START : intent.getAction();
        if (ACTION_STOP.equals(action)) {
            String requestedRunId = value(intent, EXTRA_RUN_ID, "");
            if (!requestedRunId.isEmpty()
                    && !requestedRunId.equals(activeRunIdInProcess)) {
                StudyStore.appendTaskEvent(
                        this,
                        currentTask.isEmpty() ? "capture" : currentTask,
                        "stale_stop_rejected",
                        "requested_run_id=" + requestedRunId,
                        currentPosture,
                        requestedRunId);
                if (!captureActiveInProcess) {
                    stopSelf();
                }
                return START_NOT_STICKY;
            }
            stopCapture("user_stop");
            return START_NOT_STICKY;
        }
        if (ACTION_PHASE.equals(action)) {
            String requestedRunId = value(intent, EXTRA_RUN_ID, "");
            if (!captureActiveInProcess || !StudyStore.isRecording(this)) {
                StudyStore.appendTaskEvent(
                        this,
                        currentTask.isEmpty() ? "capture" : currentTask,
                        "stale_phase_rejected",
                        "requested_run_id=" + requestedRunId,
                        currentPosture,
                        requestedRunId);
                stopSelf();
                return START_NOT_STICKY;
            }
            if (!requestedRunId.isEmpty()
                    && !requestedRunId.equals(activeRunIdInProcess)) {
                StudyStore.appendTaskEvent(
                        this,
                        currentTask,
                        "stale_phase_rejected",
                        "requested_run_id=" + requestedRunId,
                        currentPosture,
                        requestedRunId);
                return START_NOT_STICKY;
            }
            currentPhase = value(intent, EXTRA_PHASE, currentPhase);
            StudyStore.setPhase(this, currentPhase);
            StudyStore.appendTaskEvent(this, currentTask, "phase", currentPhase);
            updateNotification();
            return START_NOT_STICKY;
        }

        String requestedTask = value(intent, EXTRA_TASK, "study");
        String requestedPhase = value(intent, EXTRA_PHASE, "");
        String requestedPosture = value(intent, EXTRA_POSTURE, "sitting");
        String requestedRunId =
                value(intent, EXTRA_RUN_ID, StudyStore.newRunId(requestedTask));
        if (captureActiveInProcess
                || StudyStore.isRecording(this)
                || captureFinalizingInProcess
                || stopping) {
            StudyStore.appendTaskEvent(
                    this,
                    requestedTask,
                    "duplicate_start_rejected",
                    "requested_run_id=" + requestedRunId,
                    requestedPosture,
                    requestedRunId);
            return START_NOT_STICKY;
        }
        currentTask = requestedTask;
        currentPhase = requestedPhase;
        currentPosture = requestedPosture;
        currentRunId = requestedRunId;
        try {
            stopping = false;
            captureFinalizingInProcess = false;
            captureActiveInProcess = false;
            accelerometerSeenInProcess = false;
            gyroscopeSeenInProcess = false;
            activeRunIdInProcess = "";
            StudyStore.recordCaptureError(this, "");
            startForegroundCompatible(buildNotification());
            resetHealth();
            openWriter();
            registerSensors();
            acquireWakeLock();
            StudyStore.setRecording(
                    this,
                    true,
                    currentTask,
                    currentPhase,
                    currentPosture,
                    currentRunId);
            captureActiveInProcess = true;
            activeRunIdInProcess = currentRunId;
            StudyStore.appendTaskEvent(this, currentTask, "recording_started", currentPhase);
        } catch (IOException | RuntimeException error) {
            StudyStore.recordCaptureError(
                    this, error.getClass().getSimpleName() + ": " + error.getMessage());
            StudyStore.appendTaskEvent(
                    this, currentTask, "recording_error", error.getClass().getSimpleName());
            stopCapture("start_error");
            return START_NOT_STICKY;
        }
        return START_NOT_STICKY;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!ImuBus.get().acceptsReal()) {
            return;
        }
        try {
            handleSensorChanged(event);
        } catch (RuntimeException error) {
            StudyStore.recordCaptureError(
                    this, error.getClass().getSimpleName() + ": " + error.getMessage());
            StudyStore.appendTaskEvent(
                    this,
                    currentTask,
                    "sensor_runtime_error",
                    error.getClass().getSimpleName(),
                    currentPosture,
                    currentRunId);
            stopCapture("sensor_runtime_error");
        }
    }

    private void handleSensorChanged(SensorEvent event) {
        // The real sensors' way in.  It unpacks the event and hands the values
        // to the same sink the injected stream uses, so what reaches the CSV
        // has one shape and one code path whoever supplied it.
        float x = event.values.length > 0 ? event.values[0] : Float.NaN;
        float y = event.values.length > 1 ? event.values[1] : Float.NaN;
        float z = event.values.length > 2 ? event.values[2] : Float.NaN;
        ImuBus.get().noteReal(event.sensor.getType(), event.timestamp, x, y, z);
        recordSample(event.sensor.getType(), event.timestamp, event.accuracy, x, y, z);
    }

    /**
     * Write one inertial sample, whatever produced it.
     *
     * <p>Package-visible because {@link ImuBus} calls it for injected frames.
     * The row it writes is identical in every field to one a real sensor would
     * have produced -- deliberately so: this app is meant to be the same
     * instrument whether a person or an agent is operating it, and a column
     * recording which was which would put the experiment's answer in its own
     * data.
     */
    void recordSample(int sensorType, long timestampNs, int accuracy, float x, float y, float z) {
        if (!StudyStore.isRecording(this)) {
            return;
        }
        String sensor;
        if (sensorType == Sensor.TYPE_ACCELEROMETER) {
            sensor = "accelerometer";
        } else if (sensorType == Sensor.TYPE_GYROSCOPE) {
            sensor = "gyroscope";
        } else if (sensorType == Sensor.TYPE_ROTATION_VECTOR) {
            sensor = "rotation_vector";
        } else {
            return;
        }
        long elapsedNow = SystemClock.elapsedRealtimeNanos();
        long wallNow = System.currentTimeMillis();
        long wallForEvent = wallNow + Math.round((timestampNs - elapsedNow) / 1_000_000.0);
        int rotation = displayRotation();
        String line = join(
                StudyStore.SCHEMA,
                StudyStore.sessionId(this),
                StudyStore.profileId(this),
                currentTask,
                currentPhase,
                sensor,
                timestampNs,
                wallForEvent,
                accuracy,
                format(x),
                format(y),
                format(z),
                rotation,
                currentPosture,
                currentRunId
        );
        synchronized (this) {
            if (!StudyStore.isRecording(this) || imuWriter == null) {
                return;
            }
            try {
                imuWriter.write(line);
                imuWriter.write('\n');
                if ("accelerometer".equals(sensor)) {
                    updateAccelerometerHealth(timestampNs);
                } else if ("gyroscope".equals(sensor)) {
                    updateGyroscopeHealth(timestampNs);
                } else {
                    rotationVectorRows++;
                }
                rowsSinceFlush++;
                if (rowsSinceFlush >= 100) {
                    imuWriter.flush();
                    rowsSinceFlush = 0;
                }
                if ("accelerometer".equals(sensor)) {
                    accelerometerSeenInProcess = true;
                } else if ("gyroscope".equals(sensor)) {
                    gyroscopeSeenInProcess = true;
                }
            } catch (IOException error) {
                sensorWriteErrors++;
                StudyStore.recordWriteError(this);
                StudyStore.recordCaptureError(
                        this, error.getClass().getSimpleName() + ": " + error.getMessage());
                StudyStore.appendTaskEvent(
                        this,
                        currentTask,
                        "sensor_write_error",
                        error.getClass().getSimpleName(),
                        currentPosture,
                        currentRunId);
                stopCapture("sensor_write_error");
                return;
            }
        }
        if (!"rotation_vector".equals(sensor)
                && (accelerometerRows + gyroscopeRows) % 500L == 0L) {
            publishLiveHealth();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Accuracy is written with every sample.
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        stopCapture("service_destroyed");
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
        ImuBus.get().detach(this);
        if (sensorThread != null) {
            sensorThread.quitSafely();
            sensorThread = null;
        }
        super.onDestroy();
    }

    /** How {@link ImuBus} restores the real supplier when it releases the bus. */
    void registerSensorsForBus() {
        if (sensorManager != null) {
            registerSensors();
        }
    }

    private void registerSensors() {
        sensorManager.unregisterListener(this);
        register(Sensor.TYPE_ACCELEROMETER, true);
        register(Sensor.TYPE_GYROSCOPE, true);
        register(Sensor.TYPE_ROTATION_VECTOR, false);
    }

    private void register(int type, boolean required) {
        Sensor sensor = sensorManager.getDefaultSensor(type);
        if (sensor == null) {
            if (required) {
                throw new IllegalStateException("Required sensor missing: " + type);
            }
            return;
        }
        boolean registered = sensorManager.registerListener(
                this, sensor, REQUEST_PERIOD_US, 0, sensorHandler);
        if (!registered) {
            if (required) {
                throw new IllegalStateException(
                        "Required sensor registration failed: " + type);
            }
            return;
        }
        String header = "schema,session_id,sensor_type,name,vendor,version,resolution," +
                "maximum_range,min_delay_us,max_delay_us,power_ma,posture,run_id\n";
        StudyStore.appendCsv(
                this,
                "sensor_info.csv",
                header,
                StudyStore.SCHEMA,
                StudyStore.sessionId(this),
                sensor.getType(),
                sensor.getName(),
                sensor.getVendor(),
                sensor.getVersion(),
                sensor.getResolution(),
                sensor.getMaximumRange(),
                sensor.getMinDelay(),
                sensor.getMaxDelay(),
                sensor.getPower(),
                currentPosture,
                currentRunId
        );
    }

    private synchronized void openWriter() throws IOException {
        closeWriter();
        File directory = StudyStore.currentSessionDir(this);
        if (directory == null || !directory.isDirectory()) {
            throw new IOException("Create a session before recording");
        }
        File file = new File(directory, "imu.csv");
        imuStream = new FileOutputStream(file, true);
        imuWriter = new BufferedWriter(new OutputStreamWriter(
                imuStream, StandardCharsets.UTF_8), 128 * 1024);
    }

    private synchronized void closeWriter() {
        if (imuWriter == null) {
            return;
        }
        try {
            imuWriter.flush();
            if (imuStream != null) {
                imuStream.getFD().sync();
            }
            imuWriter.close();
        } catch (IOException error) {
            sensorWriteErrors++;
            StudyStore.recordWriteError(this);
            StudyStore.recordCaptureError(
                    this, error.getClass().getSimpleName() + ": " + error.getMessage());
        }
        imuWriter = null;
        imuStream = null;
        rowsSinceFlush = 0;
    }

    private synchronized void stopCapture(String reason) {
        if (stopping) {
            return;
        }
        stopping = true;
        boolean wasRecording = StudyStore.isRecording(this);
        if (wasRecording) {
            captureFinalizingInProcess = true;
        }
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
        StudyStore.setRecording(this, false, "", "", "", "");
        captureActiveInProcess = false;
        accelerometerSeenInProcess = false;
        gyroscopeSeenInProcess = false;
        activeRunIdInProcess = "";
        closeWriter();
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
        }
        if (wasRecording) {
            StudyStore.appendTaskEvent(
                    this,
                    currentTask,
                    "recording_stopped",
                    reason,
                    currentPosture,
                    currentRunId);
            appendCaptureHealth(reason);
        }
        stopForeground(STOP_FOREGROUND_REMOVE);
        if (!wasRecording) {
            stopSelf();
            return;
        }
        new Thread(() -> {
            try {
                StudyStore.flushAndClosePendingWrites(this);
            } catch (IOException ignored) {
                // flushAndClosePendingWrites records the write error counter.
            } finally {
                stopSelf();
                captureFinalizingInProcess = false;
            }
        }, "capture-finalizer").start();
    }

    private void acquireWakeLock() {
        if (wakeLock == null) {
            PowerManager manager = (PowerManager) getSystemService(POWER_SERVICE);
            wakeLock = manager.newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK, "SensorStudy:Capture");
            wakeLock.setReferenceCounted(false);
        }
        if (!wakeLock.isHeld()) {
            wakeLock.acquire(WAKE_LOCK_TIMEOUT_MS);
        }
    }

    private void createNotificationChannel() {
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Sensor study recording",
                NotificationManager.IMPORTANCE_LOW);
        channel.setDescription("Visible indicator while a user-started study task records IMU.");
        getSystemService(NotificationManager.class).createNotificationChannel(channel);
    }

    private Notification buildNotification() {
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent openPending = PendingIntent.getActivity(
                this, 1, open, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        Intent stop = new Intent(this, CaptureService.class).setAction(ACTION_STOP);
        PendingIntent stopPending = PendingIntent.getService(
                this, 2, stop, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        String detail = currentTask;
        if (!currentPhase.isEmpty()) {
            detail += " · " + currentPhase;
        }
        float accHz = StudyStore.prefs(this).getFloat(StudyStore.KEY_LIVE_ACC_HZ, 0f);
        float gyroHz = StudyStore.prefs(this).getFloat(StudyStore.KEY_LIVE_GYRO_HZ, 0f);
        if (accHz > 0f && gyroHz > 0f) {
            detail += String.format(Locale.US, " · A/G %.0f/%.0fHz", accHz, gyroHz);
        }
        return new Notification.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.presence_online)
                .setContentTitle("Recording IMU")
                .setContentText(detail)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setContentIntent(openPending)
                .addAction(new Notification.Action.Builder(
                        null, "Stop", stopPending).build())
                .build();
    }

    private void startForegroundCompatible(Notification notification) {
        startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
    }

    private void updateNotification() {
        getSystemService(NotificationManager.class)
                .notify(NOTIFICATION_ID, buildNotification());
    }

    private int displayRotation() {
        return StudyStore.displayRotation(this);
    }

    private static String value(Intent intent, String name, String fallback) {
        String value = intent == null ? null : intent.getStringExtra(name);
        return value == null ? fallback : value;
    }

    private static String format(float value) {
        return String.format(Locale.US, "%.9g", value);
    }

    private static String join(Object... values) {
        StringBuilder builder = new StringBuilder();
        for (int index = 0; index < values.length; index++) {
            if (index > 0) {
                builder.append(',');
            }
            builder.append(StudyStore.csv(values[index]));
        }
        return builder.toString();
    }

    private void resetHealth() {
        captureStartElapsedNs = SystemClock.elapsedRealtimeNanos();
        accelerometerRows = 0L;
        gyroscopeRows = 0L;
        rotationVectorRows = 0L;
        accelerometerFirstNs = -1L;
        gyroscopeFirstNs = -1L;
        accelerometerLastNs = -1L;
        gyroscopeLastNs = -1L;
        accelerometerMaxGapNs = 0L;
        gyroscopeMaxGapNs = 0L;
        sensorWriteErrors = 0L;
        StudyStore.prefs(this).edit()
                .putFloat(StudyStore.KEY_LIVE_ACC_HZ, 0f)
                .putFloat(StudyStore.KEY_LIVE_GYRO_HZ, 0f)
                .putFloat(StudyStore.KEY_LIVE_ACC_GAP_MS, 0f)
                .putFloat(StudyStore.KEY_LIVE_GYRO_GAP_MS, 0f)
                .apply();
    }

    private void updateAccelerometerHealth(long timestamp) {
        if (accelerometerFirstNs < 0) {
            accelerometerFirstNs = timestamp;
        }
        if (accelerometerLastNs >= 0) {
            accelerometerMaxGapNs = Math.max(
                    accelerometerMaxGapNs, timestamp - accelerometerLastNs);
        }
        accelerometerLastNs = timestamp;
        accelerometerRows++;
    }

    private void updateGyroscopeHealth(long timestamp) {
        if (gyroscopeFirstNs < 0) {
            gyroscopeFirstNs = timestamp;
        }
        if (gyroscopeLastNs >= 0) {
            gyroscopeMaxGapNs = Math.max(
                    gyroscopeMaxGapNs, timestamp - gyroscopeLastNs);
        }
        gyroscopeLastNs = timestamp;
        gyroscopeRows++;
    }

    private void appendCaptureHealth(String reason) {
        publishLiveHealth();
        long endNs = SystemClock.elapsedRealtimeNanos();
        double durationMs = Math.max(0L, endNs - captureStartElapsedNs) / 1_000_000.0;
        StudyStore.appendCsv(
                this,
                "capture_health.csv",
                StudyStore.CAPTURE_HEALTH_HEADER,
                StudyStore.SCHEMA,
                StudyStore.sessionId(this),
                StudyStore.profileId(this),
                currentTask,
                currentPhase,
                captureStartElapsedNs,
                endNs,
                durationMs,
                accelerometerRows,
                gyroscopeRows,
                rotationVectorRows,
                effectiveHz(accelerometerRows, accelerometerFirstNs, accelerometerLastNs),
                effectiveHz(gyroscopeRows, gyroscopeFirstNs, gyroscopeLastNs),
                accelerometerMaxGapNs / 1_000_000.0,
                gyroscopeMaxGapNs / 1_000_000.0,
                sensorWriteErrors,
                reason,
                currentPosture,
                currentRunId
        );
    }

    private static double effectiveHz(long rows, long firstNs, long lastNs) {
        if (rows < 2 || firstNs < 0 || lastNs <= firstNs) {
            return 0.0;
        }
        return (rows - 1) * 1_000_000_000.0 / (lastNs - firstNs);
    }

    private void publishLiveHealth() {
        StudyStore.prefs(this).edit()
                .putFloat(
                        StudyStore.KEY_LIVE_ACC_HZ,
                        (float) effectiveHz(
                                accelerometerRows,
                                accelerometerFirstNs,
                                accelerometerLastNs))
                .putFloat(
                        StudyStore.KEY_LIVE_GYRO_HZ,
                        (float) effectiveHz(
                                gyroscopeRows,
                                gyroscopeFirstNs,
                                gyroscopeLastNs))
                .putFloat(
                        StudyStore.KEY_LIVE_ACC_GAP_MS,
                        (float) (accelerometerMaxGapNs / 1_000_000.0))
                .putFloat(
                        StudyStore.KEY_LIVE_GYRO_GAP_MS,
                        (float) (gyroscopeMaxGapNs / 1_000_000.0))
                .apply();
        updateNotification();
    }
}
