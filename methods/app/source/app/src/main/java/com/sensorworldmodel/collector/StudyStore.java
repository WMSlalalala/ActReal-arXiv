package com.sensorworldmodel.collector;

import android.content.Context;
import android.content.SharedPreferences;
import android.hardware.display.DisplayManager;
import android.os.Build;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.view.Display;
import android.view.Surface;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;
import java.util.UUID;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public final class StudyStore {
    public static final String SCHEMA = "personal_android_imu_touch_v2";
    public static final String PREFS = "study_state";
    public static final String KEY_SESSION = "session_id";
    public static final String KEY_PROFILE = "profile_id";
    public static final String KEY_RECORDING = "recording";
    public static final String KEY_TASK = "active_task";
    public static final String KEY_PHASE = "active_phase";
    public static final String KEY_POSTURE = "active_posture";
    public static final String KEY_RUN_ID = "active_run_id";
    public static final String KEY_WRITE_ERRORS = "write_errors";
    public static final String KEY_DROPPED_ROWS = "dropped_rows";
    public static final String KEY_LAST_CAPTURE_ERROR = "last_capture_error";
    public static final String KEY_LIVE_ACC_HZ = "live_acc_hz";
    public static final String KEY_LIVE_GYRO_HZ = "live_gyro_hz";
    public static final String KEY_LIVE_ACC_GAP_MS = "live_acc_gap_ms";
    public static final String KEY_LIVE_GYRO_GAP_MS = "live_gyro_gap_ms";
    public static final String KEY_COMPLETED_PREFIX = "completed_run.";

    public static final String IMU_HEADER =
            "schema,session_id,profile_id,task,phase,sensor,event_elapsed_ns,event_wall_ms," +
            "accuracy,x,y,z,display_rotation,posture,run_id\n";
    public static final String TOUCH_HEADER =
            "schema,session_id,profile_id,task,phase,event_id,labeled_action," +
            "event_elapsed_ns,event_wall_ms,motion_action,pointer_count,pointer_index," +
            "pointer_id,x_px,y_px,raw_x_px,raw_y_px,pressure,size,display_rotation," +
            "display_width_px,display_height_px,density_dpi,posture,run_id\n";
    public static final String EVENT_HEADER =
            "schema,session_id,profile_id,task,event_id,source,action,start_elapsed_ns," +
            "end_elapsed_ns,start_wall_ms,end_wall_ms,duration_ms,pointer_count," +
            "condition_quality,xy_source,x_start,y_start,x_end,y_end,bounds_left," +
            "bounds_top,bounds_right,bounds_bottom,orientation_id,n_keys,n_letters," +
            "observable_fields,notes,display_width_px,display_height_px,density_dpi," +
            "posture,run_id\n";
    public static final String TASK_HEADER =
            "schema,session_id,profile_id,event_elapsed_ns,event_wall_ms,task,event,details," +
            "posture,run_id\n";
    public static final String KEYSTROKE_HEADER =
            "schema,session_id,profile_id,task,phase,event_id,event_elapsed_ns," +
            "event_wall_ms,before_count,after_count,added_count,removed_count,posture,run_id\n";
    public static final String CAPTURE_HEALTH_HEADER =
            "schema,session_id,profile_id,task,phase,start_elapsed_ns,end_elapsed_ns," +
            "duration_ms,accelerometer_rows,gyroscope_rows,rotation_vector_rows," +
            "accelerometer_effective_hz,gyroscope_effective_hz,accelerometer_max_gap_ms," +
            "gyroscope_max_gap_ms,sensor_write_errors,stop_reason,posture,run_id\n";
    private static final int MAX_PENDING_ROWS = 50_000;
    private static final int FLUSH_EVERY_ROWS = 32;
    private static final Map<String, WriterState> CSV_WRITERS = new HashMap<>();
    private static final Object COUNTER_LOCK = new Object();
    private static final ThreadPoolExecutor CSV_EXECUTOR = new ThreadPoolExecutor(
            1,
            1,
            0L,
            TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<>(MAX_PENDING_ROWS),
            runnable -> {
                Thread thread = new Thread(runnable, "study-csv-writer");
                thread.setDaemon(true);
                return thread;
            });

    private StudyStore() {}

    private static final class WriterState {
        final FileOutputStream stream;
        final BufferedWriter writer;
        int rowsSinceFlush;

        WriterState(File path) throws IOException {
            stream = new FileOutputStream(path, true);
            writer = new BufferedWriter(new OutputStreamWriter(
                    stream, StandardCharsets.UTF_8), 128 * 1024);
        }

        void write(String line) throws IOException {
            writer.write(line);
            writer.write('\n');
            rowsSinceFlush++;
            if (rowsSinceFlush >= FLUSH_EVERY_ROWS) {
                writer.flush();
                rowsSinceFlush = 0;
            }
        }

        void closeAndSync() throws IOException {
            IOException failure = null;
            try {
                writer.flush();
                stream.getFD().sync();
            } catch (IOException error) {
                failure = error;
            }
            try {
                writer.close();
            } catch (IOException error) {
                if (failure == null) {
                    failure = error;
                }
            }
            if (failure != null) {
                throw failure;
            }
        }
    }

    public static SharedPreferences prefs(Context context) {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    public static synchronized String createSession(Context context, String profileId)
            throws IOException, JSONException {
        flushAndClosePendingWrites(context);
        String safeProfile = sanitize(profileId);
        if (safeProfile.isEmpty()) {
            safeProfile = "P001";
        }
        String stamp = utcNow().replace(":", "").replace("-", "");
        String sessionId = stamp + "_" + UUID.randomUUID().toString().substring(0, 8);
        SharedPreferences.Editor editor = prefs(context).edit();
        for (String key : prefs(context).getAll().keySet()) {
            if (key.startsWith(KEY_COMPLETED_PREFIX)) {
                editor.remove(key);
            }
        }
        editor.putString(KEY_SESSION, sessionId);
        editor.putString(KEY_PROFILE, safeProfile);
        editor.putBoolean(KEY_RECORDING, false);
        editor.putString(KEY_TASK, "");
        editor.putString(KEY_PHASE, "");
        editor.putString(KEY_POSTURE, "");
        editor.putString(KEY_RUN_ID, "");
        editor.putLong(KEY_WRITE_ERRORS, 0L);
        editor.putLong(KEY_DROPPED_ROWS, 0L);
        editor.putString(KEY_LAST_CAPTURE_ERROR, "");
        editor.apply();

        File directory = sessionDir(context, sessionId);
        if (!directory.mkdirs() && !directory.isDirectory()) {
            throw new IOException("Cannot create session directory " + directory);
        }
        ensureCsv(directory, "imu.csv", IMU_HEADER);
        ensureCsv(directory, "touch.csv", TOUCH_HEADER);
        ensureCsv(directory, "events.csv", EVENT_HEADER);
        ensureCsv(directory, "task_events.csv", TASK_HEADER);
        ensureCsv(directory, "keystroke.csv", KEYSTROKE_HEADER);
        ensureCsv(directory, "capture_health.csv", CAPTURE_HEALTH_HEADER);

        JSONObject manifest = new JSONObject();
        manifest.put("schema", SCHEMA);
        manifest.put("session_id", sessionId);
        manifest.put("profile_id", safeProfile);
        manifest.put("created_utc", utcNow());
        manifest.put("created_wall_ms", System.currentTimeMillis());
        manifest.put("created_elapsed_ns", SystemClock.elapsedRealtimeNanos());
        manifest.put("manufacturer", Build.MANUFACTURER);
        manifest.put("brand", Build.BRAND);
        manifest.put("model", Build.MODEL);
        manifest.put("device", Build.DEVICE);
        manifest.put("android_release", Build.VERSION.RELEASE);
        manifest.put("sdk_int", Build.VERSION.SDK_INT);
        manifest.put("display_width_px_at_creation", displayWidthPx(context));
        manifest.put("display_height_px_at_creation", displayHeightPx(context));
        manifest.put("density_dpi_at_creation", densityDpi(context));
        manifest.put("sampling_request_hz", 100);
        manifest.put("imu_channels", "accelerometer_m_s2_xyz,gyroscope_rad_s_xyz");
        manifest.put("formal_xy_transform",
                "Per-event display pixels are linearly scaled to the HMOG 1080x1920 "
                        + "portrait plane (1920x1080 for rotations 1/3); raw touch rows "
                        + "remain unchanged.");
        manifest.put("privacy", "No typed string is stored.");
        manifest.put("run_isolation",
                "Only explicitly completed run_id values are eligible for formal evaluation.");
        manifest.put("posture_values", "sitting,walking");
        writeText(new File(directory, "manifest.json"), manifest.toString(2) + "\n");
        appendTaskEvent(context, "session", "created", safeProfile);
        return sessionId;
    }

    public static File sessionsRoot(Context context) {
        return new File(context.getFilesDir(), "sessions");
    }

    public static File sessionDir(Context context, String sessionId) {
        return new File(sessionsRoot(context), sanitize(sessionId));
    }

    public static File currentSessionDir(Context context) {
        String id = prefs(context).getString(KEY_SESSION, "");
        return id == null || id.isEmpty() ? null : sessionDir(context, id);
    }

    public static boolean hasSession(Context context) {
        File directory = currentSessionDir(context);
        if (directory == null || !directory.isDirectory()) {
            return false;
        }
        File manifest = new File(directory, "manifest.json");
        try {
            String text = new String(
                    java.nio.file.Files.readAllBytes(manifest.toPath()),
                    StandardCharsets.UTF_8);
            return SCHEMA.equals(new JSONObject(text).optString("schema"));
        } catch (IOException | JSONException error) {
            return false;
        }
    }

    public static boolean isRecording(Context context) {
        return prefs(context).getBoolean(KEY_RECORDING, false);
    }

    public static String sessionId(Context context) {
        return prefs(context).getString(KEY_SESSION, "");
    }

    public static String profileId(Context context) {
        return prefs(context).getString(KEY_PROFILE, "P001");
    }

    public static String activeTask(Context context) {
        return prefs(context).getString(KEY_TASK, "");
    }

    public static String activePhase(Context context) {
        return prefs(context).getString(KEY_PHASE, "");
    }

    public static String activePosture(Context context) {
        return prefs(context).getString(KEY_POSTURE, "");
    }

    public static String activeRunId(Context context) {
        return prefs(context).getString(KEY_RUN_ID, "");
    }

    public static String newRunId(String task) {
        return sanitize(task) + "_" + utcNow().replace(":", "").replace("-", "")
                + "_" + UUID.randomUUID().toString().substring(0, 8);
    }

    public static int displayWidthPx(Context context) {
        return displayMetrics(context).widthPixels;
    }

    public static int displayHeightPx(Context context) {
        return displayMetrics(context).heightPixels;
    }

    public static int densityDpi(Context context) {
        return displayMetrics(context).densityDpi;
    }

    public static int displayRotation(Context context) {
        Display display = defaultDisplay(context);
        return display == null ? Surface.ROTATION_0 : display.getRotation();
    }

    private static Display defaultDisplay(Context context) {
        try {
            DisplayManager manager = context.getSystemService(DisplayManager.class);
            return manager == null ? null : manager.getDisplay(Display.DEFAULT_DISPLAY);
        } catch (RuntimeException ignored) {
            return null;
        }
    }

    @SuppressWarnings("deprecation")
    private static DisplayMetrics displayMetrics(Context context) {
        DisplayMetrics metrics = new DisplayMetrics();
        Display display = defaultDisplay(context);
        if (display != null) {
            try {
                display.getRealMetrics(metrics);
            } catch (RuntimeException ignored) {
                metrics.setTo(context.getResources().getDisplayMetrics());
            }
        } else {
            metrics.setTo(context.getResources().getDisplayMetrics());
        }
        return metrics;
    }

    public static void setRecording(
            Context context,
            boolean value,
            String task,
            String phase,
            String posture,
            String runId) {
        prefs(context).edit()
                .putBoolean(KEY_RECORDING, value)
                .putString(KEY_TASK, task == null ? "" : task)
                .putString(KEY_PHASE, phase == null ? "" : phase)
                .putString(KEY_POSTURE, posture == null ? "" : posture)
                .putString(KEY_RUN_ID, runId == null ? "" : runId)
                .apply();
    }

    public static void setPhase(Context context, String phase) {
        prefs(context).edit().putString(KEY_PHASE, phase == null ? "" : phase).apply();
    }

    public static void appendCsv(
            Context context, String fileName, String header, Object... values) {
        File directory = currentSessionDir(context);
        if (directory == null || !directory.isDirectory()) {
            return;
        }
        int expectedColumns = header.trim().split(",", -1).length;
        if (values.length != expectedColumns) {
            incrementCounter(context.getApplicationContext(), KEY_WRITE_ERRORS);
            return;
        }
        StringBuilder line = new StringBuilder();
        for (int index = 0; index < values.length; index++) {
            if (index > 0) {
                line.append(',');
            }
            line.append(csv(values[index]));
        }
        Context appContext = context.getApplicationContext();
        File path = new File(directory, fileName);
        try {
            CSV_EXECUTOR.execute(() -> {
                try {
                    ensureCsv(directory, fileName, header);
                    String key = path.getAbsolutePath();
                    WriterState state = CSV_WRITERS.get(key);
                    if (state == null) {
                        state = new WriterState(path);
                        CSV_WRITERS.put(key, state);
                    }
                    state.write(line.toString());
                } catch (IOException error) {
                    incrementCounter(appContext, KEY_WRITE_ERRORS);
                }
            });
        } catch (java.util.concurrent.RejectedExecutionException error) {
            incrementCounter(appContext, KEY_DROPPED_ROWS);
        }
    }

    public static void flushAndClosePendingWrites(Context context) throws IOException {
        Future<?> future;
        try {
            future = CSV_EXECUTOR.submit(() -> {
                IOException failure = null;
                for (WriterState state : CSV_WRITERS.values()) {
                    try {
                        state.closeAndSync();
                    } catch (IOException error) {
                        failure = error;
                    }
                }
                CSV_WRITERS.clear();
                if (failure != null) {
                    throw new RuntimeException(failure);
                }
            });
        } catch (java.util.concurrent.RejectedExecutionException error) {
            incrementCounter(context.getApplicationContext(), KEY_DROPPED_ROWS);
            throw new IOException("CSV writer queue rejected flush", error);
        }
        try {
            future.get(30, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while flushing CSV", error);
        } catch (ExecutionException error) {
            incrementCounter(context.getApplicationContext(), KEY_WRITE_ERRORS);
            throw new IOException("Failed to flush CSV", error.getCause());
        } catch (TimeoutException error) {
            throw new IOException("Timed out flushing CSV writer", error);
        }
    }

    public static int pendingRows() {
        return CSV_EXECUTOR.getQueue().size();
    }

    public static synchronized void writeExportAudit(Context context)
            throws IOException, JSONException {
        File directory = currentSessionDir(context);
        if (directory == null || !directory.isDirectory()) {
            throw new IOException("No active session");
        }
        JSONObject audit = new JSONObject();
        audit.put("schema", SCHEMA);
        audit.put("session_id", sessionId(context));
        audit.put("export_utc", utcNow());
        audit.put("export_wall_ms", System.currentTimeMillis());
        audit.put("csv_pending_rows_after_flush", pendingRows());
        audit.put("csv_write_errors", prefs(context).getLong(KEY_WRITE_ERRORS, 0L));
        audit.put("csv_dropped_rows", prefs(context).getLong(KEY_DROPPED_ROWS, 0L));
        writeText(new File(directory, "export_audit.json"), audit.toString(2) + "\n");
    }

    private static void incrementCounter(Context context, String key) {
        synchronized (COUNTER_LOCK) {
            SharedPreferences preferences = prefs(context);
            preferences.edit().putLong(
                    key, preferences.getLong(key, 0L) + 1L).apply();
        }
    }

    public static void recordWriteError(Context context) {
        incrementCounter(context.getApplicationContext(), KEY_WRITE_ERRORS);
    }

    public static void recordCaptureError(Context context, String error) {
        prefs(context).edit()
                .putString(KEY_LAST_CAPTURE_ERROR, error == null ? "" : error)
                .apply();
    }

    public static void appendTaskEvent(
            Context context, String task, String event, String details) {
        appendTaskEvent(
                context,
                task,
                event,
                details,
                activePosture(context),
                activeRunId(context));
    }

    public static void appendTaskEvent(
            Context context,
            String task,
            String event,
            String details,
            String posture,
            String runId) {
        if ("calibration_complete".equals(event) || "task_complete".equals(event)) {
            prefs(context).edit().putString(
                    KEY_COMPLETED_PREFIX + sanitize(task) + "."
                            + sanitize(posture),
                    runId).apply();
        }
        appendCsv(
                context,
                "task_events.csv",
                TASK_HEADER,
                SCHEMA,
                sessionId(context),
                profileId(context),
                SystemClock.elapsedRealtimeNanos(),
                System.currentTimeMillis(),
                task,
                event,
                details,
                posture,
                runId
        );
    }

    public static String csv(Object value) {
        if (value == null) {
            return "";
        }
        String text = String.valueOf(value);
        if (text.contains(",") || text.contains("\"") || text.contains("\n")
                || text.contains("\r")) {
            return "\"" + text.replace("\"", "\"\"") + "\"";
        }
        return text;
    }

    public static String utcNow() {
        SimpleDateFormat format = new SimpleDateFormat(
                "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US);
        format.setTimeZone(TimeZone.getTimeZone("UTC"));
        return format.format(new Date());
    }

    public static String sanitize(String value) {
        if (value == null) {
            return "";
        }
        return value.trim().replaceAll("[^A-Za-z0-9_.-]", "_");
    }

    private static void ensureCsv(File directory, String name, String header)
            throws IOException {
        File path = new File(directory, name);
        if (!path.exists()) {
            writeText(path, header);
        }
    }

    private static void writeText(File path, String text) throws IOException {
        File parent = path.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Cannot create " + parent);
        }
        try (FileOutputStream stream = new FileOutputStream(path, false)) {
            stream.write(text.getBytes(StandardCharsets.UTF_8));
            stream.getFD().sync();
        }
    }
}
