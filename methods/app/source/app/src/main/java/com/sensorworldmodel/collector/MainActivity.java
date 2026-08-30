package com.sensorworldmodel.collector;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;

import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    private static final int REQUEST_NOTIFICATION = 110;
    private static final int REQUEST_EXPORT = 120;
    private static final AtomicBoolean EXPORT_IN_PROGRESS = new AtomicBoolean(false);
    private static volatile String exportSessionId = "";
    private static final long STATUS_REFRESH_INTERVAL_MS = 250L;
    private static final int IDLE_REFRESH_PASSES = 3;

    private static final int BG = Color.parseColor("#F4F6F8");
    private static final int CARD = Color.WHITE;
    private static final int INK = Color.parseColor("#101418");
    private static final int MUTED = Color.parseColor("#6B7280");
    private static final int ACCENT = Color.parseColor("#2357D9");
    private static final int HAIRLINE = Color.parseColor("#E6E9EE");

    /** Every run in this study is seated and still, so there is nothing to pick. */
    private static final String POSTURE = "sitting";

    private EditText profile;
    private TextView status;
    private TextView statusPill;
    private TextView statusMeta;
    private String statusDetail = "";
    private boolean resumed;
    private int idleRefreshPasses;
    private final Runnable statusRefresh = new Runnable() {
        @Override
        public void run() {
            if (!resumed) return;
            refreshStatus();
            boolean captureBusy = StudyStore.isRecording(MainActivity.this)
                    || CaptureService.isCaptureActiveInProcess()
                    || CaptureService.isCaptureFinalizingInProcess()
                    || StudyStore.pendingRows() > 0;
            idleRefreshPasses = captureBusy ? 0 : idleRefreshPasses + 1;
            if (captureBusy || idleRefreshPasses < IDLE_REFRESH_PASSES) {
                status.postDelayed(this, STATUS_REFRESH_INTERVAL_MS);
            }
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        recoverStaleCaptureState();
        buildUi();
        maybeShowDisclosure();
        requestNotificationPermission();
    }

    @Override
    protected void onResume() {
        super.onResume();
        resumed = true;
        startStatusRefreshLoop();
    }

    @Override
    protected void onPause() {
        resumed = false;
        if (status != null) status.removeCallbacks(statusRefresh);
        super.onPause();
    }

    private void buildUi() {
        LinearLayout page = Ui.col(this);
        page.setBackgroundColor(BG);
        page.setPadding(Ui.dp(this, 18), Ui.dp(this, 16), Ui.dp(this, 18), Ui.dp(this, 30));

        LinearLayout titleRow = Ui.row(this);
        LinearLayout titles = Ui.col(this);
        titles.addView(Ui.text(this, "Sensor Study", 27, INK, true), Ui.matchWrap());
        titles.addView(
                Ui.text(this, "IMU and touch trajectory collection", 13, MUTED),
                Ui.margins(this, Ui.matchWrap(), 0, 3, 0, 0));
        titleRow.addView(titles, Ui.weight(1f));
        ImageView info = Ui.icon(this, R.drawable.ic_info, 24, MUTED);
        info.setOnClickListener(view -> showAboutDialog());
        titleRow.addView(info);
        page.addView(titleRow, Ui.margins(this, Ui.matchWrap(), 0, 0, 0, 18));

        page.addView(statusCard(), Ui.margins(this, Ui.matchWrap(), 0, 0, 0, 22));

        page.addView(sectionLabel("Session"), Ui.matchWrap());
        page.addView(sessionCard(), Ui.margins(this, Ui.matchWrap(), 0, 0, 0, 22));

        page.addView(sectionLabel("Calibration"), Ui.matchWrap());
        page.addView(actionCard(
                R.drawable.ic_bolt,
                Color.parseColor("#FFF4DA"),
                Color.parseColor("#B7791F"),
                "Five-shot calibration",
                "5 actions x 5 samples  ·  about 4 min",
                view -> {
                    if (!requireIdleSession()) return;
                    startActivity(new Intent(this, CalibrationActivity.class)
                            .putExtra(CalibrationActivity.EXTRA_POSTURE, POSTURE));
                }), Ui.margins(this, Ui.matchWrap(), 0, 0, 0, 22));

        page.addView(sectionLabel("Simulated app tasks"), Ui.matchWrap());
        LinearLayout tasks = Ui.col(this);
        tasks.setBackground(Ui.rounded(this, CARD, 16));
        tasks.setElevation(Ui.dp(this, 1.5f));
        tasks.addView(taskRow(R.drawable.cat_electronics, "Amazon",
                "Search, browse a product, add to cart", "4 min",
                SimulatedTaskActivity.SHOPPING), Ui.matchWrap());
        tasks.addView(inset(), Ui.matchWrap());
        tasks.addView(taskRow(R.drawable.ic_search, "Google",
                "Two searches, open a result, read", "4 min",
                SimulatedTaskActivity.SEARCH), Ui.matchWrap());
        tasks.addView(inset(), Ui.matchWrap());
        tasks.addView(taskRow(R.drawable.ic_wa_send, "WhatsApp",
                "Open a chat, scroll, send 3 messages", "3 min",
                SimulatedTaskActivity.SOCIAL), Ui.matchWrap());
        page.addView(tasks, Ui.margins(this, Ui.matchWrap(), 0, 0, 0, 22));

        page.addView(sectionLabel("Data"), Ui.matchWrap());
        page.addView(actionCard(
                R.drawable.ic_download,
                Color.parseColor("#E3ECFD"),
                ACCENT,
                "Export session",
                "ZIP for the desktop pipeline",
                view -> beginExport()), Ui.margins(this, Ui.matchWrap(), 0, 0, 0, 10));

        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(BG);
        scroll.addView(page);
        UiInsets.applySystemBarsAndIme(scroll, 0, 0, 0, 0);
        setContentView(scroll);
    }

    // ---- home screen building blocks ------------------------------------

    private TextView sectionLabel(String value) {
        TextView label = Ui.text(this, value, 12, MUTED, true);
        label.setAllCaps(true);
        label.setLetterSpacing(0.09f);
        label.setPadding(Ui.dp(this, 4), 0, 0, Ui.dp(this, 9));
        return label;
    }

    private LinearLayout card() {
        LinearLayout card = Ui.col(this);
        card.setBackground(Ui.rounded(this, CARD, 16));
        card.setElevation(Ui.dp(this, 1.5f));
        card.setPadding(Ui.dp(this, 16), Ui.dp(this, 15), Ui.dp(this, 16), Ui.dp(this, 15));
        return card;
    }

    private View inset() {
        View line = new View(this);
        line.setBackgroundColor(HAIRLINE);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Math.max(1, Ui.dp(this, 0.7f)));
        params.setMargins(Ui.dp(this, 68), 0, 0, 0);
        line.setLayoutParams(params);
        return line;
    }

    private LinearLayout statusCard() {
        LinearLayout card = card();

        LinearLayout head = Ui.row(this);
        statusPill = Ui.text(this, "", 11, Color.parseColor("#4B5563"), true);
        statusPill.setPadding(Ui.dp(this, 10), Ui.dp(this, 4), Ui.dp(this, 10), Ui.dp(this, 4));
        statusPill.setBackground(Ui.rounded(this, Color.parseColor("#E8EAEE"), 11));
        head.addView(statusPill, Ui.wrap());
        head.addView(Ui.flexSpacer(this));
        TextView details = Ui.text(this, "Details", 13, ACCENT, true);
        details.setPadding(Ui.dp(this, 10), Ui.dp(this, 4), 0, Ui.dp(this, 4));
        details.setOnClickListener(view -> new AlertDialog.Builder(this)
                .setTitle("Capture diagnostics")
                .setMessage(statusDetail)
                .setPositiveButton("Close", null)
                .show());
        head.addView(details);
        card.addView(head, Ui.matchWrap());

        status = Ui.text(this, "", 15, INK, true);
        card.addView(status, Ui.margins(this, Ui.matchWrap(), 0, 10, 0, 0));

        statusMeta = Ui.text(this, "", 13, MUTED);
        card.addView(statusMeta, Ui.margins(this, Ui.matchWrap(), 0, 4, 0, 0));
        return card;
    }

    private LinearLayout sessionCard() {
        LinearLayout card = card();
        card.addView(Ui.text(this, "Participant ID", 13, MUTED), Ui.matchWrap());
        card.addView(
                Ui.text(this, "Use a research ID only; do not enter a name or contact information.", 11, MUTED),
                Ui.margins(this, Ui.matchWrap(), 0, 3, 0, 6));

        profile = new EditText(this);
        profile.setHint("P001");
        profile.setSingleLine(true);
        profile.setText(StudyStore.profileId(this));
        profile.setTextSize(16);
        profile.setTextColor(INK);
        profile.setBackground(Ui.rounded(this, Color.parseColor("#F7F8FA"), 10, HAIRLINE, 1));
        profile.setPadding(Ui.dp(this, 12), Ui.dp(this, 11), Ui.dp(this, 12), Ui.dp(this, 11));
        card.addView(profile, Ui.margins(this, Ui.matchWrap(), 0, 8, 0, 12));

        TextView create = Ui.text(this, "New capture session", 15, Color.WHITE, true);
        create.setGravity(Gravity.CENTER);
        create.setBackground(Ui.rounded(this, ACCENT, 12));
        create.setPadding(Ui.dp(this, 16), Ui.dp(this, 13), Ui.dp(this, 16), Ui.dp(this, 13));
        create.setOnClickListener(view -> createSession());
        card.addView(create, Ui.matchWrap());
        return card;
    }

    private LinearLayout actionCard(
            int iconRes,
            int iconBackground,
            int iconTint,
            String title,
            String subtitle,
            View.OnClickListener click) {
        LinearLayout card = card();
        card.setOrientation(LinearLayout.HORIZONTAL);
        card.setGravity(Gravity.CENTER_VERTICAL);

        LinearLayout badge = Ui.row(this);
        badge.setGravity(Gravity.CENTER);
        badge.setBackground(Ui.rounded(this, iconBackground, 12));
        badge.addView(Ui.icon(this, iconRes, 22, iconTint));
        card.addView(badge, Ui.size(this, 42, 42));

        LinearLayout copy = Ui.col(this);
        copy.addView(Ui.text(this, title, 16, INK, true), Ui.matchWrap());
        copy.addView(Ui.text(this, subtitle, 13, MUTED),
                Ui.margins(this, Ui.matchWrap(), 0, 3, 0, 0));
        card.addView(copy, Ui.margins(this, Ui.weight(1f), 13, 0, 8, 0));

        card.addView(Ui.icon(this, R.drawable.ic_chevron_right, 20, Color.parseColor("#B7BDC7")));
        card.setOnClickListener(click);
        return card;
    }

    private LinearLayout taskRow(
            int iconRes, String title, String subtitle, String duration, String task) {
        LinearLayout row = Ui.row(this);
        row.setPadding(Ui.dp(this, 14), Ui.dp(this, 14), Ui.dp(this, 14), Ui.dp(this, 14));

        ImageView icon = new ImageView(this);
        icon.setImageResource(iconRes);
        row.addView(icon, Ui.size(this, 44, 44));

        LinearLayout copy = Ui.col(this);
        LinearLayout titleRow = Ui.row(this);
        titleRow.addView(Ui.text(this, title, 16, INK, true), Ui.wrap());
        TextView chip = Ui.text(this, duration, 11, Color.parseColor("#4B5563"), true);
        chip.setBackground(Ui.rounded(this, Color.parseColor("#EEF0F3"), 9));
        chip.setPadding(Ui.dp(this, 7), Ui.dp(this, 2), Ui.dp(this, 7), Ui.dp(this, 2));
        titleRow.addView(chip, Ui.margins(this, Ui.wrap(), 8, 0, 0, 0));
        copy.addView(titleRow, Ui.matchWrap());
        copy.addView(Ui.text(this, subtitle, 13, MUTED),
                Ui.margins(this, Ui.matchWrap(), 0, 3, 0, 0));
        row.addView(copy, Ui.margins(this, Ui.weight(1f), 13, 0, 8, 0));

        row.addView(Ui.icon(this, R.drawable.ic_chevron_right, 20, Color.parseColor("#B7BDC7")));
        row.setOnClickListener(view -> {
            if (!requireIdleSession()) return;
            startActivity(new Intent(this, SimulatedTaskActivity.class)
                    .putExtra(SimulatedTaskActivity.EXTRA_TASK, task)
                    .putExtra(SimulatedTaskActivity.EXTRA_POSTURE, POSTURE));
        });
        return row;
    }

    private void showAboutDialog() {
        new AlertDialog.Builder(this)
                .setTitle("About this study capture")
                .setMessage("Workflow\n"
                        + "1. Create a session for the participant.\n"
                        + "2. Record the five-shot calibration set.\n"
                        + "3. Run the three simulated app tasks.\n"
                        + "4. Export the ZIP and freeze the cache on the desktop.\n\n"
                        + "What is recorded\n"
                        + "The simulated tasks run inside this app, so the full per-point "
                        + "XY, pressure, size, pointer id, event time and synchronised IMU "
                        + "are captured.\n\n"
                        + "Typing\n"
                        + "Only the timing and the character counts are stored. The typed "
                        + "characters themselves are never written to disk.\n\n"
                        + "Sensors only run after you start a task yourself, the status bar "
                        + "notification is always visible, and you can stop at any time.")
                .setPositiveButton("Got it", null)
                .show();
    }

    private void createSession() {
        if (rejectWhileExporting()) return;
        if (captureLifecycleBusy()) {
            Toast.makeText(this, "Stop the current capture first.", Toast.LENGTH_LONG).show();
            return;
        }
        try {
            String id = StudyStore.createSession(this, profile.getText().toString());
            Toast.makeText(this, "Created " + id, Toast.LENGTH_LONG).show();
            refreshStatus();
        } catch (IOException | JSONException error) {
            Toast.makeText(
                    this,
                    "Could not create the session: " + error.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }

    private void beginExport() {
        if (EXPORT_IN_PROGRESS.get()) {
            Toast.makeText(
                    this,
                    "The current session is exporting. Please wait.",
                    Toast.LENGTH_LONG).show();
            return;
        }
        if (!requireSession()) return;
        if (captureLifecycleBusy()) {
            Toast.makeText(
                    this,
                    "Finish the current task before exporting.",
                    Toast.LENGTH_LONG).show();
            return;
        }
        String expectedSessionId = StudyStore.sessionId(this);
        if (expectedSessionId == null || expectedSessionId.isEmpty()) {
            Toast.makeText(this, "The current session is not valid.", Toast.LENGTH_LONG).show();
            return;
        }
        if (!EXPORT_IN_PROGRESS.compareAndSet(false, true)) {
            Toast.makeText(
                    this,
                    "The current session is exporting. Please wait.",
                    Toast.LENGTH_LONG).show();
            return;
        }
        exportSessionId = expectedSessionId;
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/zip");
        intent.putExtra(
                Intent.EXTRA_TITLE,
                "sensor_study_" + expectedSessionId + ".zip");
        try {
            startActivityForResult(intent, REQUEST_EXPORT);
        } catch (RuntimeException error) {
            releaseExportLock(expectedSessionId);
            Toast.makeText(
                    this,
                    "Could not open the export location: " + error.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != REQUEST_EXPORT) {
            return;
        }
        String expectedSessionId = exportSessionId;
        if (resultCode != RESULT_OK || data == null || data.getData() == null) {
            releaseExportLock(expectedSessionId);
            refreshStatus();
            return;
        }
        if (!EXPORT_IN_PROGRESS.get()
                || expectedSessionId == null
                || expectedSessionId.isEmpty()) {
            releaseExportLock(expectedSessionId);
            Toast.makeText(
                    this,
                    "The export state expired. Please export again.",
                    Toast.LENGTH_LONG).show();
            refreshStatus();
            return;
        }
        Uri destination = data.getData();
        statusMeta.setText("Exporting. Please keep this screen open.");
        new Thread(() -> {
            try {
                SessionExporter.exportCurrent(this, destination, expectedSessionId);
                runOnUiThread(() ->
                        Toast.makeText(this, "Export complete", Toast.LENGTH_LONG).show());
            } catch (IOException error) {
                runOnUiThread(() -> Toast.makeText(
                        this,
                        "Export failed: " + error.getMessage(),
                        Toast.LENGTH_LONG).show());
            } finally {
                releaseExportLock(expectedSessionId);
                runOnUiThread(() -> {
                    refreshStatus();
                    startStatusRefreshLoop();
                });
            }
        }, "session-export").start();
    }

    private boolean requireSession() {
        if (StudyStore.hasSession(this)) return true;
        Toast.makeText(this, "Create a session first.", Toast.LENGTH_LONG).show();
        return false;
    }

    private boolean requireIdleSession() {
        if (rejectWhileExporting()) return false;
        if (!requireSession()) return false;
        if (captureLifecycleBusy()) {
            Toast.makeText(
                    this,
                    "A capture task is already running. Stop it first.",
                    Toast.LENGTH_LONG).show();
            return false;
        }
        return true;
    }

    private boolean rejectWhileExporting() {
        if (!EXPORT_IN_PROGRESS.get()) return false;
        Toast.makeText(
                this,
                "The current session is exporting. Wait for it to finish before "
                        + "creating a session or starting a task.",
                Toast.LENGTH_LONG).show();
        return true;
    }

    private boolean captureLifecycleBusy() {
        return StudyStore.isRecording(this)
                || CaptureService.isCaptureActiveInProcess()
                || CaptureService.isCaptureFinalizingInProcess();
    }

    private static void releaseExportLock(String expectedSessionId) {
        String lockedSessionId = exportSessionId;
        if (expectedSessionId != null
                && !expectedSessionId.isEmpty()
                && !expectedSessionId.equals(lockedSessionId)) {
            return;
        }
        exportSessionId = "";
        EXPORT_IN_PROGRESS.set(false);
    }

    private void startStatusRefreshLoop() {
        if (!resumed || status == null) return;
        status.removeCallbacks(statusRefresh);
        idleRefreshPasses = 0;
        status.post(statusRefresh);
    }

    private void recoverStaleCaptureState() {
        if (!StudyStore.isRecording(this)
                || CaptureService.isCaptureActiveInProcess()) {
            return;
        }
        String task = StudyStore.activeTask(this);
        String detail = "process_restarted_without_capture_service"
                + ";run_id=" + StudyStore.activeRunId(this)
                + ";phase=" + StudyStore.activePhase(this);
        StudyStore.appendTaskEvent(
                this,
                task.isEmpty() ? "capture" : task,
                "stale_recording_recovered",
                detail);
        StudyStore.setRecording(this, false, "", "", "", "");
    }

    private void refreshStatus() {
        if (status == null) return;
        if (!StudyStore.hasSession(this)) {
            statusPill.setText("No session");
            statusPill.setTextColor(Color.parseColor("#B42318"));
            statusPill.setBackground(Ui.rounded(this, Color.parseColor("#FDE7E7"), 11));
            status.setText(getString(R.string.status_no_session));
            statusMeta.setText("Create a session to unlock the tasks.");
            statusDetail = "No session has been created yet.";
            return;
        }
        boolean recording = StudyStore.isRecording(this);
        File directory = StudyStore.currentSessionDir(this);
        long bytes = directorySize(directory);
        int completedRuns = 0;
        for (String key : StudyStore.prefs(this).getAll().keySet()) {
            if (key.startsWith(StudyStore.KEY_COMPLETED_PREFIX)) {
                completedRuns++;
            }
        }
        statusPill.setText(recording ? "Recording" : "Idle");
        statusPill.setTextColor(recording
                ? Color.parseColor("#15803D") : Color.parseColor("#4B5563"));
        statusPill.setBackground(Ui.rounded(this, recording
                ? Color.parseColor("#DCFCE7") : Color.parseColor("#E8EAEE"), 11));
        status.setText(StudyStore.sessionId(this));
        statusMeta.setText(String.format(
                Locale.US,
                "%d completed runs  ·  %.2f MB%s",
                completedRuns,
                bytes / 1024.0 / 1024.0,
                recording
                        ? "  ·  " + StudyStore.activeTask(this)
                                + " / " + StudyStore.activePhase(this)
                        : ""));
        statusDetail = String.format(
                Locale.US,
                "Session: %s\nStatus: %s\nPosture: %s\nCompleted task runs: %d\n"
                        + "Current files: %.2f MB\n\n"
                        + "IMU A/G: %.1f / %.1f Hz\nMax gap: %.1f / %.1f ms\n\n"
                        + "Write queue: %d\nWrite errors: %d\nDropped rows: %d",
                StudyStore.sessionId(this),
                recording
                        ? "recording " + StudyStore.activeTask(this)
                                + " / " + StudyStore.activePhase(this)
                        : "not recording",
                StudyStore.activePosture(this),
                completedRuns,
                bytes / 1024.0 / 1024.0,
                StudyStore.prefs(this).getFloat(StudyStore.KEY_LIVE_ACC_HZ, 0f),
                StudyStore.prefs(this).getFloat(StudyStore.KEY_LIVE_GYRO_HZ, 0f),
                StudyStore.prefs(this).getFloat(StudyStore.KEY_LIVE_ACC_GAP_MS, 0f),
                StudyStore.prefs(this).getFloat(StudyStore.KEY_LIVE_GYRO_GAP_MS, 0f),
                StudyStore.pendingRows(),
                StudyStore.prefs(this).getLong(StudyStore.KEY_WRITE_ERRORS, 0L),
                StudyStore.prefs(this).getLong(StudyStore.KEY_DROPPED_ROWS, 0L));
    }

    private void maybeShowDisclosure() {
        if (StudyStore.prefs(this).getBoolean("disclosure_seen", false)) return;
        new AlertDialog.Builder(this)
                .setTitle("About this study capture")
                .setMessage("This app only records sensors after you start a "
                        + "task yourself.\n\nThe simulated tasks record the full "
                        + "touch trajectory. Typing is stored as timing and "
                        + "character counts only, never as text.\n\nThe status "
                        + "bar notification is always visible and you can stop at "
                        + "any time.")
                .setCancelable(false)
                .setPositiveButton("Got it", (dialog, which) ->
                        StudyStore.prefs(this).edit()
                                .putBoolean("disclosure_seen", true).apply())
                .show();
    }

    private void requestNotificationPermission() {
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.POST_NOTIFICATIONS},
                    REQUEST_NOTIFICATION);
        }
    }

    private static long directorySize(File file) {
        if (file == null || !file.exists()) return 0L;
        if (file.isFile()) return file.length();
        long total = 0L;
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) total += directorySize(child);
        }
        return total;
    }
}
