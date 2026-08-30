package com.sensorworldmodel.collector;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.util.UUID;

public final class CalibrationActivity extends Activity
        implements GestureCaptureView.Listener {
    public static final String EXTRA_POSTURE = "posture";
    private static final String[] ACTIONS =
            {"tap", "scroll", "swipe", "pinch", "keystroke"};
    private static final int REQUIRED = 5;
    /**
     * One short phrase per keystroke sample. Fixed and ordered, so every
     * participant types the same five strings but never the same one twice.
     */
    private static final String[] TYPING_PHRASES = {
        "hello there",
        "good morning",
        "see you soon",
        "thanks a lot",
        "have a good day",
    };

    private int actionIndex = 0;
    private int accepted = 0;
    private String lastEventId = "";
    private Button undo;
    private TextView title;
    private TextView progress;
    private TextView hint;
    private GestureCaptureView gestureView;
    private LinearLayout typingPanel;
    private EditText editText;
    private Button finishTyping;
    private long typingStartElapsedNs = -1L;
    private long typingStartWallMs = -1L;
    private String typingEventId = "";
    private int beforeCount = 0;
    private boolean suppressTextWatcher = false;
    private boolean completed = false;
    private boolean captureReady = false;
    private boolean captureRequested = false;
    private String posture;
    private String runId;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        if (!StudyStore.hasSession(this)) {
            Toast.makeText(this, "Create a study session first.", Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        posture = getIntent().getStringExtra(EXTRA_POSTURE);
        if (!"walking".equals(posture)) {
            posture = "sitting";
        }
        runId = StudyStore.newRunId("fewshot_calibration");
        buildUi();
        startCapture();
        showCurrentAction();
        setCaptureControlsEnabled(false);
        hint.setText("Warming up the sensors. Please hold still for a moment.");
        awaitCaptureReady(0);
    }

    @Override
    protected void onDestroy() {
        captureReady = false;
        if (isFinishing() && captureRequested) {
            if (!completed) {
                StudyStore.appendTaskEvent(
                        this,
                        "fewshot_calibration",
                        "calibration_aborted",
                        ACTIONS[Math.min(actionIndex, ACTIONS.length - 1)],
                        posture,
                        runId);
            }
            Intent stop = new Intent(this, CaptureService.class)
                    .setAction(CaptureService.ACTION_STOP)
                    .putExtra(CaptureService.EXTRA_RUN_ID, runId);
            startService(stop);
        }
        super.onDestroy();
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (!isChangingConfigurations() && !completed && !isFinishing()) {
            finish();
        }
    }

    @Override
    public void onAccepted(String action, String eventId) {
        if (!captureReady || !captureRunIsActive()) {
            return;
        }
        accepted++;
        lastEventId = eventId;
        Toast.makeText(this, "Recorded " + accepted + "/" + REQUIRED, Toast.LENGTH_SHORT).show();
        if (accepted >= REQUIRED) {
            nextAction();
        } else {
            showCurrentAction();
        }
    }

    @Override
    public void onRejected(String action, String reason) {
        if (!captureReady || !captureRunIsActive()) {
            return;
        }
        hint.setText(reason);
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.WHITE);
        UiInsets.applySystemBarsAndIme(root, 28, 28, 28, 28);

        title = text(26, true);
        progress = text(18, true);
        hint = text(16, false);
        hint.setTextColor(Color.rgb(110, 55, 25));
        root.addView(title);
        root.addView(progress);
        root.addView(hint);

        gestureView = new GestureCaptureView(this);
        gestureView.setListener(this);
        root.addView(gestureView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        typingPanel = new LinearLayout(this);
        typingPanel.setOrientation(LinearLayout.VERTICAL);
        typingPanel.setGravity(Gravity.CENTER);
        editText = new EditText(this);
        editText.setHint("Type: " + currentPhrase());
        editText.setSingleLine(false);
        editText.setMinLines(3);
        typingPanel.addView(editText, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        finishTyping = new Button(this);
        finishTyping.setText("Done");
        typingPanel.addView(finishTyping);
        typingPanel.setVisibility(View.GONE);
        root.addView(typingPanel, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        undo = new Button(this);
        undo.setText("Undo last sample");
        undo.setAllCaps(false);
        undo.setOnClickListener(view -> undoLastSample());
        root.addView(undo);

        Button cancel = new Button(this);
        cancel.setText("Stop and go back");
        cancel.setAllCaps(false);
        cancel.setOnClickListener(view -> finish());
        root.addView(cancel);
        setContentView(root);

        editText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(
                    CharSequence text, int start, int count, int after) {
                beforeCount = text.length();
            }

            @Override
            public void onTextChanged(
                    CharSequence text, int start, int before, int count) {
                if (suppressTextWatcher || !captureReady || !captureRunIsActive()) {
                    return;
                }
                if (typingStartElapsedNs < 0) {
                    typingStartElapsedNs = SystemClock.elapsedRealtimeNanos();
                    typingStartWallMs = System.currentTimeMillis();
                    typingEventId = "key_" + UUID.randomUUID().toString().substring(0, 12);
                }
                StudyStore.appendCsv(
                        CalibrationActivity.this,
                        "keystroke.csv",
                        StudyStore.KEYSTROKE_HEADER,
                        StudyStore.SCHEMA,
                        StudyStore.sessionId(CalibrationActivity.this),
                        StudyStore.profileId(CalibrationActivity.this),
                        "fewshot_calibration",
                        "keystroke",
                        typingEventId,
                        SystemClock.elapsedRealtimeNanos(),
                        System.currentTimeMillis(),
                        beforeCount,
                        text.length(),
                        count,
                        before,
                        StudyStore.activePosture(CalibrationActivity.this),
                        StudyStore.activeRunId(CalibrationActivity.this)
                );
            }

            @Override
            public void afterTextChanged(Editable text) {}
        });
        finishTyping.setOnClickListener(view -> finishTypingSample());
    }

    private TextView text(int sp, boolean strong) {
        TextView value = new TextView(this);
        value.setTextSize(sp);
        value.setTextColor(Color.rgb(30, 40, 60));
        value.setPadding(0, 8, 0, 8);
        if (strong) {
            value.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        }
        return value;
    }

    private void startCapture() {
        Intent intent = new Intent(this, CaptureService.class)
                .setAction(CaptureService.ACTION_START)
                .putExtra(CaptureService.EXTRA_TASK, "fewshot_calibration")
                .putExtra(CaptureService.EXTRA_PHASE, ACTIONS[actionIndex])
                .putExtra(CaptureService.EXTRA_POSTURE, posture)
                .putExtra(CaptureService.EXTRA_RUN_ID, runId);
        captureRequested = true;
        startForegroundService(intent);
    }

    private void awaitCaptureReady(int attempt) {
        if (isFinishing()) {
            return;
        }
        if (CaptureService.isCaptureReadyInProcess(runId)) {
            gestureView.postDelayed(() -> {
                if (isFinishing()) {
                    return;
                }
                if (!CaptureService.isCaptureReadyInProcess(runId)) {
                    finishAfterCaptureLoss();
                    return;
                }
                captureReady = true;
                setCaptureControlsEnabled(true);
                showCurrentAction();
                monitorCaptureLiveness();
            }, 300L);
            return;
        }
        if (attempt >= 50) {
            Toast.makeText(
                    this,
                    "The sensor service did not start within 5 seconds, so this run "
                            + "was aborted. Check the notification permission and the sensors.",
                    Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        gestureView.postDelayed(() -> awaitCaptureReady(attempt + 1), 100L);
    }

    private void setCaptureControlsEnabled(boolean enabled) {
        gestureView.setEnabled(enabled);
        editText.setEnabled(enabled);
        finishTyping.setEnabled(enabled);
        if (undo != null) {
            undo.setEnabled(enabled);
        }
    }

    private void updatePhase() {
        Intent intent = new Intent(this, CaptureService.class)
                .setAction(CaptureService.ACTION_PHASE)
                .putExtra(CaptureService.EXTRA_PHASE, ACTIONS[actionIndex])
                .putExtra(CaptureService.EXTRA_RUN_ID, runId);
        startService(intent);
    }

    private void showCurrentAction() {
        String action = ACTIONS[actionIndex];
        title.setText(String.format(
                java.util.Locale.US, "Five-shot: %s", displayName(action)));
        progress.setText(String.format(
                java.util.Locale.US,
                "Action %d of 5  ·  %d of %d captured",
                actionIndex + 1,
                accepted,
                REQUIRED));
        hint.setText(instruction(action));
        if ("keystroke".equals(action)) {
            hint.setText("Type “" + currentPhrase() + "” and tap Done. Only the "
                    + "timing and the character counts are stored, never the text.");
        }
        gestureView.setLabeledAction(action);
        boolean typing = "keystroke".equals(action);
        gestureView.setVisibility(typing ? View.GONE : View.VISIBLE);
        typingPanel.setVisibility(typing ? View.VISIBLE : View.GONE);
        if (typing) {
            editText.requestFocus();
            editText.postDelayed(() -> ((InputMethodManager)
                    getSystemService(INPUT_METHOD_SERVICE))
                    .showSoftInput(editText, InputMethodManager.SHOW_IMPLICIT), 200);
        }
    }

    private void finishTypingSample() {
        if (!captureReady || !captureRunIsActive()) {
            return;
        }
        String value = editText.getText().toString();
        if (typingStartElapsedNs < 0 || value.trim().isEmpty()) {
            Toast.makeText(this, "Type the phrase first.", Toast.LENGTH_SHORT).show();
            return;
        }
        if (!value.trim().toLowerCase(java.util.Locale.US).replaceAll("\\s+", " ")
                .equals(currentPhrase())) {
            Toast.makeText(
                    this,
                    "Please type it exactly: " + currentPhrase(),
                    Toast.LENGTH_SHORT).show();
            return;
        }
        long endNs = SystemClock.elapsedRealtimeNanos();
        long endWall = System.currentTimeMillis();
        int nKeys = value.length();
        int nLetters = 0;
        for (int index = 0; index < value.length(); index++) {
            if (Character.isLetter(value.charAt(index))) {
                nLetters++;
            }
        }
        StudyStore.appendCsv(
                this,
                "events.csv",
                StudyStore.EVENT_HEADER,
                StudyStore.SCHEMA,
                StudyStore.sessionId(this),
                StudyStore.profileId(this),
                "fewshot_calibration",
                typingEventId,
                "in_app_textwatcher_redacted",
                "keystroke",
                typingStartElapsedNs,
                endNs,
                typingStartWallMs,
                endWall,
                (endNs - typingStartElapsedNs) / 1_000_000.0,
                0,
                "time_and_counts_exact_text_redacted",
                "not_applicable",
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                0,
                nKeys,
                nLetters,
                "duration,orientation,n_keys,n_letters",
                "Typed content was not stored.",
                StudyStore.displayWidthPx(this),
                StudyStore.displayHeightPx(this),
                StudyStore.densityDpi(this),
                StudyStore.activePosture(this),
                StudyStore.activeRunId(this)
        );
        accepted++;
        lastEventId = typingEventId;
        suppressTextWatcher = true;
        editText.setText("");
        suppressTextWatcher = false;
        typingStartElapsedNs = -1L;
        typingStartWallMs = -1L;
        typingEventId = "";
        if (accepted >= REQUIRED) {
            nextAction();
        } else {
            showCurrentAction();
        }
    }

    /**
     * Drops the most recent accepted sample for the current action. The CSV row
     * that was already written is never deleted; a retraction is appended
     * instead, so the raw log stays append-only and the desktop pipeline can
     * filter that event id out.
     */
    private void undoLastSample() {
        if (!captureReady || !captureRunIsActive()) {
            return;
        }
        if (accepted <= 0) {
            Toast.makeText(
                    this,
                    "Nothing to undo for " + displayName(ACTIONS[actionIndex]) + ".",
                    Toast.LENGTH_SHORT).show();
            return;
        }
        accepted--;
        StudyStore.appendTaskEvent(
                this,
                "fewshot_calibration",
                "sample_retracted",
                "action=" + ACTIONS[actionIndex] + ";event_id=" + lastEventId
                        + ";remaining=" + accepted + "/" + REQUIRED,
                posture,
                runId);
        lastEventId = "";
        Toast.makeText(
                this,
                "Removed one sample. Now " + accepted + "/" + REQUIRED + ".",
                Toast.LENGTH_SHORT).show();
        showCurrentAction();
    }

    private void nextAction() {
        StudyStore.appendTaskEvent(
                this, "fewshot_calibration", "action_complete", ACTIONS[actionIndex]);
        actionIndex++;
        accepted = 0;
        if (actionIndex >= ACTIONS.length) {
            completed = true;
            StudyStore.appendTaskEvent(
                    this, "fewshot_calibration", "calibration_complete",
                    "five actions x five samples");
            Toast.makeText(
                    this,
                    "Five-shot capture complete. You can export it now.",
                    Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        updatePhase();
        showCurrentAction();
    }

    /** The phrase for the keystroke sample currently being collected. */
    private String currentPhrase() {
        return TYPING_PHRASES[Math.min(accepted, TYPING_PHRASES.length - 1)];
    }

    private static String displayName(String action) {
        if ("tap".equals(action)) return "Tap";
        if ("scroll".equals(action)) return "Scroll";
        if ("swipe".equals(action)) return "Swipe";
        if ("pinch".equals(action)) return "Pinch";
        return "Keystroke";
    }

    private static String instruction(String action) {
        if ("tap".equals(action)) return "Tap the blue dot once, at your natural speed.";
        if ("scroll".equals(action)) return "Make one natural vertical scroll inside the area.";
        if ("swipe".equals(action)) return "Make one natural horizontal swipe inside the area.";
        if ("pinch".equals(action)) return "Pinch once with two fingers to zoom in or out.";
        return "Type the phrase shown above and tap Done. Only the timing and the "
                + "character counts are stored, never the text.";
    }

    private boolean captureRunIsActive() {
        return CaptureService.isCaptureReadyInProcess(runId)
                && StudyStore.isRecording(this)
                && runId.equals(StudyStore.activeRunId(this));
    }

    private void monitorCaptureLiveness() {
        if (isFinishing() || !captureReady) {
            return;
        }
        if (!captureRunIsActive()) {
            finishAfterCaptureLoss();
            return;
        }
        gestureView.postDelayed(this::monitorCaptureLiveness, 500L);
    }

    private void finishAfterCaptureLoss() {
        captureReady = false;
        setCaptureControlsEnabled(false);
        StudyStore.appendTaskEvent(
                this,
                "fewshot_calibration",
                "capture_lost",
                ACTIONS[Math.min(actionIndex, ACTIONS.length - 1)],
                posture,
                runId);
        Toast.makeText(
                this,
                "Sensor capture stopped, so this run was aborted. Please run the "
                        + "whole five-shot set again.",
                Toast.LENGTH_LONG).show();
        finish();
    }
}
