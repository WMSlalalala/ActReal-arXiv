package com.sensorworldmodel.collector;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Locale;
import java.util.UUID;

/**
 * Hosts the three simulated apps (Amazon, Google, WhatsApp). This class owns the
 * capture lifecycle, the raw touch recorder and the redacted keystroke logging;
 * the individual screens live in {@link AmazonSim}, {@link GoogleSim} and
 * {@link WhatsAppSim}.
 *
 * <p>The task ids below are part of the exported data schema and are consumed by
 * the desktop pipeline (pc/*.py), so they must not be renamed.
 */
public final class SimulatedTaskActivity extends Activity
        implements RawTouchRecorder.Listener {
    public static final String EXTRA_TASK = "simulated_task";
    public static final String EXTRA_POSTURE = "posture";
    public static final String SHOPPING = "simulated_shopping";
    public static final String SEARCH = "simulated_search";
    public static final String SOCIAL = "simulated_social";

    String task;
    private String phase;
    private String posture;
    private String runId;
    private RawTouchRecorder touchRecorder;
    private FrameLayout content;
    private Runnable progressListener;

    int taps;
    int scrolls;
    int swipes;
    int pinches;
    int textSubmissions;
    int searchRounds;
    int openedResults;
    int sentMessages;
    int cartItems;
    int chatsVisited;

    private long typingStartElapsedNs = -1L;
    private long typingStartWallMs = -1L;
    private String typingEventId = "";
    private int beforeTextCount;
    private boolean suppressTextWatcher;
    private boolean completed;
    private boolean captureReady;
    private boolean captureRequested;

    private AmazonSim amazon;
    private GoogleSim google;
    private WhatsAppSim whatsApp;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        task = getIntent().getStringExtra(EXTRA_TASK);
        posture = getIntent().getStringExtra(EXTRA_POSTURE);
        if (!"walking".equals(posture)) {
            posture = "sitting";
        }
        runId = StudyStore.newRunId(task);
        if (!SHOPPING.equals(task) && !SEARCH.equals(task) && !SOCIAL.equals(task)) {
            finish();
            return;
        }
        if (!StudyStore.hasSession(this)) {
            Toast.makeText(this, "Create a study session first.", Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        touchRecorder = new RawTouchRecorder(this, task, this);
        amazon = new AmazonSim(this);
        google = new GoogleSim(this);
        whatsApp = new WhatsAppSim(this);
        buildShell();
        startCapture();
        showCaptureWarmup();
        awaitCaptureReady(0);
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        if (!captureReady || !captureRunIsActive()) {
            return true;
        }
        if (touchRecorder != null) {
            touchRecorder.observe(event);
        }
        return super.dispatchTouchEvent(event);
    }

    @Override
    public void onGestureRecorded(String action) {
        if (!shouldCountGesture(action)) {
            StudyStore.appendTaskEvent(
                    this,
                    task,
                    "gesture_not_counted",
                    "action=" + action + ";phase=" + phase,
                    posture,
                    runId);
            return;
        }
        if ("tap".equals(action)) {
            taps++;
        } else if ("scroll".equals(action)) {
            scrolls++;
        } else if ("swipe".equals(action)) {
            swipes++;
        } else if ("pinch".equals(action)) {
            pinches++;
        }
        notifyProgress();
    }

    private boolean shouldCountGesture(String action) {
        if ("tap".equals(action)) {
            return true;
        }
        if (phase == null) {
            return false;
        }
        if (SHOPPING.equals(task)) {
            if ("scroll".equals(action)) {
                return "amazon_results".equals(phase) || "amazon_product".equals(phase);
            }
            if ("swipe".equals(action) || "pinch".equals(action)) {
                return "amazon_product".equals(phase);
            }
            return false;
        }
        if (SEARCH.equals(task)) {
            if ("scroll".equals(action)) {
                return phase.startsWith("google_results_") || phase.startsWith("google_article_");
            }
            // Every article carries a zoomable figure, so a participant who forgot
            // the pinch in round 1 can still complete it in the last round.
            return "pinch".equals(action) && phase.startsWith("google_article_");
        }
        if (SOCIAL.equals(task)) {
            return "scroll".equals(action) && "whatsapp_chat".equals(phase);
        }
        return false;
    }

    @Override
    protected void onDestroy() {
        captureReady = false;
        if (isFinishing() && captureRequested) {
            stopCapture(completed ? "task_activity_closed" : "task_aborted");
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

    // ---- shell ----------------------------------------------------------

    private void buildShell() {
        LinearLayout root = Ui.col(this);
        root.setBackgroundColor(Color.WHITE);
        UiInsets.applySystemBarsAndIme(root, 0, 0, 0, 0);
        content = new FrameLayout(this);
        root.addView(content, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);
        installBackHandler();
    }

    /**
     * Where the system Back button should go from the screen now showing.
     *
     * Nothing set it before, so Back fell through to the Activity default and
     * finished the task outright -- no dialog, no warning, and re-entering
     * started the run again from round one. An agent following an instruction
     * to "go back and search again" therefore lost every round it had done,
     * which reads from outside as the agent repeating itself for no reason.
     */
    private Runnable backAction;

    void setBackAction(Runnable action) {
        backAction = action;
    }

    /**
     * Registered as well as {@link #onBackPressed}, because which of the two the
     * platform calls depends on the predictive-back setting and this app targets
     * an SDK where that defaults on.  Registering both is cheap and removes the
     * need to be right about it.
     */
    private void installBackHandler() {
        if (android.os.Build.VERSION.SDK_INT >= 33) {
            getOnBackInvokedDispatcher().registerOnBackInvokedCallback(
                    android.window.OnBackInvokedDispatcher.PRIORITY_DEFAULT,
                    this::handleBack);
        }
    }

    private void handleBack() {
        Runnable action = backAction;
        if (action != null) {
            action.run();
            return;
        }
        confirmExit();
    }

    @Override
    public void onBackPressed() {
        Runnable action = backAction;
        if (action != null) {
            action.run();
            return;
        }
        // At the top of a simulated app there is nowhere further back to go, so
        // this is a request to leave -- which is worth confirming rather than
        // doing silently, exactly as the on-screen close button already does.
        confirmExit();
    }

    /**
     * Makes the keyboard's own action key submit, the way every real app does.
     *
     * These fields had no editor action at all: pressing Enter, or the search
     * key, did nothing whatsoever, and the only way to submit was the on-screen
     * button. Measured against the agent, every search cost an extra iteration
     * discovering that again.
     */
    void submitOn(EditText input, Runnable action) {
        input.setImeOptions(android.view.inputmethod.EditorInfo.IME_ACTION_GO);
        input.setOnEditorActionListener((view, actionId, event) -> {
            boolean go = actionId == android.view.inputmethod.EditorInfo.IME_ACTION_GO
                    || actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEARCH
                    || actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND
                    || actionId == android.view.inputmethod.EditorInfo.IME_ACTION_DONE
                    || (event != null
                        && event.getKeyCode() == android.view.KeyEvent.KEYCODE_ENTER
                        && event.getAction() == android.view.KeyEvent.ACTION_DOWN);
            if (!go) {
                return false;
            }
            action.run();
            return true;
        });
    }

    /** Lets a simulated app refresh its checklist after any recorded action. */
    void setProgressListener(Runnable listener) {
        progressListener = listener;
        notifyProgress();
    }

    private void notifyProgress() {
        if (progressListener != null) {
            progressListener.run();
        }
    }

    void confirmExit() {
        new AlertDialog.Builder(this)
                .setTitle("Leave this task?")
                .setMessage("The current run will be marked as aborted and will not "
                        + "count towards the study. You can start it again afterwards.")
                .setNegativeButton("Keep going", null)
                .setPositiveButton("Leave", (dialog, which) -> finish())
                .show();
    }

    /** Tints the status bar to match the app being simulated. */
    void setChrome(int statusBarColor, boolean darkIcons) {
        getWindow().setStatusBarColor(statusBarColor);
        android.view.WindowInsetsController controller =
                getWindow().getInsetsController();
        if (controller != null) {
            controller.setSystemBarsAppearance(
                    darkIcons
                            ? android.view.WindowInsetsController
                                    .APPEARANCE_LIGHT_STATUS_BARS
                            : 0,
                    android.view.WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS);
        }
    }

    void show(View view) {
        content.removeAllViews();
        content.addView(view, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));
    }

    void setPhase(String value) {
        phase = value;
        if (touchRecorder != null) {
            touchRecorder.setPhase(value);
        }
        Intent intent = new Intent(this, CaptureService.class)
                .setAction(CaptureService.ACTION_PHASE)
                .putExtra(CaptureService.EXTRA_PHASE, value)
                .putExtra(CaptureService.EXTRA_RUN_ID, runId);
        startService(intent);
    }

    String phase() {
        return phase;
    }

    int cartCount() {
        return cartItems;
    }

    void addToCart() {
        cartItems++;
        notifyProgress();
    }

    void chatOpened() {
        chatsVisited++;
        notifyProgress();
    }

    void toast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }

    void focus(EditText input) {
        input.requestFocus();
        input.postDelayed(() -> {
            InputMethodManager manager =
                    (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
            manager.showSoftInput(input, InputMethodManager.SHOW_IMPLICIT);
        }, 250);
    }

    void hideKeyboard(View view) {
        InputMethodManager manager =
                (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        manager.hideSoftInputFromWindow(view.getWindowToken(), 0);
    }

    // ---- redacted keystroke capture -------------------------------------

    /**
     * An EditText whose edit timing and character counts are logged while the
     * typed characters themselves are never written to disk.
     */
    EditText trackedInput(String hint) {
        EditText input = new EditText(this);
        input.setHint(hint);
        input.setSingleLine(true);
        // A Latin keyboard with autocorrect and word suggestions switched off, so
        // the recorded keystroke stream is the participant's own typing rather
        // than the IME completing words for them.
        input.setInputType(android.text.InputType.TYPE_CLASS_TEXT
                | android.text.InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
                | android.text.InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS);
        input.setTypeface(android.graphics.Typeface.DEFAULT);
        input.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence text, int start, int count, int after) {
                beforeTextCount = text.length();
            }

            @Override
            public void onTextChanged(CharSequence text, int start, int before, int count) {
                if (suppressTextWatcher || !captureReady || !captureRunIsActive()) {
                    return;
                }
                if (typingStartElapsedNs < 0) {
                    typingStartElapsedNs = SystemClock.elapsedRealtimeNanos();
                    typingStartWallMs = System.currentTimeMillis();
                    typingEventId = "key_" + UUID.randomUUID().toString().substring(0, 12);
                }
                StudyStore.appendCsv(
                        SimulatedTaskActivity.this,
                        "keystroke.csv",
                        StudyStore.KEYSTROKE_HEADER,
                        StudyStore.SCHEMA,
                        StudyStore.sessionId(SimulatedTaskActivity.this),
                        StudyStore.profileId(SimulatedTaskActivity.this),
                        task,
                        phase,
                        typingEventId,
                        SystemClock.elapsedRealtimeNanos(),
                        System.currentTimeMillis(),
                        beforeTextCount,
                        text.length(),
                        count,
                        before,
                        StudyStore.activePosture(SimulatedTaskActivity.this),
                        StudyStore.activeRunId(SimulatedTaskActivity.this)
                );
            }

            @Override
            public void afterTextChanged(Editable text) {}
        });
        return input;
    }

    /**
     * Validates the typed value against the exact phrase the protocol asked for,
     * then writes the redacted keystroke summary. Enforcing the phrase keeps the
     * typed content identical across participants.
     */
    boolean submitText(EditText input, String purpose, String expected) {
        if (!captureReady || !captureRunIsActive()) {
            finishAfterCaptureLoss();
            return false;
        }
        String value = input.getText().toString();
        if (typingStartElapsedNs < 0 || value.trim().isEmpty()) {
            toast("Type the phrase shown above first.");
            return false;
        }
        if (expected != null && !normalize(value).equals(normalize(expected))) {
            toast("Please type it exactly: " + expected);
            return false;
        }
        long endNs = SystemClock.elapsedRealtimeNanos();
        long endWall = System.currentTimeMillis();
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
                task,
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
                orientationId(),
                value.length(),
                nLetters,
                "duration,orientation,n_keys,n_letters",
                "purpose=" + purpose + ";prompted_length="
                        + (expected == null ? -1 : expected.length())
                        + ";typed_content_not_stored",
                StudyStore.displayWidthPx(this),
                StudyStore.displayHeightPx(this),
                StudyStore.densityDpi(this),
                StudyStore.activePosture(this),
                StudyStore.activeRunId(this)
        );
        textSubmissions++;
        suppressTextWatcher = true;
        input.setText("");
        suppressTextWatcher = false;
        typingStartElapsedNs = -1L;
        typingStartWallMs = -1L;
        typingEventId = "";
        notifyProgress();
        return true;
    }

    private static String normalize(String value) {
        return value.trim().toLowerCase(Locale.US).replaceAll("\\s+", " ");
    }

    // ---- completion -----------------------------------------------------

    void finishIfValid() {
        if (!captureReady || !captureRunIsActive()) {
            finishAfterCaptureLoss();
            return;
        }
        String missing = missingRequirements();
        if (!missing.isEmpty()) {
            Toast.makeText(this, "Still needed: " + missing, Toast.LENGTH_LONG).show();
            return;
        }
        completed = true;
        StudyStore.appendTaskEvent(
                this,
                task,
                "task_complete",
                String.format(
                        Locale.US,
                        "tap=%d;scroll=%d;swipe=%d;pinch=%d;text=%d;search_rounds=%d;"
                                + "opened=%d;sent=%d;cart=%d;chats=%d",
                        taps, scrolls, swipes, pinches, textSubmissions,
                        searchRounds, openedResults, sentMessages,
                        cartItems, chatsVisited));
        Toast.makeText(
                this,
                "Task complete. The data was written to the current session.",
                Toast.LENGTH_LONG).show();
        finish();
    }

    private String missingRequirements() {
        if (SHOPPING.equals(task)) {
            if (textSubmissions < AmazonSim.ROUNDS) {
                return AmazonSim.ROUNDS + " searches";
            }
            if (cartItems < AmazonSim.ROUNDS) {
                return AmazonSim.ROUNDS + " items in the cart";
            }
            if (scrolls < 2) return "2 vertical scrolls";
            if (swipes < 1) return "1 horizontal swipe on the photo strip";
            if (pinches < 1) return "1 pinch-zoom on the product photo";
            if (taps < 4) return "4 taps";
        } else if (SEARCH.equals(task)) {
            if (searchRounds < GoogleSim.ROUNDS || textSubmissions < GoogleSim.ROUNDS) {
                return GoogleSim.ROUNDS + " searches";
            }
            if (openedResults < GoogleSim.ROUNDS) return "open one result per search";
            if (scrolls < 2) return "2 reading scrolls";
            if (pinches < 1) return "1 pinch-zoom on the chart";
        } else {
            int wanted = WhatsAppSim.MESSAGES * WhatsAppSim.CONTACTS;
            if (sentMessages < wanted || textSubmissions < wanted) {
                return wanted + " messages";
            }
            if (chatsVisited < WhatsAppSim.CONTACTS) {
                return "open all " + WhatsAppSim.CONTACTS + " chats";
            }
            if (scrolls < 2) return "2 scrolls through the history";
            if (taps < 4) return "4 taps";
        }
        return "";
    }

    // ---- capture lifecycle ----------------------------------------------

    private void startCapture() {
        Intent intent = new Intent(this, CaptureService.class)
                .setAction(CaptureService.ACTION_START)
                .putExtra(CaptureService.EXTRA_TASK, task)
                .putExtra(CaptureService.EXTRA_PHASE, "starting")
                .putExtra(CaptureService.EXTRA_POSTURE, posture)
                .putExtra(CaptureService.EXTRA_RUN_ID, runId);
        captureRequested = true;
        startForegroundService(intent);
    }

    private void showCaptureWarmup() {
        LinearLayout panel = Ui.col(this);
        panel.setGravity(Gravity.CENTER);
        panel.setPadding(Ui.dp(this, 40), 0, Ui.dp(this, 40), 0);
        TextView waiting = Ui.text(this, "Warming up IMU capture", 20, Ui.INK, true);
        waiting.setGravity(Gravity.CENTER);
        panel.addView(waiting, Ui.matchWrap());
        TextView detail = Ui.text(
                this,
                "The task opens as soon as the accelerometer and gyroscope are streaming.",
                14,
                Ui.INK_SOFT);
        detail.setGravity(Gravity.CENTER);
        panel.addView(detail, Ui.margins(this, Ui.matchWrap(), 0, 10, 0, 0));
        show(panel);
    }

    private void awaitCaptureReady(int attempt) {
        if (isFinishing()) {
            return;
        }
        if (CaptureService.isCaptureReadyInProcess(runId)) {
            content.postDelayed(() -> {
                if (isFinishing()) {
                    return;
                }
                if (!CaptureService.isCaptureReadyInProcess(runId)) {
                    finishAfterCaptureLoss();
                    return;
                }
                captureReady = true;
                StudyStore.appendTaskEvent(this, task, "task_opened", taskTitle());
                if (SHOPPING.equals(task)) {
                    amazon.showHome();
                } else if (SEARCH.equals(task)) {
                    google.showHome();
                } else {
                    whatsApp.showChatList();
                }
                monitorCaptureLiveness();
            }, 300L);
            return;
        }
        if (attempt >= 50) {
            Toast.makeText(
                    this,
                    "The sensor service did not start within 5 seconds, so this run was "
                            + "aborted. Check the notification permission and the sensors.",
                    Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        content.postDelayed(() -> awaitCaptureReady(attempt + 1), 100L);
    }

    private void stopCapture(String reason) {
        StudyStore.appendTaskEvent(this, task, reason, phase, posture, runId);
        Intent stop = new Intent(this, CaptureService.class)
                .setAction(CaptureService.ACTION_STOP)
                .putExtra(CaptureService.EXTRA_RUN_ID, runId);
        startService(stop);
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
        content.postDelayed(this::monitorCaptureLiveness, 500L);
    }

    private void finishAfterCaptureLoss() {
        captureReady = false;
        StudyStore.appendTaskEvent(this, task, "capture_lost", phase, posture, runId);
        Toast.makeText(
                this,
                "Sensor capture stopped, so this run was aborted. Please run the task again.",
                Toast.LENGTH_LONG).show();
        finish();
    }

    // ---- misc -----------------------------------------------------------

    private String taskTitle() {
        if (SHOPPING.equals(task)) return "Shopping app simulation";
        if (SEARCH.equals(task)) return "Search app simulation";
        return "Messaging app simulation";
    }

    private int orientationId() {
        int rotation = StudyStore.displayRotation(this);
        return rotation == Surface.ROTATION_180 ? 0 : rotation;
    }
}
