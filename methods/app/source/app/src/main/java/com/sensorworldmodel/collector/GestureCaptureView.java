package com.sensorworldmodel.collector;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;

import java.util.Locale;
import java.util.UUID;

public final class GestureCaptureView extends View {
    public interface Listener {
        void onAccepted(String action, String eventId);
        void onRejected(String action, String reason);
    }

    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private Listener listener;
    private String labeledAction = "tap";
    private String eventId = "";
    private long startElapsedNs;
    private long startWallMs;
    private float startX;
    private float startY;
    private float endX;
    private float endY;
    private float maxDistance;
    private int maxPointers;
    private float pinchStartSpan = Float.NaN;
    private float pinchEndSpan = Float.NaN;

    public GestureCaptureView(Context context) {
        super(context);
        setBackgroundColor(Color.rgb(244, 247, 253));
        paint.setColor(Color.rgb(35, 87, 217));
        paint.setStyle(Paint.Style.FILL);
        textPaint.setColor(Color.rgb(35, 45, 65));
        textPaint.setTextAlign(Paint.Align.CENTER);
        textPaint.setTextSize(42f);
        setFocusable(true);
    }

    public void setListener(Listener value) {
        listener = value;
    }

    public void setLabeledAction(String value) {
        labeledAction = value;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float cx = getWidth() / 2f;
        float cy = getHeight() / 2f;
        if ("tap".equals(labeledAction)) {
            canvas.drawCircle(cx, cy, 70f, paint);
            canvas.drawText("Tap the blue dot", cx, cy + 130f, textPaint);
        } else if ("scroll".equals(labeledAction)) {
            canvas.drawText("Drag vertically once", cx, cy, textPaint);
        } else if ("swipe".equals(labeledAction)) {
            canvas.drawText("Swipe horizontally once", cx, cy, textPaint);
        } else if ("pinch".equals(labeledAction)) {
            canvas.drawText("Pinch with two fingers once", cx, cy, textPaint);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (!isEnabled()) {
            return true;
        }
        if ("keystroke".equals(labeledAction)) {
            return false;
        }
        int masked = event.getActionMasked();
        if (masked == MotionEvent.ACTION_DOWN) {
            eventId = "raw_" + UUID.randomUUID().toString().substring(0, 12);
            latchClockOffset();
            startElapsedNs = motionElapsedNs(event.getEventTimeNanos());
            startWallMs = wallForElapsed(startElapsedNs);
            startX = screenX(event, 0);
            startY = screenY(event, 0);
            endX = startX;
            endY = startY;
            maxDistance = 0f;
            maxPointers = 1;
            pinchStartSpan = Float.NaN;
            pinchEndSpan = Float.NaN;
        }
        if (eventId.isEmpty()) {
            return true;
        }

        maxPointers = Math.max(maxPointers, event.getPointerCount());
        if (event.getPointerCount() >= 2) {
            float span = pointerSpan(event, -1);
            if (!Float.isFinite(pinchStartSpan)) {
                pinchStartSpan = span;
            }
            pinchEndSpan = span;
        }
        logHistorical(event);
        logCurrent(event);
        endX = centroidX(event);
        endY = centroidY(event);
        maxDistance = Math.max(
                maxDistance,
                (float) Math.hypot(endX - startX, endY - startY));

        if (masked == MotionEvent.ACTION_UP || masked == MotionEvent.ACTION_CANCEL) {
            long endElapsedNs = motionElapsedNs(event.getEventTimeNanos());
            long endWallMs = wallForElapsed(endElapsedNs);
            double durationMs = (endElapsedNs - startElapsedNs) / 1_000_000.0;
            boolean accepted = masked == MotionEvent.ACTION_UP
                    && isValidGesture(durationMs);
            if (accepted) {
                StudyStore.appendCsv(
                        getContext(),
                        "events.csv",
                        StudyStore.EVENT_HEADER,
                        StudyStore.SCHEMA,
                        StudyStore.sessionId(getContext()),
                        StudyStore.profileId(getContext()),
                        "fewshot_calibration",
                        eventId,
                        "in_app_raw_motion",
                        labeledAction,
                        startElapsedNs,
                        endElapsedNs,
                        startWallMs,
                        endWallMs,
                        durationMs,
                        maxPointers,
                        "raw_exact",
                        "raw_motion_event_screen_xy",
                        startX,
                        startY,
                        endX,
                        endY,
                        null,
                        null,
                        null,
                        null,
                        orientationId(),
                        null,
                        null,
                        "duration,orientation,raw_xy_trajectory,pressure,size,pointer_id",
                        gestureNotes(),
                        StudyStore.displayWidthPx(getContext()),
                        StudyStore.displayHeightPx(getContext()),
                        StudyStore.densityDpi(getContext()),
                        StudyStore.activePosture(getContext()),
                        StudyStore.activeRunId(getContext())
                );
                if (listener != null) {
                    listener.onAccepted(labeledAction, eventId);
                }
            } else if (listener != null) {
                listener.onRejected(labeledAction, rejectionReason(durationMs));
            }
            eventId = "";
            if (masked == MotionEvent.ACTION_UP) {
                performClick();
            }
        }
        return true;
    }

    @Override
    public boolean performClick() {
        super.performClick();
        return true;
    }

    private void logHistorical(MotionEvent event) {
        for (int history = 0; history < event.getHistorySize(); history++) {
            long elapsedNs = motionElapsedNs(event.getHistoricalEventTimeNanos(history));
            for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
                appendTouch(
                        event,
                        history,
                        pointer,
                        elapsedNs,
                        getLeftOnScreen() + event.getHistoricalX(pointer, history),
                        getTopOnScreen() + event.getHistoricalY(pointer, history),
                        event.getHistoricalPressure(pointer, history),
                        event.getHistoricalSize(pointer, history),
                        "HISTORY");
            }
        }
    }

    private void logCurrent(MotionEvent event) {
        long elapsedNs = motionElapsedNs(event.getEventTimeNanos());
        for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
            appendTouch(
                    event,
                    -1,
                    pointer,
                    elapsedNs,
                    screenX(event, pointer),
                    screenY(event, pointer),
                    event.getPressure(pointer),
                    event.getSize(pointer),
                    MotionEvent.actionToString(event.getAction()));
        }
    }

    private void appendTouch(
            MotionEvent event,
            int history,
            int pointer,
            long elapsedNs,
            float x,
            float y,
            float pressure,
            float size,
            String motionAction) {
        StudyStore.appendCsv(
                getContext(),
                "touch.csv",
                StudyStore.TOUCH_HEADER,
                StudyStore.SCHEMA,
                StudyStore.sessionId(getContext()),
                StudyStore.profileId(getContext()),
                "fewshot_calibration",
                labeledAction,
                eventId,
                labeledAction,
                elapsedNs,
                wallForElapsed(elapsedNs),
                motionAction,
                event.getPointerCount(),
                pointer,
                event.getPointerId(pointer),
                x,
                y,
                x,
                y,
                pressure,
                size,
                StudyStore.displayRotation(getContext()),
                StudyStore.displayWidthPx(getContext()),
                StudyStore.displayHeightPx(getContext()),
                StudyStore.densityDpi(getContext()),
                StudyStore.activePosture(getContext()),
                StudyStore.activeRunId(getContext())
        );
    }

    private boolean isValidGesture(double durationMs) {
        if (durationMs < 0 || durationMs > 10_000) {
            return false;
        }
        if ("tap".equals(labeledAction)) {
            float targetX = getLeftOnScreen() + getWidth() / 2f;
            float targetY = getTopOnScreen() + getHeight() / 2f;
            return maxPointers == 1
                    && maxDistance <= 70f
                    && durationMs <= 1_000
                    && Math.hypot(startX - targetX, startY - targetY) <= 105f;
        }
        if ("scroll".equals(labeledAction)) {
            // Must also be predominantly vertical, otherwise a sideways drag is
            // accepted as a scroll here while RawTouchRecorder would classify the
            // very same motion as a swipe during the real tasks.
            return maxPointers == 1
                    && Math.abs(endY - startY) >= 120f
                    && Math.abs(endY - startY) > Math.abs(endX - startX);
        }
        if ("swipe".equals(labeledAction)) {
            return maxPointers == 1
                    && Math.abs(endX - startX) >= 120f
                    && Math.abs(endX - startX) >= Math.abs(endY - startY);
        }
        if ("pinch".equals(labeledAction)) {
            return maxPointers >= 2
                    && Float.isFinite(pinchStartSpan)
                    && Float.isFinite(pinchEndSpan)
                    && Math.abs(pinchEndSpan - pinchStartSpan) >= 30f;
        }
        return false;
    }

    private String rejectionReason(double durationMs) {
        return String.format(
                Locale.US,
                "Not accepted (%s). duration=%.0fms  dx=%.0fpx  dy=%.0fpx  fingers=%d",
                whyRejected(durationMs),
                durationMs,
                Math.abs(endX - startX),
                Math.abs(endY - startY),
                maxPointers,
                Float.isFinite(pinchStartSpan) && Float.isFinite(pinchEndSpan)
                        ? Math.abs(pinchEndSpan - pinchStartSpan) : 0f);
    }

    /** A short, specific reason so the participant knows what to change. */
    private String whyRejected(double durationMs) {
        float dx = Math.abs(endX - startX);
        float dy = Math.abs(endY - startY);
        if (durationMs < 0 || durationMs > 10_000) {
            return "took too long";
        }
        if ("tap".equals(labeledAction)) {
            if (maxPointers > 1) return "use one finger";
            if (maxDistance > 70f) return "finger moved, keep it still";
            if (durationMs > 1_000) return "held too long";
            return "tap on the blue dot";
        }
        if ("scroll".equals(labeledAction)) {
            if (maxPointers > 1) return "use one finger";
            if (dx > dy) return "that was sideways, drag up or down instead";
            return "drag further up or down";
        }
        if ("swipe".equals(labeledAction)) {
            if (maxPointers > 1) return "use one finger";
            if (dy > dx) return "that was vertical, swipe left or right instead";
            return "swipe further left or right";
        }
        if ("pinch".equals(labeledAction)) {
            if (maxPointers < 2) return "use two fingers";
            return "move the fingers further apart or together";
        }
        return "try again";
    }

    private String gestureNotes() {
        if (!"pinch".equals(labeledAction)) {
            return "";
        }
        return String.format(
                Locale.US,
                "pinch_start_span_px=%.6f;pinch_end_span_px=%.6f",
                pinchStartSpan,
                pinchEndSpan);
    }

    private float screenX(MotionEvent event, int pointer) {
        return getLeftOnScreen() + event.getX(pointer);
    }

    private float screenY(MotionEvent event, int pointer) {
        return getTopOnScreen() + event.getY(pointer);
    }

    private float centroidX(MotionEvent event) {
        float value = 0f;
        for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
            value += screenX(event, pointer);
        }
        return value / event.getPointerCount();
    }

    private float centroidY(MotionEvent event) {
        float value = 0f;
        for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
            value += screenY(event, pointer);
        }
        return value / event.getPointerCount();
    }

    private float pointerSpan(MotionEvent event, int history) {
        float x0 = history < 0 ? event.getX(0) : event.getHistoricalX(0, history);
        float y0 = history < 0 ? event.getY(0) : event.getHistoricalY(0, history);
        float x1 = history < 0 ? event.getX(1) : event.getHistoricalX(1, history);
        float y1 = history < 0 ? event.getY(1) : event.getHistoricalY(1, history);
        return (float) Math.hypot(x1 - x0, y1 - y0);
    }

    private int getLeftOnScreen() {
        int[] location = new int[2];
        getLocationOnScreen(location);
        return location[0];
    }

    private int getTopOnScreen() {
        int[] location = new int[2];
        getLocationOnScreen(location);
        return location[1];
    }

    /**
     * MotionEvent timestamps live on the uptime clock; the IMU stream lives on
     * the elapsed-realtime clock. Sampling both clocks on every conversion adds
     * up to a millisecond of jitter, because uptimeMillis() is truncated to
     * milliseconds -- enough to make a fast tap come out with a negative
     * duration. The offset is therefore latched once per gesture and reused for
     * every sample in that gesture, so intervals within a gesture come straight
     * from the MotionEvent nanosecond clock.
     */
    private long motionElapsedNs(long motionEventNs) {
        return motionEventNs + clockOffsetNs;
    }

    /** Re-latches the uptime -> elapsed-realtime offset at the start of a gesture. */
    private void latchClockOffset() {
        long uptimeNs = SystemClock.uptimeMillis() * 1_000_000L;
        clockOffsetNs = SystemClock.elapsedRealtimeNanos() - uptimeNs;
    }

    private long clockOffsetNs;

    private static long wallForElapsed(long elapsedNs) {
        return System.currentTimeMillis()
                + Math.round((elapsedNs - SystemClock.elapsedRealtimeNanos()) / 1_000_000.0);
    }

    private int orientationId() {
        int rotation = StudyStore.displayRotation(getContext());
        return rotation == Surface.ROTATION_180 ? 0 : rotation;
    }
}
