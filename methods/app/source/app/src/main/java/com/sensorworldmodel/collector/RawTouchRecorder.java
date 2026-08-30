package com.sensorworldmodel.collector;

import android.content.Context;
import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.Surface;

import java.util.Locale;
import java.util.UUID;

/**
 * Observes Activity.dispatchTouchEvent without consuming the event. This gives the
 * study pages a complete copy of their own MotionEvent stream while the original
 * event still reaches the clicked/scrolled child view.
 */
public final class RawTouchRecorder {
    public interface Listener {
        void onGestureRecorded(String action);
    }

    private final Context context;
    private final String task;
    private final Listener listener;
    private String phase = "";
    private String eventId = "";
    private long startElapsedNs;
    private long startWallMs;
    private float startX;
    private float startY;
    private float endX;
    private float endY;
    private float lastMultiX;
    private float lastMultiY;
    private float firstMultiX;
    private float firstMultiY;
    private float maxDistance;
    private int maxPointers;
    private float pinchStartSpan = Float.NaN;
    private float pinchEndSpan = Float.NaN;

    public RawTouchRecorder(Context context, String task, Listener listener) {
        this.context = context;
        this.task = task;
        this.listener = listener;
    }

    public void setPhase(String value) {
        phase = value == null ? "" : value;
    }

    public void observe(MotionEvent event) {
        int masked = event.getActionMasked();
        if (masked == MotionEvent.ACTION_DOWN) {
            eventId = "task_" + UUID.randomUUID().toString().substring(0, 12);
            latchClockOffset();
            startElapsedNs = motionElapsedNs(event.getEventTimeNanos());
            startWallMs = wallForElapsed(startElapsedNs);
            startX = screenX(event, 0, -1);
            startY = screenY(event, 0, -1);
            endX = startX;
            endY = startY;
            lastMultiX = startX;
            lastMultiY = startY;
            firstMultiX = Float.NaN;
            firstMultiY = Float.NaN;
            maxDistance = 0f;
            maxPointers = 1;
            pinchStartSpan = Float.NaN;
            pinchEndSpan = Float.NaN;
        }
        if (eventId.isEmpty()) {
            return;
        }

        maxPointers = Math.max(maxPointers, event.getPointerCount());
        if (event.getPointerCount() >= 2) {
            float span = pointerSpan(event, -1);
            if (!Float.isFinite(pinchStartSpan)) {
                pinchStartSpan = span;
                firstMultiX = centroidX(event, -1);
                firstMultiY = centroidY(event, -1);
            }
            pinchEndSpan = span;
            lastMultiX = centroidX(event, -1);
            lastMultiY = centroidY(event, -1);
        }
        logHistorical(event);
        logCurrent(event);
        endX = centroidX(event, -1);
        endY = centroidY(event, -1);
        maxDistance = Math.max(maxDistance,
                (float) Math.hypot(endX - startX, endY - startY));

        if (masked == MotionEvent.ACTION_UP) {
            appendSummary(event);
            eventId = "";
        } else if (masked == MotionEvent.ACTION_CANCEL) {
            StudyStore.appendTaskEvent(
                    context, task, "motion_cancelled", eventId + ";phase=" + phase);
            eventId = "";
        }
    }

    private void appendSummary(MotionEvent event) {
        long endElapsedNs = motionElapsedNs(event.getEventTimeNanos());
        long endWallMs = wallForElapsed(endElapsedNs);
        double durationMs = Math.max(0, endElapsedNs - startElapsedNs) / 1_000_000.0;
        String action = classify(durationMs);
        float summaryStartX = startX;
        float summaryStartY = startY;
        float summaryEndX = endX;
        float summaryEndY = endY;
        if ("pinch".equals(action)) {
            // For a two-finger gesture the conditioning XY is its screen centroid.
            // Per-pointer start/end coordinates remain available in touch.csv.
            summaryStartX = firstMultiX;
            summaryStartY = firstMultiY;
            summaryEndX = lastMultiX;
            summaryEndY = lastMultiY;
        }
        StudyStore.appendCsv(
                context,
                "events.csv",
                StudyStore.EVENT_HEADER,
                StudyStore.SCHEMA,
                StudyStore.sessionId(context),
                StudyStore.profileId(context),
                task,
                eventId,
                "in_app_raw_motion",
                action,
                startElapsedNs,
                endElapsedNs,
                startWallMs,
                endWallMs,
                durationMs,
                maxPointers,
                "raw_exact",
                "raw_motion_event_screen_xy",
                summaryStartX,
                summaryStartY,
                summaryEndX,
                summaryEndY,
                null,
                null,
                null,
                null,
                orientationId(),
                null,
                null,
                "duration,orientation,raw_xy_trajectory,pressure,size,pointer_id",
                gestureNotes(),
                StudyStore.displayWidthPx(context),
                StudyStore.displayHeightPx(context),
                StudyStore.densityDpi(context),
                StudyStore.activePosture(context),
                StudyStore.activeRunId(context)
        );
        if (listener != null) {
            listener.onGestureRecorded(action);
        }
    }

    private void logHistorical(MotionEvent event) {
        for (int history = 0; history < event.getHistorySize(); history++) {
            long uptimeNs = event.getHistoricalEventTimeNanos(history);
            long elapsedNs = motionElapsedNs(uptimeNs);
            for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
                float x = screenX(event, pointer, history);
                float y = screenY(event, pointer, history);
                float pressure = event.getHistoricalPressure(pointer, history);
                float size = event.getHistoricalSize(pointer, history);
                appendTouch(event, pointer, elapsedNs, x, y, pressure, size, "HISTORY");
                // Same sample, kept in memory too, so a run can be checked
                // while it happens rather than only after the file is read.
                TouchLog.get().note(
                        event, pointer, uptimeNs, elapsedNs, x, y, pressure, size,
                        "HISTORY", true);
            }
        }
    }

    private void logCurrent(MotionEvent event) {
        long uptimeNs = event.getEventTimeNanos();
        long elapsedNs = motionElapsedNs(uptimeNs);
        String action = MotionEvent.actionToString(event.getAction());
        for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
            float x = screenX(event, pointer, -1);
            float y = screenY(event, pointer, -1);
            float pressure = event.getPressure(pointer);
            float size = event.getSize(pointer);
            appendTouch(event, pointer, elapsedNs, x, y, pressure, size, action);
            TouchLog.get().note(
                    event, pointer, uptimeNs, elapsedNs, x, y, pressure, size, action, false);
        }
    }

    private void appendTouch(
            MotionEvent event,
            int pointer,
            long elapsedNs,
            float x,
            float y,
            float pressure,
            float size,
            String motionAction) {
        StudyStore.appendCsv(
                context,
                "touch.csv",
                StudyStore.TOUCH_HEADER,
                StudyStore.SCHEMA,
                StudyStore.sessionId(context),
                StudyStore.profileId(context),
                task,
                phase,
                eventId,
                "",
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
                orientationId(),
                StudyStore.displayWidthPx(context),
                StudyStore.displayHeightPx(context),
                StudyStore.densityDpi(context),
                StudyStore.activePosture(context),
                StudyStore.activeRunId(context)
        );
    }

    private String classify(double durationMs) {
        if (maxPointers >= 2) {
            if (Float.isFinite(pinchStartSpan)
                    && Float.isFinite(pinchEndSpan)
                    && Math.abs(pinchEndSpan - pinchStartSpan) >= 30f) {
                return "pinch";
            }
            return "unclassified";
        }
        float dx = endX - startX;
        float dy = endY - startY;
        if (maxDistance <= 70f && durationMs <= 1_000) {
            return "tap";
        }
        float absDx = Math.abs(dx);
        float absDy = Math.abs(dy);
        if (Math.max(absDx, absDy) < 120f) {
            return "unclassified";
        }
        return absDx >= absDy ? "swipe" : "scroll";
    }

    private String gestureNotes() {
        if (maxPointers < 2) {
            return String.format(Locale.US, "max_distance_px=%.6f", maxDistance);
        }
        return String.format(
                Locale.US,
                "max_distance_px=%.6f;pinch_start_span_px=%.6f;pinch_end_span_px=%.6f",
                maxDistance,
                pinchStartSpan,
                pinchEndSpan);
    }

    private float screenX(MotionEvent event, int pointer, int history) {
        if (history < 0) {
            return event.getRawX(pointer);
        }
        float offset = event.getRawX(0) - event.getX(0);
        return offset + event.getHistoricalX(pointer, history);
    }

    private float screenY(MotionEvent event, int pointer, int history) {
        if (history < 0) {
            return event.getRawY(pointer);
        }
        float offset = event.getRawY(0) - event.getY(0);
        return offset + event.getHistoricalY(pointer, history);
    }

    private float centroidX(MotionEvent event, int history) {
        float value = 0f;
        for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
            value += screenX(event, pointer, history);
        }
        return value / event.getPointerCount();
    }

    private float centroidY(MotionEvent event, int history) {
        float value = 0f;
        for (int pointer = 0; pointer < event.getPointerCount(); pointer++) {
            value += screenY(event, pointer, history);
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

    private int orientationId() {
        int rotation = StudyStore.displayRotation(context);
        return rotation == Surface.ROTATION_180 ? 0 : rotation;
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
}
