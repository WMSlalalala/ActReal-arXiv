package com.actreal.target;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Locale;

/**
 * The surface actions land on, and a live read-out of what arrived.
 *
 * <p>The outlined rectangle is the part of this screen that has a coordinate
 * in the trajectory's own screen. Anything outside it cannot be expressed to
 * the detectors, so the host refuses to aim there and this view makes the
 * boundary visible rather than implicit.
 */
public final class TargetView extends View {

    /** The screen every stored trajectory was recorded on. */
    public static final float SOURCE_W = 1080f;
    public static final float SOURCE_H = 1920f;

    private static final int TRAIL_LIMIT = 512;

    private final Paint framePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint gridPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint trailPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint barPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Deque<float[]> trail = new ArrayDeque<>();
    private final RectF usable = new RectF();

    private String status = "";

    public TargetView(Context context) {
        super(context);
        setBackgroundColor(Color.parseColor("#101014"));

        framePaint.setStyle(Paint.Style.STROKE);
        framePaint.setStrokeWidth(3f);
        framePaint.setColor(Color.parseColor("#4C8DFF"));

        gridPaint.setStyle(Paint.Style.STROKE);
        gridPaint.setStrokeWidth(1f);
        gridPaint.setColor(Color.parseColor("#232733"));

        trailPaint.setStyle(Paint.Style.FILL);
        trailPaint.setColor(Color.parseColor("#FF7043"));

        textPaint.setColor(Color.parseColor("#D6DAE3"));
        textPaint.setTextSize(30f);

        barPaint.setStyle(Paint.Style.FILL);
        barPaint.setColor(Color.parseColor("#1A1C24"));
    }

    public void setStatus(String value) {
        status = value == null ? "" : value;
        postInvalidateOnAnimation();
    }

    public void addTouch(float x, float y, float pressure) {
        synchronized (trail) {
            if (trail.size() >= TRAIL_LIMIT) {
                trail.pollFirst();
            }
            trail.addLast(new float[] {x, y, pressure});
        }
        postInvalidateOnAnimation();
    }

    public void clearTrail() {
        synchronized (trail) {
            trail.clear();
        }
        postInvalidateOnAnimation();
    }

    public RectF usableRect() {
        return new RectF(usable);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        float scale = Math.min(w / SOURCE_W, h / SOURCE_H);
        float mappedW = SOURCE_W * scale;
        float mappedH = SOURCE_H * scale;
        float left = (w - mappedW) / 2f;
        float top = (h - mappedH) / 2f;
        usable.set(left, top, left + mappedW, top + mappedH);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // The bars are the part of the screen the mapping cannot reach.
        canvas.drawRect(0, 0, getWidth(), usable.top, barPaint);
        canvas.drawRect(0, usable.bottom, getWidth(), getHeight(), barPaint);

        for (int i = 1; i < 6; i++) {
            float x = usable.left + usable.width() * i / 6f;
            canvas.drawLine(x, usable.top, x, usable.bottom, gridPaint);
        }
        for (int i = 1; i < 10; i++) {
            float y = usable.top + usable.height() * i / 10f;
            canvas.drawLine(usable.left, y, usable.right, y, gridPaint);
        }
        canvas.drawRect(usable, framePaint);

        synchronized (trail) {
            for (float[] point : trail) {
                float radius = 6f + 18f * Math.min(1f, Math.max(0f, point[2]));
                canvas.drawCircle(point[0], point[1], radius, trailPaint);
            }
        }

        float y = usable.top + 44f;
        for (String line : status.split("\n")) {
            canvas.drawText(line, usable.left + 24f, y, textPaint);
            y += 38f;
        }
    }

    public String describeGeometry() {
        return String.format(
                Locale.US,
                "display %dx%d  usable %.0f,%.0f-%.0f,%.0f",
                getWidth(),
                getHeight(),
                usable.left,
                usable.top,
                usable.right,
                usable.bottom);
    }
}
