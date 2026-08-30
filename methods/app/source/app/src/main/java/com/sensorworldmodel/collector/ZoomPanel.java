package com.sensorworldmodel.collector;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.view.MotionEvent;
import android.view.ViewConfiguration;
import android.view.View;
import android.view.ViewParent;

/**
 * Shows a product / article image that can be pinch-zoomed. Multi-touch is kept
 * away from the enclosing ScrollView so that the pinch is not swallowed, while a
 * single-finger drag is still allowed to scroll the page.
 */
public final class ZoomPanel extends View {
    private final Paint hintPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hintBackground = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Rect drawBounds = new Rect();
    private Drawable image;
    private float scale = 1f;
    private float priorSpan = Float.NaN;
    private float downX;
    private float downY;
    private boolean released;
    private boolean zoomed;
    private String hint = "Pinch to zoom";

    public ZoomPanel(Context context) {
        super(context);
        setBackgroundColor(Color.WHITE);
        hintPaint.setColor(Color.parseColor("#565959"));
        hintPaint.setTextAlign(Paint.Align.CENTER);
        hintPaint.setTextSize(Ui.dp(context, 12));
        hintBackground.setColor(Color.parseColor("#F2F3F3"));
        setContentDescription("Image, pinch with two fingers to zoom");
        setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_YES);
    }

    public void setImage(int drawableRes) {
        image = getContext().getDrawable(drawableRes);
        scale = 1f;
        zoomed = false;
        invalidate();
    }

    public void setHint(String value) {
        hint = value;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (image == null) {
            return;
        }
        int width = getWidth();
        int height = getHeight();
        int intrinsicWidth = Math.max(1, image.getIntrinsicWidth());
        int intrinsicHeight = Math.max(1, image.getIntrinsicHeight());
        float base = Math.min(
                width / (float) intrinsicWidth, height / (float) intrinsicHeight);
        float drawWidth = intrinsicWidth * base * scale;
        float drawHeight = intrinsicHeight * base * scale;
        drawBounds.set(
                Math.round(width / 2f - drawWidth / 2f),
                Math.round(height / 2f - drawHeight / 2f),
                Math.round(width / 2f + drawWidth / 2f),
                Math.round(height / 2f + drawHeight / 2f));
        canvas.save();
        canvas.clipRect(0, 0, width, height);
        image.setBounds(drawBounds);
        image.draw(canvas);
        canvas.restore();

        if (!zoomed && hint != null && !hint.isEmpty()) {
            float padX = Ui.dp(getContext(), 9);
            float padY = Ui.dp(getContext(), 5);
            float textWidth = hintPaint.measureText(hint);
            float centerX = width / 2f;
            float bottom = height - Ui.dp(getContext(), 8);
            float top = bottom - hintPaint.getTextSize() - padY * 2;
            float radius = (bottom - top) / 2f;
            canvas.drawRoundRect(
                    centerX - textWidth / 2f - padX,
                    top,
                    centerX + textWidth / 2f + padX,
                    bottom,
                    radius,
                    radius,
                    hintBackground);
            canvas.drawText(hint, centerX, bottom - padY - Ui.dp(getContext(), 1.5f), hintPaint);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        int masked = event.getActionMasked();
        if (masked == MotionEvent.ACTION_DOWN) {
            // Held from the first contact, because asking only when the second
            // finger arrives is too late: the enclosing ScrollView decides the
            // gesture is a scroll on the first finger's movement, and once it
            // has, the second finger never reaches this view and a pinch does
            // nothing.  Measured against the agent: pinches aimed squarely at
            // the chart, and a chart that did not move.
            //
            // Held, not kept.  Claiming it outright broke the other half --
            // a one-finger drag starting on the chart stopped scrolling the
            // article, and the run failed three times over on a page that
            // would not move.  So the claim is released again below the moment
            // the gesture proves to be a drag rather than a pinch.
            disallowParentIntercept(true);
            downX = event.getX();
            downY = event.getY();
            released = false;
        }
        if (masked == MotionEvent.ACTION_POINTER_DOWN) {
            disallowParentIntercept(true);
            released = false;
        }
        if (masked == MotionEvent.ACTION_MOVE
                && event.getPointerCount() == 1
                && !released
                && Math.hypot(event.getX() - downX, event.getY() - downY)
                    > ViewConfiguration.get(getContext()).getScaledTouchSlop()) {
            // One finger, and it has travelled: this is a scroll, and it
            // belongs to the view that scrolls.
            released = true;
            disallowParentIntercept(false);
            return false;
        }
        if (event.getPointerCount() >= 2) {
            float span = (float) Math.hypot(
                    event.getX(1) - event.getX(0),
                    event.getY(1) - event.getY(0));
            if (Float.isFinite(priorSpan) && priorSpan > 0f) {
                scale = Math.max(1f, Math.min(3f, scale * span / priorSpan));
                zoomed = true;
                invalidate();
            }
            priorSpan = span;
        }
        if (masked == MotionEvent.ACTION_UP || masked == MotionEvent.ACTION_CANCEL) {
            priorSpan = Float.NaN;
            disallowParentIntercept(false);
            if (masked == MotionEvent.ACTION_UP) {
                performClick();
            }
        }
        return true;
    }

    private void disallowParentIntercept(boolean disallow) {
        ViewParent parent = getParent();
        if (parent != null) {
            parent.requestDisallowInterceptTouchEvent(disallow);
        }
    }

    @Override
    public boolean performClick() {
        super.performClick();
        return true;
    }
}
