package com.sensorworldmodel.collector;

import android.os.Build;
import android.view.View;
import android.view.WindowInsets;

public final class UiInsets {
    private UiInsets() {}

    /**
     * Like {@link #applySystemBars}, but the bottom padding also clears the
     * on-screen keyboard. Needed because the app targets SDK 35 and therefore
     * draws edge to edge, so the IME would otherwise cover a bottom composer or
     * a submit button.
     */
    public static void applySystemBarsAndIme(
            View view,
            int left,
            int top,
            int right,
            int bottom) {
        view.setOnApplyWindowInsetsListener((target, insets) -> {
            android.graphics.Insets safe = insets.getInsets(
                    WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
            android.graphics.Insets ime = insets.getInsets(WindowInsets.Type.ime());
            target.setPadding(
                    left + safe.left,
                    top + safe.top,
                    right + safe.right,
                    bottom + Math.max(safe.bottom, ime.bottom));
            return insets;
        });
        requestInsets(view);
    }

    private static void requestInsets(View view) {
        if (view.isAttachedToWindow()) {
            view.requestApplyInsets();
        } else {
            view.addOnAttachStateChangeListener(new View.OnAttachStateChangeListener() {
                @Override
                public void onViewAttachedToWindow(View attached) {
                    attached.removeOnAttachStateChangeListener(this);
                    attached.requestApplyInsets();
                }

                @Override
                public void onViewDetachedFromWindow(View detached) {}
            });
        }
    }

    @SuppressWarnings("deprecation")
    public static void applySystemBars(
            View view,
            int left,
            int top,
            int right,
            int bottom) {
        view.setOnApplyWindowInsetsListener((target, insets) -> {
            int insetLeft;
            int insetTop;
            int insetRight;
            int insetBottom;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                android.graphics.Insets safe = insets.getInsets(
                        WindowInsets.Type.systemBars()
                                | WindowInsets.Type.displayCutout());
                insetLeft = safe.left;
                insetTop = safe.top;
                insetRight = safe.right;
                insetBottom = safe.bottom;
            } else {
                insetLeft = insets.getSystemWindowInsetLeft();
                insetTop = insets.getSystemWindowInsetTop();
                insetRight = insets.getSystemWindowInsetRight();
                insetBottom = insets.getSystemWindowInsetBottom();
            }
            target.setPadding(
                    left + insetLeft,
                    top + insetTop,
                    right + insetRight,
                    bottom + insetBottom);
            return insets;
        });
        if (view.isAttachedToWindow()) {
            view.requestApplyInsets();
        } else {
            view.addOnAttachStateChangeListener(new View.OnAttachStateChangeListener() {
                @Override
                public void onViewAttachedToWindow(View attached) {
                    attached.removeOnAttachStateChangeListener(this);
                    attached.requestApplyInsets();
                }

                @Override
                public void onViewDetachedFromWindow(View detached) {}
            });
        }
    }
}
