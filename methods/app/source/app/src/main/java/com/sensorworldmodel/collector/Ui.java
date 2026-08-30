package com.sensorworldmodel.collector;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

/**
 * Small view/drawable toolkit shared by the three simulated apps. Everything is
 * built from framework classes only, because this module is compiled without
 * AndroidX (gradle.properties: android.useAndroidX=false).
 */
final class Ui {
    private Ui() {}

    // ---- brand palettes -------------------------------------------------

    static final int AMZ_NAVY = Color.parseColor("#131921");
    static final int AMZ_BAR = Color.parseColor("#232F3E");
    static final int AMZ_ORANGE = Color.parseColor("#FEBD69");
    static final int AMZ_YELLOW = Color.parseColor("#FFD814");
    static final int AMZ_BUYNOW = Color.parseColor("#FFA41C");
    static final int AMZ_BG = Color.parseColor("#EAEDED");
    static final int AMZ_LINK = Color.parseColor("#007185");
    static final int AMZ_STAR = Color.parseColor("#FFA41C");
    static final int AMZ_PRIME = Color.parseColor("#1D95C9");
    static final int AMZ_GREEN = Color.parseColor("#007600");
    static final int AMZ_RED = Color.parseColor("#CC0C39");

    static final int G_BLUE = Color.parseColor("#4285F4");
    static final int G_RED = Color.parseColor("#EA4335");
    static final int G_YELLOW = Color.parseColor("#FBBC05");
    static final int G_GREEN = Color.parseColor("#34A853");
    static final int G_LINK = Color.parseColor("#1A0DAB");
    static final int G_URL = Color.parseColor("#202124");
    static final int G_SNIPPET = Color.parseColor("#4D5156");
    static final int G_BORDER = Color.parseColor("#DFE1E5");
    static final int G_CHIP = Color.parseColor("#F1F3F4");

    static final int WA_TEAL = Color.parseColor("#075E54");
    static final int WA_TEAL_LIGHT = Color.parseColor("#128C7E");
    static final int WA_GREEN = Color.parseColor("#25D366");
    static final int WA_BUBBLE_OUT = Color.parseColor("#DCF8C6");
    static final int WA_WALLPAPER = Color.parseColor("#ECE5DD");
    static final int WA_TICK_BLUE = Color.parseColor("#34B7F1");
    static final int WA_SUBTITLE = Color.parseColor("#667781");

    static final int INK = Color.parseColor("#0F1111");
    static final int INK_SOFT = Color.parseColor("#565959");
    static final int WHITE = Color.WHITE;

    // ---- units ----------------------------------------------------------

    static int dp(Context context, float value) {
        return Math.round(TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP,
                value,
                context.getResources().getDisplayMetrics()));
    }

    // ---- containers -----------------------------------------------------

    static LinearLayout col(Context context) {
        LinearLayout layout = new LinearLayout(context);
        layout.setOrientation(LinearLayout.VERTICAL);
        return layout;
    }

    static LinearLayout row(Context context) {
        LinearLayout layout = new LinearLayout(context);
        layout.setOrientation(LinearLayout.HORIZONTAL);
        layout.setGravity(Gravity.CENTER_VERTICAL);
        return layout;
    }

    static LinearLayout.LayoutParams wrap() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    static LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    static LinearLayout.LayoutParams size(Context context, float widthDp, float heightDp) {
        return new LinearLayout.LayoutParams(dp(context, widthDp), dp(context, heightDp));
    }

    static LinearLayout.LayoutParams weight(float value) {
        return new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, value);
    }

    static LinearLayout.LayoutParams margins(
            Context context,
            LinearLayout.LayoutParams params,
            float left,
            float top,
            float right,
            float bottom) {
        params.setMargins(
                dp(context, left), dp(context, top), dp(context, right), dp(context, bottom));
        return params;
    }

    /**
     * A plain {@link View} measured with WRAP_CONTENT expands to fill the parent,
     * because {@code View.onMeasure} hands back the whole AT_MOST size. Spacers
     * and rules therefore pin their own height so they stay correct whatever
     * layout params a caller passes to {@code addView}.
     */
    private static final class FixedHeight extends View {
        private final int heightPx;

        FixedHeight(Context context, int heightPx) {
            super(context);
            this.heightPx = heightPx;
        }

        @Override
        protected void onMeasure(int widthSpec, int heightSpec) {
            setMeasuredDimension(getDefaultSize(getSuggestedMinimumWidth(), widthSpec),
                    heightPx);
        }
    }

    static View spacer(Context context, float heightDp) {
        View view = new FixedHeight(context, dp(context, heightDp));
        view.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(context, heightDp)));
        return view;
    }

    static View flexSpacer(Context context) {
        View view = new View(context);
        view.setLayoutParams(new LinearLayout.LayoutParams(0, 1, 1f));
        return view;
    }

    static View divider(Context context, int color) {
        int thickness = Math.max(1, dp(context, 0.7f));
        View view = new FixedHeight(context, thickness);
        view.setBackgroundColor(color);
        view.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, thickness));
        return view;
    }

    // ---- text -----------------------------------------------------------

    static TextView text(Context context, CharSequence value, float sizeSp, int color) {
        return text(context, value, sizeSp, color, false);
    }

    static TextView text(
            Context context, CharSequence value, float sizeSp, int color, boolean bold) {
        TextView view = new TextView(context);
        view.setText(value);
        view.setTextSize(sizeSp);
        view.setTextColor(color);
        view.setIncludeFontPadding(false);
        if (bold) {
            view.setTypeface(Typeface.DEFAULT_BOLD);
        }
        return view;
    }

    /** The word "Google" rendered with its per-letter brand colours. */
    static SpannableString googleWordmark(String word) {
        int[] colors = {G_BLUE, G_RED, G_YELLOW, G_BLUE, G_GREEN, G_RED};
        SpannableString value = new SpannableString(word);
        for (int index = 0; index < word.length(); index++) {
            value.setSpan(
                    new ForegroundColorSpan(colors[index % colors.length]),
                    index,
                    index + 1,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        return value;
    }

    // ---- drawables ------------------------------------------------------

    static GradientDrawable rounded(Context context, int color, float radiusDp) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(dp(context, radiusDp));
        return drawable;
    }

    static GradientDrawable rounded(
            Context context, int color, float radiusDp, int strokeColor, float strokeDp) {
        GradientDrawable drawable = rounded(context, color, radiusDp);
        drawable.setStroke(Math.max(1, dp(context, strokeDp)), strokeColor);
        return drawable;
    }

    static GradientDrawable circle(int color) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setShape(GradientDrawable.OVAL);
        drawable.setColor(color);
        return drawable;
    }

    // ---- images ---------------------------------------------------------

    static ImageView image(Context context, int drawableRes, float widthDp, float heightDp) {
        ImageView view = new ImageView(context);
        view.setImageResource(drawableRes);
        view.setScaleType(ImageView.ScaleType.FIT_CENTER);
        view.setLayoutParams(size(context, widthDp, heightDp));
        return view;
    }

    static ImageView icon(Context context, int drawableRes, float sizeDp, int tint) {
        ImageView view = new ImageView(context);
        view.setImageResource(drawableRes);
        view.setScaleType(ImageView.ScaleType.FIT_CENTER);
        if (tint != 0) {
            view.setColorFilter(tint);
        }
        view.setLayoutParams(size(context, sizeDp, sizeDp));
        return view;
    }

    /** Circular badge carrying one or two letters, used for result favicons. */
    static Drawable letterBadge(String letters, int background, int ink) {
        return new LetterDrawable(letters, background, ink);
    }

    private static final class LetterDrawable extends Drawable {
        private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint glyph = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final String letters;
        private final Rect textBounds = new Rect();

        LetterDrawable(String letters, int background, int ink) {
            this.letters = letters;
            fill.setColor(background);
            glyph.setColor(ink);
            glyph.setTypeface(Typeface.DEFAULT_BOLD);
            glyph.setTextAlign(Paint.Align.CENTER);
        }

        @Override
        public void draw(Canvas canvas) {
            Rect area = getBounds();
            if (area.width() <= 0 || area.height() <= 0) {
                return;
            }
            float radius = Math.min(area.width(), area.height()) / 2f;
            canvas.drawCircle(area.centerX(), area.centerY(), radius, fill);
            glyph.setTextSize(radius * (letters.length() > 1 ? 0.85f : 1.1f));
            glyph.getTextBounds(letters, 0, letters.length(), textBounds);
            canvas.drawText(
                    letters,
                    area.centerX(),
                    area.centerY() + textBounds.height() / 2f,
                    glyph);
        }

        @Override
        public void setAlpha(int alpha) {
            fill.setAlpha(alpha);
            glyph.setAlpha(alpha);
        }

        @Override
        public void setColorFilter(ColorFilter filter) {
            fill.setColorFilter(filter);
            glyph.setColorFilter(filter);
        }

        @Override
        public int getOpacity() {
            return PixelFormat.TRANSLUCENT;
        }
    }

    // ---- prompted-input gating ------------------------------------------

    /**
     * Runs the listener on every edit, so a submit control can stay disabled
     * until the participant has typed exactly the prompted phrase. Gating the
     * button this way avoids showing an error popup for a mistyped phrase.
     */
    static void onTextChange(android.widget.EditText input, Runnable listener) {
        input.addTextChangedListener(new android.text.TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence text, int a, int b, int c) {}

            @Override
            public void onTextChanged(CharSequence text, int a, int b, int c) {
                listener.run();
            }

            @Override
            public void afterTextChanged(android.text.Editable text) {}
        });
    }

    static String normalise(String value) {
        return value.trim().toLowerCase(java.util.Locale.US).replaceAll("\\s+", " ");
    }

    static boolean matches(android.widget.EditText input, String expected) {
        return normalise(input.getText().toString()).equals(normalise(expected));
    }

    /** Amazon-style five star strip for a rating between 0 and 5. */
    static LinearLayout stars(Context context, double rating, float sizeDp) {
        LinearLayout strip = row(context);
        for (int index = 1; index <= 5; index++) {
            int resource;
            if (rating >= index) {
                resource = R.drawable.ic_star_full;
            } else if (rating >= index - 0.5) {
                resource = R.drawable.ic_star_half;
            } else {
                resource = R.drawable.ic_star_empty;
            }
            strip.addView(icon(context, resource, sizeDp, 0));
        }
        return strip;
    }
}
