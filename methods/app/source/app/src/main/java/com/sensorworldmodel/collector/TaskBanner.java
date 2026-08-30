package com.sensorworldmodel.collector;

import android.content.Context;
import android.graphics.Color;
import android.graphics.Typeface;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.text.style.StyleSpan;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The in-screen task checklist. It lives inside each simulated app, tinted with
 * that app's colours, so the instructions read as part of the interface rather
 * than as a separate overlay. Anything the participant must type or tap is
 * quoted in the label and gets highlighted.
 */
final class TaskBanner extends LinearLayout {
    private static final int DONE = Color.parseColor("#15803D");
    private static final int PENDING = Color.parseColor("#8A94A6");

    private final Context context;
    private final int accent;
    private final TextView counter;
    private TextView title;
    private final LinearLayout items;
    private final List<TextView> marks = new ArrayList<>();
    private final List<TextView> labels = new ArrayList<>();
    private String[] steps = new String[0];

    TaskBanner(Context context, int accent, int tint) {
        super(context);
        this.context = context;
        this.accent = accent;
        setOrientation(HORIZONTAL);
        setBackground(Ui.rounded(context, tint, 14));

        View stripe = new View(context);
        stripe.setBackgroundColor(accent);
        addView(stripe, new LayoutParams(Ui.dp(context, 4), LayoutParams.MATCH_PARENT));

        LinearLayout body = Ui.col(context);
        body.setPadding(
                Ui.dp(context, 13), Ui.dp(context, 11),
                Ui.dp(context, 12), Ui.dp(context, 12));

        LinearLayout head = Ui.row(context);
        title = Ui.text(context, "Your task", 11, accent, true);
        title.setAllCaps(true);
        title.setLetterSpacing(0.1f);
        head.addView(title, Ui.wrap());
        head.addView(Ui.flexSpacer(context));
        counter = Ui.text(context, "", 11, PENDING, true);
        head.addView(counter, Ui.wrap());
        body.addView(head, Ui.matchWrap());

        items = Ui.col(context);
        body.addView(items, Ui.margins(context, Ui.matchWrap(), 0, 8, 0, 0));

        addView(body, new LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f));
    }

    /** Names the round, so a three-part task reads as three short pages. */
    void setRound(int current, int total) {
        title.setText(String.format(Locale.US, "Your task  ·  round %d of %d", current, total));
    }

    void setSteps(String[] values) {
        steps = values;
        items.removeAllViews();
        marks.clear();
        labels.clear();
        for (String value : values) {
            LinearLayout row = Ui.row(context);
            row.setGravity(Gravity.TOP);

            TextView mark = Ui.text(context, "○", 13, PENDING, true);
            mark.setWidth(Ui.dp(context, 20));
            row.addView(mark, Ui.wrap());

            TextView label = Ui.text(context, value, 14, PENDING);
            label.setLineSpacing(0f, 1.2f);
            row.addView(label, Ui.weight(1f));

            marks.add(mark);
            labels.add(label);
            items.addView(row, Ui.margins(context, Ui.matchWrap(), 0, 0, 0,
                    Ui.dp(context, 1)));
        }
        setStates(new boolean[values.length]);
    }

    /** Marks each step done or not; the first unfinished step is the active one. */
    void setStates(boolean[] states) {
        int completed = 0;
        int active = -1;
        for (int index = 0; index < states.length; index++) {
            if (states[index]) {
                completed++;
            } else if (active < 0) {
                active = index;
            }
        }
        for (int index = 0; index < marks.size() && index < states.length; index++) {
            TextView mark = marks.get(index);
            TextView label = labels.get(index);
            boolean done = states[index];
            boolean current = index == active;
            mark.setText(done ? "✓" : current ? "▸" : "○");
            mark.setTextColor(done ? DONE : current ? accent : PENDING);
            label.setTextColor(done ? PENDING : current ? Ui.INK : PENDING);
            label.setTypeface(current ? Typeface.DEFAULT_BOLD : Typeface.DEFAULT);
            label.setText(current ? highlight(steps[index]) : steps[index]);
            label.setAlpha(done ? 0.75f : 1f);
        }
        counter.setText(String.format(
                Locale.US, "%d of %d", Math.min(completed + (active < 0 ? 0 : 0), states.length),
                states.length));
    }

    /** Colours the quoted phrase inside the active step. */
    private CharSequence highlight(String value) {
        int open = value.indexOf('"');
        int close = value.indexOf('"', open + 1);
        if (open < 0 || close < 0) {
            return value;
        }
        SpannableString text = new SpannableString(value);
        text.setSpan(new ForegroundColorSpan(accent), open, close + 1,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        text.setSpan(new StyleSpan(Typeface.BOLD), open, close + 1,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        return text;
    }

    static LinearLayout.LayoutParams params(Context context) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.setMargins(
                Ui.dp(context, 10), Ui.dp(context, 10),
                Ui.dp(context, 10), Ui.dp(context, 4));
        return params;
    }
}
