package com.sensorworldmodel.collector;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Shader;
import android.graphics.Typeface;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.LayerDrawable;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

/**
 * A local, offline messaging simulation styled after WhatsApp. No contacts, no
 * network, and no message text ever leaves the device: only the timing and the
 * character counts of the typing are recorded.
 */
final class WhatsAppSim {
    static final int MESSAGES = 3;
    /** How many different people the participant writes to. */
    static final int CONTACTS = 3;

    private static final int ACCENT = Color.parseColor("#128C7E");
    private static final int TINT = Color.parseColor("#E4F3EF");

    /** The exact messages sent in each conversation, three per contact. */
    private static final String[][] SCRIPT = {
        {"on my way", "see you at six", "thanks"},
        {"got the slides", "looks good to me", "talk tomorrow"},
        {"sounds good", "i will be there", "see you"},
    };

    /** The three chats to open, in this order. */
    private static final String[] TARGETS = {"Alex Chen", "Maya Patel", "Jonas Lind"};
    private static final String[] NAMES = {
        "Alex Chen", "Maya Patel", "Study Group", "Dad", "Jonas Lind", "Priya Raman",
        "Lab Rota", "Sam Okafor",
    };
    private static final int[] AVATARS = {
        R.drawable.avatar_1, R.drawable.avatar_2, R.drawable.avatar_3,
        R.drawable.avatar_4, R.drawable.avatar_5, R.drawable.avatar_6,
        R.drawable.avatar_7, R.drawable.avatar_8,
    };
    private static final String[] PREVIEWS = {
        "Are you heading over now?",
        "Sent you the slides",
        "Maya: see everyone at four",
        "Call me when you get a minute",
        "Photo",
        "Thanks for today!",
        "Priya: I can swap Friday",
        "Did you get the email?",
    };
    private static final String[] TIMES = {
        "13:42", "12:08", "11:20", "Yesterday", "Yesterday", "Tuesday",
        "Tuesday", "Monday",
    };
    private static final int[] UNREAD = {2, 0, 5, 0, 0, 0, 3, 0};

    /** Only the current contact is listed, so the card stays short. */
    private String[] buildSteps() {
        String[] out = new String[MESSAGES + 2];
        out[0] = "Open the chat with \"" + TARGETS[contact] + "\"";
        out[1] = "Scroll up through the older messages";
        for (int m = 0; m < MESSAGES; m++) {
            boolean lastOfAll = contact + 1 >= CONTACTS && m + 1 >= MESSAGES;
            out[m + 2] = "Send \"" + SCRIPT[contact][m] + "\""
                    + (lastOfAll ? ", then finish" : "");
        }
        return out;
    }

    /** A different history per contact so each chat reads as its own thread. */
    private static final String[][][] HISTORY = {
        {
            {"in", "Hey, are you still coming to the study session?", "10:12"},
            {"out", "Yes, I finished the forms this morning.", "10:14"},
            {"in", "Nice. Room 214 like last time.", "10:15"},
            {"out", "Got it. Is the room on the second floor?", "10:16"},
            {"in", "Second floor, past the lifts on the left.", "10:18"},
            {"photo", "", "10:19"},
            {"in", "That is the door. See it?", "10:19"},
            {"out", "Yes, I remember that corridor.", "10:24"},
            {"in", "Perfect. I will bring the consent sheets.", "10:26"},
            {"out", "Do you need me to bring anything?", "10:31"},
            {"in", "Just your phone and a charger.", "10:33"},
            {"out", "Sounds good, thanks for organising it.", "10:35"},
            {"in", "No problem. Ping me when you set off.", "13:12"},
            {"out", "Just finishing lunch, leaving in ten.", "13:20"},
            {"in", "No rush, the room is free until four.", "13:22"},
            {"out", "Is there parking near the building?", "13:30"},
            {"in", "Deck 3 is closest, first hour is free.", "13:33"},
            {"out", "Good to know, thanks.", "13:35"},
            {"in", "Are you heading over now?", "13:42"},
        },
        {
            {"in", "Did you get a chance to look at the deck?", "09:40"},
            {"out", "Opening it now.", "09:41"},
            {"in", "Slide 6 is the one I am unsure about.", "09:42"},
            {"out", "The one with the two charts?", "09:44"},
            {"in", "That one. Too much on a single slide?", "09:45"},
            {"out", "Maybe split it in two.", "09:50"},
            {"in", "That was my instinct as well.", "09:52"},
            {"out", "I can redo it after lunch.", "09:55"},
            {"in", "No rush, the review is on Thursday.", "10:02"},
            {"photo", "", "10:05"},
            {"in", "This is roughly the layout I had in mind.", "10:05"},
            {"out", "That works, much cleaner.", "10:20"},
            {"in", "I will send the updated file over.", "11:58"},
            {"in", "Sent you the slides", "12:08"},
        },
        {
            {"in", "Are we still on for the lab meeting?", "Yesterday"},
            {"out", "Yes, same time.", "Yesterday"},
            {"in", "Great. I will bring the printouts.", "Yesterday"},
            {"out", "Do you need the projector booked?", "Yesterday"},
            {"in", "Already done, room has one built in.", "Yesterday"},
            {"out", "Perfect.", "Yesterday"},
            {"in", "One more thing, can you present first?", "08:15"},
            {"out", "Sure, I only need ten minutes.", "08:30"},
            {"in", "Ten is plenty.", "08:31"},
            {"photo", "", "08:40"},
            {"in", "Here is the running order.", "08:40"},
            {"out", "Looks fine to me.", "09:02"},
            {"in", "See you there.", "09:05"},
        },
    };

    private final SimulatedTaskActivity host;
    private TaskBanner banner;
    private LinearLayout thread;
    private ScrollView threadScroll;
    private LinearLayout composerHolder;
    private EditText input;
    private LinearLayout sendButton;

    private final boolean[] didOpen = new boolean[CONTACTS];
    private final boolean[] didScroll = new boolean[CONTACTS];
    private int contact;
    private int sentHere;
    private int scrollBase;

    WhatsAppSim(SimulatedTaskActivity host) {
        this.host = host;
    }

    // ---- checklist ------------------------------------------------------

    private void recompute() {
        if ("whatsapp_chat".equals(host.phase()) && host.scrolls - scrollBase >= 2) {
            didScroll[contact] = true;
        }
        if (banner != null) {
            boolean[] states = new boolean[MESSAGES + 2];
            states[0] = didOpen[contact];
            states[1] = didScroll[contact];
            for (int m = 0; m < MESSAGES; m++) {
                states[m + 2] = sentHere > m;
            }
            banner.setStates(states);
        }
        refreshSendButton();
    }

    private void refreshSendButton() {
        if (sendButton == null || input == null) {
            return;
        }
        boolean ready = sentHere < MESSAGES
                && didScroll[contact]
                && Ui.matches(input, SCRIPT[contact][sentHere]);
        sendButton.setBackground(Ui.circle(ready ? Ui.WA_GREEN : Color.parseColor("#B7C0C4")));
        sendButton.setEnabled(ready);
        // Said in words as well as in colour: a grey circle is the only signal
        // otherwise, and a screenshot-reading agent has no way to tell a
        // disabled control from a styled one.
        sendButton.setContentDescription(ready
                ? "Send message"
                : "Send message (disabled until the message is typed)");
    }

    private TaskBanner newBanner() {
        banner = new TaskBanner(host, ACCENT, TINT);
        banner.setRound(contact + 1, CONTACTS);
        banner.setSteps(buildSteps());
        host.setProgressListener(this::recompute);
        return banner;
    }

    // ---- step 1: chat list ----------------------------------------------

    void showChatList() {
        host.setBackAction(null);   // top of this app: Back means leave
        host.setPhase("whatsapp_chats");
        host.setChrome(Ui.WA_TEAL, false);
        sendButton = null;
        input = null;

        LinearLayout screen = Ui.col(host);
        screen.setBackgroundColor(Color.WHITE);

        LinearLayout header = Ui.col(host);
        header.setBackgroundColor(Ui.WA_TEAL);

        LinearLayout titleRow = Ui.row(host);
        titleRow.setPadding(Ui.dp(host, 14), Ui.dp(host, 12), Ui.dp(host, 12), Ui.dp(host, 10));
        ImageView mark = new ImageView(host);
        mark.setImageResource(R.drawable.ic_wa_send);
        titleRow.addView(mark, Ui.size(host, 26, 26));
        titleRow.addView(Ui.text(host, "WhatsApp", 21, Color.WHITE, true),
                Ui.margins(host, Ui.weight(1f), 9, 0, 0, 0));
        titleRow.addView(Ui.icon(host, R.drawable.ic_search, 22, Color.WHITE));
        ImageView exit = Ui.icon(host, R.drawable.ic_close, 20, Color.WHITE);
        exit.setContentDescription("Close task");
        exit.setOnClickListener(view -> host.confirmExit());
        titleRow.addView(exit, Ui.margins(host, Ui.size(host, 20, 20), 20, 0, 0, 0));
        header.addView(titleRow, Ui.matchWrap());

        LinearLayout tabs = Ui.row(host);
        String[] labels = {"CHATS", "STATUS", "CALLS"};
        for (int index = 0; index < labels.length; index++) {
            LinearLayout tab = Ui.col(host);
            TextView label = Ui.text(host, labels[index], 13,
                    index == 0 ? Color.WHITE : Color.parseColor("#8FBDB6"), true);
            label.setGravity(Gravity.CENTER);
            label.setPadding(0, Ui.dp(host, 11), 0, Ui.dp(host, 10));
            label.setLetterSpacing(0.05f);
            tab.addView(label, Ui.matchWrap());
            View underline = new View(host);
            underline.setBackgroundColor(index == 0 ? Color.WHITE : Color.TRANSPARENT);
            tab.addView(underline, new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(host, 3)));
            tabs.addView(tab, Ui.weight(1f));
        }
        header.addView(tabs, Ui.matchWrap());
        screen.addView(header, Ui.matchWrap());
        screen.addView(newBanner(), TaskBanner.params(host));

        ScrollView scroll = new ScrollView(host);
        LinearLayout list = Ui.col(host);
        list.setPadding(0, Ui.dp(host, 4), 0, Ui.dp(host, 20));
        for (int index = 0; index < NAMES.length; index++) {
            list.addView(chatRow(index), Ui.matchWrap());
        }
        scroll.addView(list);
        screen.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        host.show(screen);
        recompute();
    }

    private LinearLayout chatRow(int index) {
        LinearLayout row = Ui.row(host);
        row.setPadding(Ui.dp(host, 14), Ui.dp(host, 10), Ui.dp(host, 14), Ui.dp(host, 10));

        ImageView avatar = new ImageView(host);
        avatar.setImageResource(AVATARS[index]);
        row.addView(avatar, Ui.size(host, 50, 50));

        LinearLayout details = Ui.col(host);
        details.setPadding(Ui.dp(host, 12), 0, 0, 0);

        LinearLayout topLine = Ui.row(host);
        topLine.addView(Ui.text(host, NAMES[index], 16, Color.parseColor("#111B21"), true),
                Ui.weight(1f));
        topLine.addView(Ui.text(host, TIMES[index], 11,
                UNREAD[index] > 0 ? Ui.WA_GREEN : Ui.WA_SUBTITLE));
        details.addView(topLine, Ui.matchWrap());

        LinearLayout bottomLine = Ui.row(host);
        if (UNREAD[index] == 0 && index != 2) {
            bottomLine.addView(Ui.icon(host, R.drawable.ic_tick_double, 16, 0),
                    Ui.margins(host, Ui.size(host, 16, 11), 0, 0, 4, 0));
        }
        TextView preview = Ui.text(host, PREVIEWS[index], 14, Ui.WA_SUBTITLE);
        preview.setMaxLines(1);
        preview.setEllipsize(android.text.TextUtils.TruncateAt.END);
        bottomLine.addView(preview, Ui.weight(1f));
        if (UNREAD[index] > 0) {
            TextView badge = Ui.text(host, String.valueOf(UNREAD[index]), 11, Color.WHITE, true);
            badge.setGravity(Gravity.CENTER);
            badge.setBackground(Ui.circle(Ui.WA_GREEN));
            badge.setPadding(Ui.dp(host, 6), Ui.dp(host, 2), Ui.dp(host, 6), Ui.dp(host, 2));
            bottomLine.addView(badge, Ui.margins(host, Ui.wrap(), 8, 0, 0, 0));
        }
        details.addView(bottomLine, Ui.margins(host, Ui.matchWrap(), 0, 4, 0, 0));

        row.addView(details, Ui.weight(1f));
        row.setOnClickListener(view -> {
            if (contact < CONTACTS && TARGETS[contact].equals(NAMES[index])) {
                didOpen[contact] = true;
                host.chatOpened();
                showConversation();
            }
        });
        return row;
    }

    // ---- steps 2-6: conversation ----------------------------------------

    private void showConversation() {
        host.setBackAction(this::showChatList);
        host.setPhase("whatsapp_chat");
        host.setChrome(Ui.WA_TEAL, false);
        scrollBase = host.scrolls;

        LinearLayout screen = Ui.col(host);
        screen.setBackgroundColor(Ui.WA_WALLPAPER);

        LinearLayout header = Ui.row(host);
        header.setBackgroundColor(Ui.WA_TEAL);
        header.setPadding(Ui.dp(host, 8), Ui.dp(host, 8), Ui.dp(host, 12), Ui.dp(host, 8));
        ImageView exit = Ui.icon(host, R.drawable.ic_close, 22, Color.WHITE);
        exit.setContentDescription("Close task");
        exit.setOnClickListener(view -> host.confirmExit());
        header.addView(exit);
        ImageView avatar = new ImageView(host);
        avatar.setImageResource(AVATARS[contactIndex()]);
        header.addView(avatar, Ui.margins(host, Ui.size(host, 38, 38), 6, 0, 0, 0));
        LinearLayout names = Ui.col(host);
        names.addView(Ui.text(host, TARGETS[contact], 16, Color.WHITE, true), Ui.matchWrap());
        names.addView(Ui.text(host, "online", 11, Color.parseColor("#B9D6D0")), Ui.matchWrap());
        header.addView(names, Ui.margins(host, Ui.weight(1f), 10, 0, 0, 0));
        header.addView(Ui.icon(host, R.drawable.ic_wa_video, 22, Color.WHITE));
        header.addView(Ui.icon(host, R.drawable.ic_wa_call, 20, Color.WHITE),
                Ui.margins(host, Ui.size(host, 20, 20), 18, 0, 0, 0));
        header.addView(Ui.icon(host, R.drawable.ic_wa_more, 18, Color.WHITE),
                Ui.margins(host, Ui.size(host, 18, 18), 16, 0, 0, 0));
        screen.addView(header, Ui.matchWrap());
        screen.addView(newBanner(), TaskBanner.params(host));

        threadScroll = new ScrollView(host);
        threadScroll.setBackground(wallpaper());
        thread = Ui.col(host);
        thread.setPadding(Ui.dp(host, 10), Ui.dp(host, 10), Ui.dp(host, 10), Ui.dp(host, 10));
        buildHistory();
        threadScroll.addView(thread);
        screen.addView(threadScroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        composerHolder = Ui.col(host);
        composerHolder.setBackgroundColor(Color.parseColor("#F0F0F0"));
        composerHolder.addView(buildComposer(), Ui.matchWrap());
        screen.addView(composerHolder, Ui.matchWrap());

        host.show(screen);
        threadScroll.post(() -> threadScroll.fullScroll(View.FOCUS_DOWN));
        recompute();
    }

    private void buildHistory() {
        thread.addView(dateChip("TODAY"), Ui.matchWrap());
        for (String[] line : HISTORY[contact]) {
            if ("photo".equals(line[0])) {
                addPhotoBubble(line[2]);
            } else {
                addBubble(line[1], line[2], "out".equals(line[0]), true);
            }
        }
    }

    /** Index into the avatar / name arrays for the contact being written to. */
    private int contactIndex() {
        for (int i = 0; i < NAMES.length; i++) {
            if (NAMES[i].equals(TARGETS[contact])) {
                return i;
            }
        }
        return 0;
    }

    private LinearLayout dateChip(String label) {
        LinearLayout holder = Ui.row(host);
        holder.setGravity(Gravity.CENTER);
        TextView chip = Ui.text(host, label, 11, Ui.WA_SUBTITLE, true);
        chip.setBackground(Ui.rounded(host, Color.parseColor("#E1F2FB"), 8));
        chip.setPadding(Ui.dp(host, 12), Ui.dp(host, 5), Ui.dp(host, 12), Ui.dp(host, 5));
        holder.addView(chip, Ui.margins(host, Ui.wrap(), 0, 4, 0, 10));
        return holder;
    }

    private void addBubble(String message, String time, boolean outgoing, boolean read) {
        LinearLayout holder = Ui.row(host);
        holder.setGravity(outgoing ? Gravity.END : Gravity.START);

        LinearLayout bubble = Ui.col(host);
        bubble.setBackground(bubbleBackground(outgoing));
        bubble.setPadding(Ui.dp(host, 10), Ui.dp(host, 7), Ui.dp(host, 9), Ui.dp(host, 6));

        TextView body = Ui.text(host, message, 15, Color.parseColor("#111B21"));
        body.setLineSpacing(0f, 1.2f);
        bubble.addView(body, Ui.matchWrap());

        LinearLayout meta = Ui.row(host);
        meta.setGravity(Gravity.END | Gravity.CENTER_VERTICAL);
        meta.addView(Ui.text(host, time, 10, Color.parseColor("#8696A0")));
        if (outgoing) {
            meta.addView(Ui.icon(host,
                            read ? R.drawable.ic_tick_double : R.drawable.ic_tick_double_grey,
                            15, 0),
                    Ui.margins(host, Ui.size(host, 15, 10), 4, 0, 0, 0));
        }
        bubble.addView(meta, Ui.margins(host, Ui.matchWrap(), 0, 2, 0, 0));

        LinearLayout.LayoutParams params = Ui.wrap();
        params.setMargins(
                outgoing ? Ui.dp(host, 60) : 0, Ui.dp(host, 3),
                outgoing ? 0 : Ui.dp(host, 60), Ui.dp(host, 3));
        holder.addView(bubble, params);
        thread.addView(holder, Ui.matchWrap());
    }

    private void addPhotoBubble(String time) {
        LinearLayout holder = Ui.row(host);
        holder.setGravity(Gravity.START);
        LinearLayout bubble = Ui.col(host);
        bubble.setBackground(bubbleBackground(false));
        bubble.setPadding(Ui.dp(host, 4), Ui.dp(host, 4), Ui.dp(host, 4), Ui.dp(host, 4));
        ImageView photo = new ImageView(host);
        photo.setImageResource(R.drawable.chat_photo);
        photo.setScaleType(ImageView.ScaleType.FIT_XY);
        bubble.addView(photo, Ui.size(host, 196, 130));
        TextView stamp = Ui.text(host, time, 10, Color.parseColor("#8696A0"));
        stamp.setGravity(Gravity.END);
        bubble.addView(stamp, Ui.margins(host, Ui.matchWrap(), 0, 3, 4, 1));
        LinearLayout.LayoutParams params = Ui.wrap();
        params.setMargins(0, Ui.dp(host, 3), Ui.dp(host, 60), Ui.dp(host, 3));
        holder.addView(bubble, params);
        thread.addView(holder, Ui.matchWrap());
    }

    private GradientDrawable bubbleBackground(boolean outgoing) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(outgoing ? Ui.WA_BUBBLE_OUT : Color.WHITE);
        float large = Ui.dp(host, 9);
        float small = Ui.dp(host, 1);
        // A squared-off top corner on the sender's side stands in for the tail.
        drawable.setCornerRadii(outgoing
                ? new float[] {large, large, small, small, large, large, large, large}
                : new float[] {small, small, large, large, large, large, large, large});
        return drawable;
    }

    private LinearLayout buildComposer() {
        LinearLayout row = Ui.row(host);
        row.setPadding(Ui.dp(host, 7), Ui.dp(host, 6), Ui.dp(host, 7), Ui.dp(host, 8));

        LinearLayout pill = Ui.row(host);
        pill.setBackground(Ui.rounded(host, Color.WHITE, 24));
        pill.setPadding(Ui.dp(host, 10), Ui.dp(host, 4), Ui.dp(host, 10), Ui.dp(host, 4));
        pill.addView(Ui.icon(host, R.drawable.ic_wa_emoji, 22, Color.parseColor("#8696A0")));

        // Just "Message", the way a real chat box reads.  The prompt used to be
        // "Type: <the message>", which renders as grey text sitting inside the
        // box and is indistinguishable, in a screenshot, from a message already
        // composed -- an agent read it as done and tapped a send button that was
        // disabled because the box was in fact empty.  What to send is on the
        // checklist above, which is where the rest of the task is stated.
        input = host.trackedInput("Message");
        input.setBackground(null);
        input.setTextSize(15);
        input.setTextColor(Color.parseColor("#111B21"));
        input.setHintTextColor(Color.parseColor("#8696A0"));
        input.setPadding(Ui.dp(host, 8), Ui.dp(host, 9), Ui.dp(host, 8), Ui.dp(host, 9));
        Ui.onTextChange(input, this::refreshSendButton);
        host.submitOn(input, this::sendScriptedMessage);
        pill.addView(input, Ui.weight(1f));

        pill.addView(Ui.icon(host, R.drawable.ic_wa_attach, 21, Color.parseColor("#8696A0")));
        pill.addView(Ui.icon(host, R.drawable.ic_wa_camera, 20, Color.parseColor("#8696A0")),
                Ui.margins(host, Ui.size(host, 20, 20), 14, 0, 0, 0));
        row.addView(pill, Ui.weight(1f));

        sendButton = Ui.row(host);
        sendButton.setGravity(Gravity.CENTER);
        sendButton.setBackground(Ui.circle(Color.parseColor("#B7C0C4")));
        sendButton.addView(Ui.icon(host, R.drawable.ic_wa_send, 21, Color.WHITE));
        sendButton.setContentDescription("Send message");
        sendButton.setOnClickListener(view -> sendScriptedMessage());
        row.addView(sendButton, Ui.margins(host, Ui.size(host, 46, 46), 7, 0, 0, 0));
        return row;
    }

    private void sendScriptedMessage() {
        if (sentHere >= MESSAGES || !didScroll[contact]) {
            return;
        }
        String expected = SCRIPT[contact][sentHere];
        if (!Ui.matches(input, expected)) {
            return;
        }
        if (!host.submitText(input, "whatsapp_message", expected)) {
            return;
        }
        host.sentMessages++;
        sentHere++;
        addBubble(expected, clock(), true, false);
        threadScroll.post(() -> threadScroll.fullScroll(View.FOCUS_DOWN));
        if (sentHere >= MESSAGES) {
            host.hideKeyboard(input);
            if (contact + 1 < CONTACTS) {
                contact++;
                sentHere = 0;
                showChatList();
                return;
            }
            showFinishBar();
        } else {
            input.setHint("Message");
        }
        recompute();
    }

    private String clock() {
        return String.format(java.util.Locale.US, "13:%02d", 44 + sentHere);
    }

    private void showFinishBar() {
        sendButton = null;
        composerHolder.removeAllViews();
        TextView finish = Ui.text(host, "Finish task", 15, Color.WHITE, true);
        finish.setTypeface(Typeface.DEFAULT_BOLD);
        finish.setGravity(Gravity.CENTER);
        finish.setBackground(Ui.rounded(host, Color.parseColor("#1F2933"), 22));
        finish.setPadding(Ui.dp(host, 16), Ui.dp(host, 13), Ui.dp(host, 16), Ui.dp(host, 13));
        finish.setOnClickListener(view -> host.finishIfValid());
        composerHolder.addView(finish, Ui.margins(host, Ui.matchWrap(), 12, 10, 12, 12));
    }

    /** WhatsApp-style tiled doodle wallpaper built from a vector tile. */
    private Drawable wallpaper() {
        int size = Ui.dp(host, 120);
        Drawable tile = host.getDrawable(R.drawable.wa_doodle_tile);
        if (tile == null || size <= 0) {
            return new ColorDrawable(Ui.WA_WALLPAPER);
        }
        Bitmap bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        tile.setBounds(0, 0, size, size);
        tile.draw(canvas);
        BitmapDrawable tiled = new BitmapDrawable(host.getResources(), bitmap);
        tiled.setTileModeXY(Shader.TileMode.REPEAT, Shader.TileMode.REPEAT);
        return new LayerDrawable(new Drawable[] {
            new ColorDrawable(Ui.WA_WALLPAPER), tiled,
        });
    }
}
