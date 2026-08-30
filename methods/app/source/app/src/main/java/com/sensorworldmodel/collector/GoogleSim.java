package com.sensorworldmodel.collector;

import android.graphics.Color;
import android.graphics.Typeface;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

/**
 * A local, offline search-and-read simulation styled after Google on mobile.
 * Two fixed rounds keep the typed text and the reading load identical for every
 * participant.
 */
final class GoogleSim {
    static final int ROUNDS = 3;

    private static final int ACCENT = Color.parseColor("#1A73E8");
    private static final int TINT = Color.parseColor("#E8F0FE");

    /** The exact phrases the protocol asks every participant to type. */
    private static final String[] QUERIES = {
        "weather tokyo", "train to kyoto", "osaka food guide",
    };

    private static final String[] TARGET_TITLES = {
        "Tokyo Weather - 7 Day Forecast",
        "Tokyo to Kyoto by Train - Times and Fares",
        "Osaka Street Food - What to Eat and Where",
    };
    private static final int[] HEROES = {
        R.drawable.fig_chart, R.drawable.fig_chart, R.drawable.fig_chart,
    };
    private static final String[] DOMAINS = {
        "weatherjapan.example", "railguide.example", "osakaeats.example",
    };
    private static final String[] ARTICLE_TITLES = {
        "Tokyo weather: a seven day outlook",
        "Riding the Tokyo to Kyoto line",
        "Eating your way through Osaka",
    };
    private static final String[] BYLINES = {
        "Weather desk · Updated 2 hours ago",
        "Travel desk · Updated yesterday",
        "Food desk · Updated 3 days ago",
    };

    private static final String[][] RESULT_TITLES = {
        {
            "Tokyo, Japan - Current conditions",
            "Tokyo Weather - 7 Day Forecast",
            "Climate and average weather in Tokyo",
            "When is the best time to visit Tokyo?",
            "Tokyo rainfall and typhoon season explained",
            "Hourly forecast for Shinjuku and Shibuya",
            "Tokyo air quality and pollen index today",
        },
        {
            "How to travel from Tokyo to Kyoto",
            "Tokyo to Kyoto by Train - Times and Fares",
            "Rail pass options for the Tokyo-Kyoto route",
            "Is the night bus cheaper than the train?",
            "Kyoto station guide for first time visitors",
            "Luggage forwarding between Tokyo and Kyoto hotels",
            "Best seats for a Mount Fuji view on the way",
        },
        {
            "Osaka food guide for a first visit",
            "Osaka Street Food - What to Eat and Where",
            "Dotonbori at night: what is worth queueing for",
            "Where do locals eat in Namba?",
            "Osaka vs Tokyo: which city eats better?",
            "Vegetarian options in Osaka street food",
            "How much does a food tour cost?",
        },
    };
    private static final String[][] RESULT_SITES = {
        {"Japan Weather", "Weather Japan", "Climate Data", "Travel Notes", "Storm Watch",
         "City Forecast", "Air Index"},
        {"Japan Travel", "Rail Guide", "Pass Compare", "Budget Trips", "Station Maps",
         "Baggage Go", "Window Seat"},
        {"Osaka Eats", "Street Food JP", "Night Markets", "Local Bites", "City Compare",
         "Green Table", "Tour Costs"},
    };
    private static final String[][] RESULT_SNIPPETS = {
        {
            "Current conditions for central Tokyo with hourly temperature, humidity "
                    + "and wind readings updated every ten minutes.",
            "A seven day outlook for Tokyo including daily highs and lows, chance of "
                    + "rain and a short summary for each morning and evening.",
            "Monthly averages going back thirty years, with charts for temperature, "
                    + "rainfall and the number of sunshine hours.",
            "Spring and autumn are usually the mildest windows. This guide compares "
                    + "each season for crowds, cost and daylight hours.",
            "The rainy season normally runs from early June to mid July, followed by "
                    + "the warmest and most humid weeks of the year.",
            "Hour by hour readings for the central wards, refreshed every ten "
                    + "minutes from the station on the west side of the park.",
            "Daily air quality, pollen and UV index for the metropolitan area, "
                    + "with a short note on what each band means.",
        },
        {
            "An overview of every route between the two cities, comparing the fastest "
                    + "train, the slower local service and the overnight bus.",
            "Departure times, journey duration and standard fares for the Tokyo to "
                    + "Kyoto service, including reserved and non reserved seats.",
            "Whether a rail pass pays for itself depends on how many long distance "
                    + "trips you plan. This page works through the break even point.",
            "A cost comparison covering buses, trains and budget flights, with notes "
                    + "on comfort, luggage limits and arrival times.",
            "Platform layouts, luggage lockers and the quickest transfers between the "
                    + "main lines at Kyoto station.",
            "Same day luggage forwarding between hotels costs about the price of "
                    + "a bento and saves carrying bags through the gates.",
            "The mountain appears on the right hand side about forty minutes out "
                    + "of the city, weather permitting.",
        },
        {
            "A short introduction to the dishes the city is known for and the "
                    + "districts where each one is easiest to find.",
            "Takoyaki, okonomiyaki and kushikatsu explained, with the stalls that "
                    + "locals queue at and the prices to expect.",
            "The main strip is busiest after eight in the evening; this guide "
                    + "lists which stands move fastest.",
            "A few streets back from the main run the prices drop and the queues "
                    + "shorten noticeably.",
            "Both cities do it well, but the portions and the price per plate "
                    + "differ more than most guides admit.",
            "Several stalls will make a version without bonito or pork if you ask "
                    + "before they start cooking.",
            "Group walking tours run about four thousand yen including tastings, "
                    + "or you can walk the same route alone for less.",
        },
    };
    private static final int[][] FAVICON_COLORS = {
        {0xFF1A73E8, 0xFF34A853, 0xFFEA4335, 0xFFFBBC05, 0xFF7B1FA2, 0xFF00838F, 0xFF6D4C41},
        {0xFFEA4335, 0xFF1A73E8, 0xFF00897B, 0xFFF57C00, 0xFF5E35B1, 0xFF2E7D32, 0xFFAD1457},
        {0xFFF57C00, 0xFFC62828, 0xFF283593, 0xFF00695C, 0xFF4527A0, 0xFF2E7D32, 0xFF37474F},
    };
    private static final String[][] PEOPLE_ASK = {
        {
            "What is the average temperature in Tokyo in August?",
            "Does it rain a lot in Tokyo?",
            "How humid does Tokyo get?",
            "What should I pack for August in Tokyo?",
        },
        {
            "How long is the train from Tokyo to Kyoto?",
            "Do I need to reserve a seat?",
            "Which side has the mountain view?",
            "Can I take a large suitcase on board?",
        },
        {
            "What is Osaka most famous for eating?",
            "Is street food in Osaka cash only?",
            "When do the stalls open?",
        },
    };

    /** Only the current round is listed, so the card stays short. */
    private String[] buildSteps() {
        int r = uiRound() - 1;
        boolean last = r + 1 >= ROUNDS;
        return new String[] {
            "Search for \"" + QUERIES[r] + "\"",
            "Open \"" + TARGET_TITLES[r] + "\"",
            last ? "Pinch the chart, scroll to the end, then finish"
                 : "Scroll to the end of the article",
        };
    }

    /** The round the checklist should describe, from the live phase. */
    private int uiRound() {
        String phase = host.phase();
        if (phase != null) {
            int cut = phase.lastIndexOf('_');
            if (cut > 0) {
                try {
                    return Math.max(1, Math.min(ROUNDS,
                            Integer.parseInt(phase.substring(cut + 1))));
                } catch (NumberFormatException ignored) {
                    return Math.max(1, Math.min(ROUNDS, host.searchRounds + 1));
                }
            }
        }
        return Math.max(1, Math.min(ROUNDS, host.searchRounds + 1));
    }

    private final SimulatedTaskActivity host;
    private TaskBanner banner;

    private final boolean[] didSearch = new boolean[ROUNDS];
    private final boolean[] didOpen = new boolean[ROUNDS];
    private final boolean[] didArticleScroll = new boolean[ROUNDS];
    private boolean didPinch;
    private int scrollBase;
    private TextView advance;

    GoogleSim(SimulatedTaskActivity host) {
        this.host = host;
    }

    // ---- checklist ------------------------------------------------------

    private void recompute() {
        String phase = host.phase();
        int delta = host.scrolls - scrollBase;
        for (int r = 0; r < ROUNDS; r++) {
            if (("google_article_" + (r + 1)).equals(phase) && delta >= 2) {
                didArticleScroll[r] = true;
            }
        }
        if (host.pinches >= 1) {
            didPinch = true;
        }
        if (banner != null) {
            int r = uiRound() - 1;
            boolean last = r + 1 >= ROUNDS;
            banner.setStates(new boolean[] {
                didSearch[r],
                didOpen[r],
                last ? (didPinch && didArticleScroll[r]) : didArticleScroll[r],
            });
        }
        if (advance != null) {
            // Only the final round needs the pinch, so nobody can get stuck part
            // way through; every article carries the same zoomable figure.
            int r = currentArticleRound();
            boolean last = r >= ROUNDS;
            boolean ready = last
                    ? didPinch && didArticleScroll[ROUNDS - 1]
                    : didArticleScroll[Math.max(0, r - 1)];
            advance.setEnabled(ready);
            advance.setBackground(Ui.rounded(host,
                    ready ? (last ? Color.parseColor("#1F2933") : ACCENT)
                          : Color.parseColor("#E4E6E9"), 22));
            advance.setTextColor(ready ? Color.WHITE : Color.parseColor("#9AA0A6"));
        }
    }

    private int currentArticleRound() {
        String phase = host.phase();
        if (phase != null && phase.startsWith("google_article_")) {
            try {
                return Integer.parseInt(phase.substring("google_article_".length()));
            } catch (NumberFormatException ignored) {
                return 1;
            }
        }
        return 1;
    }

    private TaskBanner newBanner() {
        banner = new TaskBanner(host, ACCENT, TINT);
        banner.setRound(uiRound(), ROUNDS);
        banner.setSteps(buildSteps());
        host.setProgressListener(this::recompute);
        return banner;
    }

    private int round() {
        return Math.min(host.searchRounds + 1, ROUNDS);
    }

    // ---- home -----------------------------------------------------------

    void showHome() {
        host.setBackAction(null);   // top of this app: Back means leave
        final int round = round();
        host.setPhase("google_home_" + round);
        host.setChrome(Color.WHITE, true);
        advance = null;

        LinearLayout screen = Ui.col(host);
        screen.setBackgroundColor(Color.WHITE);
        screen.addView(newBanner(), TaskBanner.params(host));

        LinearLayout page = Ui.col(host);
        page.setPadding(Ui.dp(host, 22), Ui.dp(host, 6), Ui.dp(host, 22), Ui.dp(host, 14));

        LinearLayout topRow = Ui.row(host);
        ImageView exit = Ui.icon(host, R.drawable.ic_close, 20, Color.parseColor("#5F6368"));
        exit.setContentDescription("Close task");
        exit.setOnClickListener(view -> host.confirmExit());
        topRow.addView(exit);
        topRow.addView(Ui.flexSpacer(host));
        topRow.addView(Ui.icon(host, R.drawable.ic_apps_grid, 22, 0));
        ImageView account = new ImageView(host);
        account.setImageDrawable(Ui.letterBadge("S", Color.parseColor("#5F6368"), Color.WHITE));
        topRow.addView(account, Ui.margins(host, Ui.size(host, 28, 28), 14, 0, 0, 0));
        page.addView(topRow, Ui.matchWrap());

        page.addView(Ui.spacer(host, 26), Ui.matchWrap());

        ImageView wordmark = new ImageView(host);
        wordmark.setImageResource(R.drawable.ic_search);
        wordmark.setAdjustViewBounds(true);
        wordmark.setScaleType(ImageView.ScaleType.FIT_CENTER);
        page.addView(wordmark, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(host, 52)));

        page.addView(Ui.spacer(host, 20), Ui.matchWrap());

        EditText input = host.trackedInput("Search or type URL");
        input.setBackground(null);
        input.setTextSize(16);
        input.setTextColor(Ui.G_URL);
        input.setHintTextColor(Color.parseColor("#9AA0A6"));
        input.setPadding(Ui.dp(host, 6), Ui.dp(host, 10), Ui.dp(host, 6), Ui.dp(host, 10));

        LinearLayout pill = Ui.row(host);
        pill.setBackground(Ui.rounded(host, Color.WHITE, 26, Ui.G_BORDER, 1));
        pill.setPadding(Ui.dp(host, 14), 0, Ui.dp(host, 12), 0);
        pill.addView(Ui.icon(host, R.drawable.ic_search, 20, Color.parseColor("#9AA0A6")));
        pill.addView(input, Ui.margins(host, Ui.weight(1f), 8, 0, 6, 0));
        pill.addView(Ui.icon(host, R.drawable.ic_mic_color, 21, 0));
        pill.addView(Ui.icon(host, R.drawable.ic_lens, 21, 0),
                Ui.margins(host, Ui.size(host, 21, 21), 14, 0, 0, 0));
        page.addView(pill, Ui.matchWrap());

        page.addView(Ui.spacer(host, 18), Ui.matchWrap());

        LinearLayout buttons = Ui.row(host);
        buttons.setGravity(Gravity.CENTER);
        Runnable submit = () -> {
            if (!Ui.matches(input, QUERIES[round - 1])) {
                return;
            }
            if (host.submitText(input, "google_query", QUERIES[round - 1])) {
                didSearch[round - 1] = true;
                host.hideKeyboard(input);
                host.searchRounds++;
                showResults();
            }
        };
        TextView search = greyButton("Google Search", view -> submit.run());
        host.submitOn(input, submit);
        // Unlocks only once the prompted phrase is typed exactly.
        Runnable gate = () -> {
            boolean ready = Ui.matches(input, QUERIES[round - 1]);
            search.setEnabled(ready);
            search.setBackground(Ui.rounded(host,
                    ready ? ACCENT : Ui.G_CHIP, 4, ready ? ACCENT : Ui.G_CHIP, 1));
            search.setTextColor(ready ? Color.WHITE : Color.parseColor("#9AA0A6"));
        };
        Ui.onTextChange(input, gate);
        gate.run();
        buttons.addView(search);
        buttons.addView(greyButton("I'm Feeling Lucky", null),
                Ui.margins(host, Ui.wrap(), 10, 0, 0, 0));
        page.addView(buttons, Ui.matchWrap());

        ScrollView scroll = new ScrollView(host);
        scroll.addView(page);
        screen.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        host.show(screen);
        host.focus(input);
        recompute();
    }

    private TextView greyButton(String label, View.OnClickListener click) {
        TextView button = Ui.text(host, label, 14, Color.parseColor("#3C4043"));
        button.setBackground(Ui.rounded(host, Ui.G_CHIP, 4, Ui.G_CHIP, 1));
        button.setPadding(Ui.dp(host, 16), Ui.dp(host, 11), Ui.dp(host, 16), Ui.dp(host, 11));
        button.setGravity(Gravity.CENTER);
        if (click != null) {
            button.setOnClickListener(click);
        }
        return button;
    }

    // ---- results --------------------------------------------------------

    private void showResults() {
        host.setBackAction(this::showHome);
        final int round = host.searchRounds;
        host.setPhase("google_results_" + round);
        host.setChrome(Color.WHITE, true);
        scrollBase = host.scrolls;
        advance = null;

        LinearLayout screen = Ui.col(host);
        screen.setBackgroundColor(Color.WHITE);
        screen.addView(resultsHeader(QUERIES[round - 1]), Ui.matchWrap());
        screen.addView(newBanner(), TaskBanner.params(host));

        ScrollView scroll = new ScrollView(host);
        LinearLayout page = Ui.col(host);
        page.setPadding(Ui.dp(host, 16), Ui.dp(host, 6), Ui.dp(host, 16), Ui.dp(host, 28));

        page.addView(Ui.text(host, "About 84,300,000 results (0.42 seconds)", 12,
                Color.parseColor("#70757A")), Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 14));
        page.addView(round == 1 ? weatherCard() : routeCard(),
                Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 18));

        for (int index = 0; index < RESULT_TITLES[round - 1].length; index++) {
            page.addView(resultBlock(round, index),
                    Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 22));
        }
        page.addView(peopleAlsoAsk(PEOPLE_ASK[round - 1]), Ui.matchWrap());

        scroll.addView(page);
        screen.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        host.show(screen);
        recompute();
    }

    private LinearLayout resultsHeader(String query) {
        LinearLayout header = Ui.col(host);
        header.setBackgroundColor(Color.WHITE);
        header.setPadding(Ui.dp(host, 10), Ui.dp(host, 10), Ui.dp(host, 14), 0);

        LinearLayout row = Ui.row(host);
        ImageView exit = Ui.icon(host, R.drawable.ic_close, 20, Color.parseColor("#5F6368"));
        exit.setContentDescription("Close task");
        exit.setOnClickListener(view -> host.confirmExit());
        row.addView(exit);

        LinearLayout pill = Ui.row(host);
        pill.setBackground(Ui.rounded(host, Ui.G_CHIP, 24));
        pill.setPadding(Ui.dp(host, 14), Ui.dp(host, 10), Ui.dp(host, 12), Ui.dp(host, 10));
        pill.addView(Ui.text(host, query, 15, Ui.G_URL), Ui.weight(1f));
        pill.addView(Ui.icon(host, R.drawable.ic_mic_color, 18, 0));
        pill.addView(Ui.icon(host, R.drawable.ic_lens, 18, 0),
                Ui.margins(host, Ui.size(host, 18, 18), 12, 0, 0, 0));
        row.addView(pill, Ui.margins(host, Ui.weight(1f), 8, 0, 0, 0));
        header.addView(row, Ui.matchWrap());

        LinearLayout tabs = Ui.row(host);
        String[] labels = {"All", "Images", "News", "Videos", "Maps"};
        for (int index = 0; index < labels.length; index++) {
            LinearLayout tab = Ui.col(host);
            TextView label = Ui.text(host, labels[index], 14,
                    index == 0 ? ACCENT : Color.parseColor("#5F6368"), index == 0);
            label.setPadding(0, Ui.dp(host, 12), 0, Ui.dp(host, 9));
            label.setGravity(Gravity.CENTER);
            tab.addView(label, Ui.matchWrap());
            View underline = new View(host);
            underline.setBackgroundColor(index == 0 ? ACCENT : Color.TRANSPARENT);
            tab.addView(underline, new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(host, 3)));
            tabs.addView(tab, Ui.margins(host, Ui.wrap(), 0, 0, 22, 0));
        }
        HorizontalScrollView tabScroll = new HorizontalScrollView(host);
        tabScroll.setHorizontalScrollBarEnabled(false);
        tabScroll.addView(tabs);
        header.addView(tabScroll, Ui.margins(host, Ui.matchWrap(), 28, 0, 0, 0));
        header.addView(Ui.divider(host, Ui.G_BORDER), Ui.matchWrap());
        return header;
    }

    private LinearLayout weatherCard() {
        LinearLayout card = Ui.col(host);
        card.setBackground(Ui.rounded(host, Color.WHITE, 8, Ui.G_BORDER, 1));
        card.setPadding(Ui.dp(host, 16), Ui.dp(host, 14), Ui.dp(host, 16), Ui.dp(host, 14));

        LinearLayout row = Ui.row(host);
        LinearLayout left = Ui.col(host);
        left.addView(Ui.text(host, "Tokyo, Japan", 15, Ui.G_URL, true), Ui.matchWrap());
        left.addView(Ui.text(host, "Wednesday 14:00 · Partly cloudy", 12, Ui.G_SNIPPET),
                Ui.margins(host, Ui.matchWrap(), 0, 3, 0, 8));
        left.addView(Ui.text(host, "29°C", 34, Ui.G_URL, true), Ui.matchWrap());
        left.addView(Ui.text(host, "Precipitation 10% · Humidity 64%", 12, Ui.G_SNIPPET),
                Ui.margins(host, Ui.matchWrap(), 0, 5, 0, 0));
        row.addView(left, Ui.weight(1f));
        row.addView(Ui.image(host, R.drawable.ic_weather, 76, 76));
        card.addView(row, Ui.matchWrap());

        LinearLayout forecast = Ui.row(host);
        String[] days = {"Thu", "Fri", "Sat", "Sun"};
        String[] highs = {"31°", "30°", "27°", "28°"};
        String[] lows = {"24°", "23°", "22°", "23°"};
        for (int index = 0; index < days.length; index++) {
            LinearLayout day = Ui.col(host);
            day.setGravity(Gravity.CENTER_HORIZONTAL);
            day.addView(Ui.text(host, days[index], 12, Ui.G_SNIPPET), Ui.wrap());
            day.addView(Ui.image(host, R.drawable.ic_weather, 30, 30));
            LinearLayout temps = Ui.row(host);
            temps.addView(Ui.text(host, highs[index], 12, Ui.G_URL, true));
            temps.addView(Ui.text(host, lows[index], 12, Color.parseColor("#9AA0A6")),
                    Ui.margins(host, Ui.wrap(), 4, 0, 0, 0));
            day.addView(temps, Ui.wrap());
            forecast.addView(day, Ui.weight(1f));
        }
        card.addView(forecast, Ui.margins(host, Ui.matchWrap(), 0, 12, 0, 0));
        return card;
    }

    private LinearLayout routeCard() {
        LinearLayout card = Ui.col(host);
        card.setBackground(Ui.rounded(host, Color.WHITE, 8, Ui.G_BORDER, 1));
        card.setPadding(Ui.dp(host, 16), Ui.dp(host, 14), Ui.dp(host, 16), Ui.dp(host, 14));
        card.addView(Ui.text(host, "Tokyo → Kyoto", 15, Ui.G_URL, true), Ui.matchWrap());
        card.addView(Ui.text(host, "Fastest train · 2 h 15 min · from ¥13,320", 13,
                Ui.G_SNIPPET), Ui.margins(host, Ui.matchWrap(), 0, 4, 0, 10));
        ImageView photo = new ImageView(host);
        photo.setImageResource(R.drawable.fig_chart);
        photo.setScaleType(ImageView.ScaleType.CENTER_CROP);
        card.addView(photo, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(host, 120)));
        LinearLayout legs = Ui.row(host);
        for (String label : new String[] {"Train 2 h 15", "Bus 7 h 30", "Flight 1 h 10"}) {
            TextView chip = Ui.text(host, label, 12, Ui.G_SNIPPET);
            chip.setBackground(Ui.rounded(host, Ui.G_CHIP, 14));
            chip.setPadding(Ui.dp(host, 10), Ui.dp(host, 6), Ui.dp(host, 10), Ui.dp(host, 6));
            legs.addView(chip, Ui.margins(host, Ui.wrap(), 0, 0, 8, 0));
        }
        card.addView(legs, Ui.margins(host, Ui.matchWrap(), 0, 10, 0, 0));
        return card;
    }

    private LinearLayout resultBlock(int round, int index) {
        LinearLayout block = Ui.col(host);
        String site = RESULT_SITES[round - 1][index];
        String title = RESULT_TITLES[round - 1][index];

        LinearLayout siteRow = Ui.row(host);
        ImageView favicon = new ImageView(host);
        favicon.setImageDrawable(Ui.letterBadge(site.substring(0, 1),
                FAVICON_COLORS[round - 1][index], Color.WHITE));
        siteRow.addView(favicon, Ui.size(host, 26, 26));
        LinearLayout siteText = Ui.col(host);
        siteText.addView(Ui.text(host, site, 13, Ui.G_URL), Ui.matchWrap());
        siteText.addView(Ui.text(host,
                site.toLowerCase(java.util.Locale.US).replace(" ", "") + ".example › guide",
                11, Color.parseColor("#5F6368")), Ui.matchWrap());
        siteRow.addView(siteText, Ui.margins(host, Ui.weight(1f), 9, 0, 0, 0));
        block.addView(siteRow, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 7));

        TextView titleView = Ui.text(host, title, 19, Ui.G_LINK);
        titleView.setLineSpacing(0f, 1.1f);
        block.addView(titleView, Ui.matchWrap());

        TextView snippet = Ui.text(host, RESULT_SNIPPETS[round - 1][index], 14, Ui.G_SNIPPET);
        snippet.setLineSpacing(0f, 1.25f);
        block.addView(snippet, Ui.margins(host, Ui.matchWrap(), 0, 5, 0, 0));

        block.setOnClickListener(view -> {
            if (title.equals(TARGET_TITLES[round - 1])) {
                host.openedResults++;
                didOpen[round - 1] = true;
                showArticle();
            }
        });
        return block;
    }

    private LinearLayout peopleAlsoAsk(String[] questions) {
        LinearLayout card = Ui.col(host);
        card.setBackground(Ui.rounded(host, Color.WHITE, 8, Ui.G_BORDER, 1));
        card.setPadding(Ui.dp(host, 16), Ui.dp(host, 14), Ui.dp(host, 16), Ui.dp(host, 6));
        card.addView(Ui.text(host, "People also ask", 17, Ui.G_URL, true),
                Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 6));
        for (String question : questions) {
            LinearLayout row = Ui.row(host);
            row.setPadding(0, Ui.dp(host, 12), 0, Ui.dp(host, 12));
            row.addView(Ui.text(host, question, 14, Ui.G_URL), Ui.weight(1f));
            row.addView(Ui.icon(host, R.drawable.ic_chevron_down, 20,
                    Color.parseColor("#5F6368")));
            card.addView(row, Ui.matchWrap());
            card.addView(Ui.divider(host, Ui.G_BORDER), Ui.matchWrap());
        }
        return card;
    }

    // ---- article --------------------------------------------------------

    private void showArticle() {
        host.setBackAction(this::showResults);
        final int round = host.searchRounds;
        host.setPhase("google_article_" + round);
        host.setChrome(Color.parseColor("#F1F3F4"), true);
        scrollBase = host.scrolls;

        LinearLayout screen = Ui.col(host);
        screen.setBackgroundColor(Color.WHITE);
        screen.addView(browserBar(DOMAINS[round - 1]), Ui.matchWrap());
        screen.addView(newBanner(), TaskBanner.params(host));

        ScrollView scroll = new ScrollView(host);
        LinearLayout page = Ui.col(host);
        page.setPadding(Ui.dp(host, 18), Ui.dp(host, 6), Ui.dp(host, 18), Ui.dp(host, 30));

        ImageView hero = new ImageView(host);
        hero.setImageResource(HEROES[round - 1]);
        hero.setScaleType(ImageView.ScaleType.CENTER_CROP);
        page.addView(hero, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(host, 168)));

        TextView title = Ui.text(host, ARTICLE_TITLES[round - 1], 25, Ui.G_URL, true);
        title.setLineSpacing(0f, 1.15f);
        page.addView(title, Ui.margins(host, Ui.matchWrap(), 0, 14, 0, 0));
        page.addView(Ui.text(host, BYLINES[round - 1], 12, Color.parseColor("#70757A")),
                Ui.margins(host, Ui.matchWrap(), 0, 7, 0, 16));

        String[] paragraphs = round == 1 ? new String[] {
            "Central Tokyo stays warm and humid through the middle of the week, with "
                    + "afternoon highs close to thirty degrees and light winds from the "
                    + "south east.",
            "Cloud cover builds after midday on Thursday and Friday. Showers are brief "
                    + "and mostly clear before the evening commute.",
            "The weekend turns a little cooler as a weak front passes over the Kanto "
                    + "plain, bringing the overnight low back down to about twenty two "
                    + "degrees.",
        } : new String[] {
            "The fastest service links the two cities in a little over two hours, "
                    + "running several times an hour for most of the day.",
            "Reserved seats can be booked from a month ahead. Non reserved cars sit at "
                    + "the front of the train and are usually easiest to find midweek.",
            "Seats on the right hand side face the mountain on a clear day, which is "
                    + "why those rows are the first to sell out in the morning.",
        };
        for (String paragraph : paragraphs) {
            TextView body = Ui.text(host, paragraph, 16, Color.parseColor("#3C4043"));
            body.setLineSpacing(0f, 1.45f);
            page.addView(body, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 16));
        }

        page.addView(Ui.text(host,
                        round == 1 ? "Seven day temperature trend" : "Journey time by service",
                        14, Ui.G_URL, true),
                Ui.margins(host, Ui.matchWrap(), 0, 4, 0, 8));
        ZoomPanel chart = new ZoomPanel(host);
        chart.setImage(R.drawable.fig_chart);
        page.addView(chart, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(host, 190)));
        page.addView(Ui.text(host,
                        round == 1
                                ? "Figure 1. Daily highs in blue, running average in red."
                                : "Figure 1. Journey time by service type.",
                        12, Color.parseColor("#70757A")),
                Ui.margins(host, Ui.matchWrap(), 0, 7, 0, 18));

        String[] tail = round == 1 ? new String[] {
            "Sunrise is just before five in the morning and the strongest sunshine "
                    + "arrives between eleven and two, so shade is worth planning for.",
            "Readings on this page come from the city centre station and are refreshed "
                    + "every ten minutes during daylight hours.",
            "Coastal districts stay two to three degrees cooler than the inland wards "
                    + "for most of the afternoon.",
        } : new String[] {
            "Luggage larger than the standard allowance needs an oversize reservation, "
                    + "which is free but has to be made before departure.",
            "Fares on this page are standard class walk up prices and do not include "
                    + "seasonal surcharges.",
            "Trains depart from the central concourse; allow ten minutes to clear the "
                    + "ticket gates at peak times.",
        };
        for (String paragraph : tail) {
            TextView body = Ui.text(host, paragraph, 16, Color.parseColor("#3C4043"));
            body.setLineSpacing(0f, 1.45f);
            page.addView(body, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 16));
        }

        boolean last = round >= ROUNDS;
        advance = Ui.text(host, last ? "Finish task" : "Next search", 15, Color.WHITE, true);
        advance.setTypeface(Typeface.DEFAULT_BOLD);
        advance.setGravity(Gravity.CENTER);
        advance.setPadding(Ui.dp(host, 16), Ui.dp(host, 13), Ui.dp(host, 16), Ui.dp(host, 13));
        advance.setOnClickListener(view -> {
            boolean ready = last
                    ? didPinch && didArticleScroll[ROUNDS - 1]
                    : didArticleScroll[round - 1];
            if (!ready) {
                return;
            }
            if (last) {
                host.finishIfValid();
            } else {
                showHome();
            }
        });
        page.addView(advance, Ui.margins(host, Ui.matchWrap(), 0, 10, 0, 0));
        page.addView(Ui.text(host,
                        "The button unlocks once the steps above are ticked off.",
                        12, Color.parseColor("#70757A")),
                Ui.margins(host, Ui.matchWrap(), 2, 7, 0, 0));

        scroll.addView(page);
        screen.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        host.show(screen);
        recompute();
    }

    private LinearLayout browserBar(String domain) {
        LinearLayout bar = Ui.col(host);
        bar.setBackgroundColor(Color.parseColor("#F1F3F4"));
        bar.setPadding(Ui.dp(host, 12), Ui.dp(host, 9), Ui.dp(host, 12), Ui.dp(host, 9));
        LinearLayout row = Ui.row(host);
        ImageView back = Ui.icon(host, R.drawable.ic_arrow_back, 20,
                Color.parseColor("#5F6368"));
        back.setContentDescription("Back to the results");
        back.setOnClickListener(view -> showResults());
        row.addView(back);
        ImageView exit = Ui.icon(host, R.drawable.ic_close, 18, Color.parseColor("#5F6368"));
        exit.setContentDescription("Close task");
        exit.setOnClickListener(view -> host.confirmExit());
        row.addView(exit, Ui.margins(host, Ui.size(host, 18, 18), 12, 0, 0, 0));
        row.addView(Ui.icon(host, R.drawable.ic_lock, 14, Color.parseColor("#5F6368")),
                Ui.margins(host, Ui.size(host, 14, 14), 10, 0, 0, 0));
        row.addView(Ui.text(host, domain, 13, Color.parseColor("#3C4043")),
                Ui.margins(host, Ui.weight(1f), 7, 0, 0, 0));
        row.addView(Ui.icon(host, R.drawable.ic_menu, 18, Color.parseColor("#5F6368")));
        bar.addView(row, Ui.matchWrap());
        return bar;
    }
}
