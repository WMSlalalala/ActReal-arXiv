package com.sensorworldmodel.collector;

import android.graphics.Color;
import android.graphics.Typeface;
import android.text.TextUtils;
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
 * A local, offline shopping simulation styled after the Amazon mobile app. It
 * never reaches the network, never takes payment and never stores typed text.
 *
 * <p>The task runs three shopping rounds. Each round searches a fixed phrase,
 * opens a fixed result and performs one gesture, so the run lasts several
 * minutes while every participant does exactly the same thing.
 */
final class AmazonSim {
    static final int ROUNDS = 3;

    private static final int ACCENT = Color.parseColor("#B45309");
    private static final int TINT = Color.parseColor("#FEF4E4");

    /** The exact phrase typed in each round. */
    private static final String[] QUERIES = {"wireless mouse", "usb c hub", "laptop stand"};
    /** The 1-based result opened in each round. */
    private static final int[] TARGETS = {3, 2, 1};

    private static final int[][] PHOTOS = {
        {R.drawable.cat_electronics, R.drawable.cat_electronics, R.drawable.cat_electronics,
         R.drawable.cat_electronics, R.drawable.cat_electronics, R.drawable.cat_electronics},
        {R.drawable.cat_home, R.drawable.cat_home, R.drawable.cat_home,
         R.drawable.cat_home, R.drawable.cat_home, R.drawable.cat_home},
        {R.drawable.cat_home, R.drawable.cat_home, R.drawable.cat_home,
         R.drawable.cat_home, R.drawable.cat_home, R.drawable.cat_home},
    };
    private static final String[][] TITLES = {
        {"Wireless Mouse, 2.4G Silent Ergonomic Computer Mouse",
         "Basics Wireless Computer Mouse with Nano Receiver",
         "Ergonomic Vertical Wireless Mouse, Adjustable DPI",
         "Wireless Optical Mouse for Laptop, 3 DPI Levels",
         "Slim Bluetooth Mouse, Rechargeable, Space Grey",
         "Lightweight Gaming Mouse, 25K Sensor"},
        {"USB C Hub 7-in-1 Multiport Adapter, 4K HDMI",
         "USB C to USB Adapter, OTG Connector 2 Pack",
         "USB C Docking Station, Dual Monitor, 100W PD",
         "USB 2.0 Hub 4 Port, Bus Powered, Compact",
         "SD and microSD Card Reader, USB C and USB A",
         "USB C to Ethernet Adapter, Gigabit RJ45"},
        {"Laptop Stand for Desk, Adjustable Aluminium Riser",
         "Laptop Cooling Stand with Two Silent Fans",
         "Minimalist Wooden Laptop Riser, Walnut Finish",
         "Foldable Tablet and Phone Stand, Non Slip",
         "Docking Station Riser for 13 to 17 inch Laptops",
         "Portable Laptop Stand, Folds Flat for Travel"},
    };
    private static final double[][] RATINGS = {
        {4.5, 4.0, 4.5, 4.0, 4.5, 5.0},
        {4.5, 4.0, 5.0, 3.5, 4.0, 4.5},
        {4.5, 4.0, 4.5, 3.5, 4.0, 4.5},
    };
    private static final int[][] REVIEWS = {
        {2143, 8762, 1284, 512, 3067, 1896},
        {5310, 2204, 876, 1450, 690, 3312},
        {4127, 1839, 623, 2510, 418, 1176},
    };
    private static final String[][] PRICES = {
        {"$12.99", "$9.49", "$25.99", "$11.99", "$21.99", "$64.99"},
        {"$32.99", "$8.99", "$79.99", "$10.49", "$14.99", "$18.99"},
        {"$27.99", "$22.49", "$44.99", "$12.99", "$36.99", "$19.99"},
    };
    private static final String[] STORES = {
        "ErgoLine Store", "PortHub Store", "DeskCraft Store",
    };
    /** Each round asks for a different gesture, so the load is spread out. */
    private static final String[] GESTURE_STEP = {
        "Swipe the photo strip sideways",
        "Scroll down to the customer reviews",
        "Pinch the big photo to zoom in",
    };

    private static final String[][] BULLETS = {
        {"Vertical grip keeps the wrist in a natural handshake position.",
         "Adjustable 800 / 1200 / 1600 DPI with a dedicated switch.",
         "2.4G wireless with a nano receiver, up to 10 m range.",
         "Silent left and right clicks rated for 5 million presses.",
         "Powered by a single AA battery, up to 18 months of use."},
        {"Seven ports in one: HDMI, two USB-A, USB-C, SD, microSD and Ethernet.",
         "4K HDMI output at 30 Hz, mirrors or extends the display.",
         "100 W pass-through charging keeps the laptop topped up.",
         "Aluminium shell spreads heat during long transfers.",
         "Plug and play, no driver install needed."},
        {"Six height settings from 6 to 24 cm for a neutral neck angle.",
         "Aluminium frame holds up to 8 kg without flexing.",
         "Open back lets air move under the laptop while it works.",
         "Silicone pads grip the desk and protect the case.",
         "Folds flat to 2 cm and slides into a laptop sleeve."},
    };
    private static final String[][][] REVIEW_TEXT = {
        {{"Comfortable after long sessions", "5.0",
          "Switched from a flat mouse and the wrist ache is gone."},
         {"Good value", "4.0", "Tracking is accurate and the receiver is tiny."},
         {"Quiet clicks", "4.0", "Much quieter than my old mouse, good in a shared office."}},
        {{"Replaced four dongles", "5.0",
          "One cable to the laptop and everything on the desk just works."},
         {"Runs warm but fine", "4.0", "Warm on big file copies, never dropped a transfer."},
         {"Solid build", "4.0", "Metal body, the cable feels like it will last."}},
        {{"Better posture straight away", "5.0",
          "Screen is at eye level now and my neck stopped complaining."},
         {"Stable at full height", "4.0", "No wobble even when typing hard."},
         {"Packs down small", "4.0", "Folds flat enough to carry between buildings."}},
    };

    private final SimulatedTaskActivity host;
    private TaskBanner banner;

    private final boolean[] didSearch = new boolean[ROUNDS];
    private final boolean[] didOpen = new boolean[ROUNDS];
    private final boolean[] didGesture = new boolean[ROUNDS];
    private final boolean[] didAdd = new boolean[ROUNDS];
    private int round;
    private int scrollBase;
    private TextView addToCart;

    AmazonSim(SimulatedTaskActivity host) {
        this.host = host;
    }

    // ---- checklist ------------------------------------------------------

    private String[] steps() {
        int r = round;
        return new String[] {
            "Search for \"" + QUERIES[r] + "\"",
            "Open result " + TARGETS[r] + ", \"" + shortTitle(TITLES[r][TARGETS[r] - 1]) + "\"",
            GESTURE_STEP[r],
            r + 1 < ROUNDS ? "Add it to the cart" : "Add it to the cart, then finish",
        };
    }

    private static String shortTitle(String value) {
        int comma = value.indexOf(',');
        return comma > 0 ? value.substring(0, comma) : value;
    }

    /** True once the round's required gesture has been performed. */
    private boolean gestureDone(int r) {
        if (r == 0) return host.swipes >= 1;
        if (r == 1) return host.scrolls - scrollBase >= 2;
        return host.pinches >= 1;
    }

    private void recompute() {
        if ("amazon_product".equals(host.phase()) && gestureDone(round)) {
            didGesture[round] = true;
        }
        if (banner != null) {
            banner.setStates(new boolean[] {
                didSearch[round], didOpen[round], didGesture[round], didAdd[round],
            });
        }
        if (addToCart != null) {
            boolean ready = didGesture[round];
            addToCart.setEnabled(ready);
            addToCart.setBackground(Ui.rounded(host,
                    ready ? Ui.AMZ_YELLOW : Color.parseColor("#E7E9EA"), 22));
            addToCart.setTextColor(ready ? Ui.INK : Color.parseColor("#9AA0A6"));
        }
    }

    private TaskBanner newBanner() {
        banner = new TaskBanner(host, ACCENT, TINT);
        banner.setRound(round + 1, ROUNDS);
        banner.setSteps(steps());
        host.setProgressListener(this::recompute);
        return banner;
    }

    // ---- home -----------------------------------------------------------

    void showHome() {
        host.setBackAction(null);   // top of this app: Back means leave
        host.setPhase("amazon_home");
        host.setChrome(Ui.AMZ_NAVY, false);
        addToCart = null;
        final int r = round;

        LinearLayout screen = Ui.col(host);
        screen.setBackgroundColor(Ui.AMZ_BG);

        EditText input = host.trackedInput("Search Amazon");
        screen.addView(searchHeader(input, () -> {
            if (host.submitText(input, "amazon_query", QUERIES[r])) {
                didSearch[r] = true;
                host.hideKeyboard(input);
                showResults();
            }
        }), Ui.matchWrap());
        screen.addView(newBanner(), TaskBanner.params(host));

        ScrollView scroll = new ScrollView(host);
        LinearLayout page = Ui.col(host);
        page.setPadding(Ui.dp(host, 10), Ui.dp(host, 6), Ui.dp(host, 10), Ui.dp(host, 20));

        if (host.cartCount() > 0) {
            LinearLayout note = card();
            LinearLayout row = Ui.row(host);
            row.addView(Ui.icon(host, R.drawable.ic_check, 20, Ui.AMZ_GREEN));
            row.addView(Ui.text(host,
                            host.cartCount() + (host.cartCount() > 1 ? " items" : " item")
                                    + " in your cart", 15, Ui.AMZ_GREEN, true),
                    Ui.margins(host, Ui.wrap(), 8, 0, 0, 0));
            note.addView(row, Ui.matchWrap());
            page.addView(note, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 10));
        }

        LinearLayout deal = card();
        deal.addView(Ui.text(host, "Today's Deals", 18, Ui.INK, true), Ui.matchWrap());
        deal.addView(Ui.text(host, "Up to 40% off computer accessories", 13, Ui.INK_SOFT),
                Ui.margins(host, Ui.matchWrap(), 0, 3, 0, 9));
        LinearLayout strip = Ui.row(host);
        for (int i = 0; i < 3; i++) {
            ImageView photo = new ImageView(host);
            photo.setImageResource(PHOTOS[r][i]);
            photo.setScaleType(ImageView.ScaleType.CENTER_CROP);
            strip.addView(photo, Ui.margins(host, Ui.size(host, 96, 96), 0, 0, 8, 0));
        }
        deal.addView(strip, Ui.matchWrap());
        page.addView(deal, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 10));

        LinearLayout cats = card();
        cats.addView(Ui.text(host, "Shop by category", 18, Ui.INK, true),
                Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 10));
        cats.addView(categoryRow(R.drawable.cat_electronics, "Electronics",
                R.drawable.cat_home, "Home"), Ui.matchWrap());
        cats.addView(categoryRow(R.drawable.cat_sports, "Sports",
                R.drawable.cat_books, "Books"),
                Ui.margins(host, Ui.matchWrap(), 0, 10, 0, 0));
        page.addView(cats, Ui.matchWrap());

        scroll.addView(page);
        screen.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        host.show(screen);
        host.focus(input);
        recompute();
    }

    private LinearLayout categoryRow(int leftIcon, String left, int rightIcon, String right) {
        LinearLayout row = Ui.row(host);
        row.addView(categoryTile(leftIcon, left), Ui.weight(1f));
        View gap = new View(host);
        row.addView(gap, new LinearLayout.LayoutParams(Ui.dp(host, 10), 1));
        row.addView(categoryTile(rightIcon, right), Ui.weight(1f));
        return row;
    }

    private LinearLayout categoryTile(int iconRes, String label) {
        LinearLayout tile = Ui.col(host);
        tile.setBackground(Ui.rounded(host, Color.parseColor("#F7F8F8"), 6,
                Color.parseColor("#E3E6E6"), 1));
        tile.setPadding(Ui.dp(host, 8), Ui.dp(host, 8), Ui.dp(host, 8), Ui.dp(host, 8));
        ImageView icon = new ImageView(host);
        icon.setImageResource(iconRes);
        icon.setScaleType(ImageView.ScaleType.FIT_CENTER);
        tile.addView(icon, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(host, 70)));
        tile.addView(Ui.text(host, label, 13, Ui.INK, true),
                Ui.margins(host, Ui.matchWrap(), 0, 6, 0, 0));
        return tile;
    }

    // ---- results --------------------------------------------------------

    private void showResults() {
        host.setBackAction(this::showHome);
        host.setPhase("amazon_results");
        host.setChrome(Ui.AMZ_NAVY, false);
        scrollBase = host.scrolls;
        addToCart = null;
        final int r = round;

        LinearLayout screen = Ui.col(host);
        screen.setBackgroundColor(Ui.AMZ_BG);
        screen.addView(queryHeader(), Ui.matchWrap());
        screen.addView(newBanner(), TaskBanner.params(host));

        ScrollView scroll = new ScrollView(host);
        LinearLayout page = Ui.col(host);
        page.setPadding(Ui.dp(host, 10), Ui.dp(host, 6), Ui.dp(host, 10), Ui.dp(host, 24));

        page.addView(Ui.text(host, "1-6 of over 4,000 results for \"" + QUERIES[r] + "\"",
                12, Ui.INK_SOFT), Ui.margins(host, Ui.matchWrap(), 2, 0, 0, 8));
        page.addView(filterRow(), Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 10));
        for (int i = 0; i < PHOTOS[r].length; i++) {
            page.addView(resultCard(r, i), Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 10));
        }

        scroll.addView(page);
        screen.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        host.show(screen);
        recompute();
    }

    private View filterRow() {
        LinearLayout row = Ui.row(host);
        for (String label : new String[] {"Sort by: Featured", "Prime", "4★ & Up", "Under $50"}) {
            TextView chip = Ui.text(host, label, 12, Ui.INK);
            chip.setBackground(Ui.rounded(host, Color.WHITE, 14,
                    Color.parseColor("#D5D9D9"), 1));
            chip.setPadding(Ui.dp(host, 11), Ui.dp(host, 6), Ui.dp(host, 11), Ui.dp(host, 6));
            row.addView(chip, Ui.margins(host, Ui.wrap(), 0, 0, 7, 0));
        }
        HorizontalScrollView scroll = new HorizontalScrollView(host);
        scroll.setHorizontalScrollBarEnabled(false);
        scroll.addView(row);
        return scroll;
    }

    private LinearLayout resultCard(int r, int index) {
        LinearLayout card = card();
        LinearLayout row = Ui.row(host);
        row.setGravity(Gravity.TOP);

        ImageView photo = new ImageView(host);
        photo.setImageResource(PHOTOS[r][index]);
        photo.setScaleType(ImageView.ScaleType.FIT_CENTER);
        row.addView(photo, Ui.size(host, 104, 104));

        LinearLayout details = Ui.col(host);
        details.setPadding(Ui.dp(host, 10), 0, 0, 0);
        TextView title = Ui.text(host, TITLES[r][index], 14, Ui.INK);
        title.setMaxLines(2);
        title.setEllipsize(TextUtils.TruncateAt.END);
        title.setLineSpacing(0f, 1.12f);
        details.addView(title, Ui.matchWrap());

        LinearLayout ratingRow = Ui.row(host);
        ratingRow.addView(Ui.stars(host, RATINGS[r][index], 13));
        ratingRow.addView(Ui.text(host, String.valueOf(REVIEWS[r][index]), 12, Ui.AMZ_LINK),
                Ui.margins(host, Ui.wrap(), 5, 0, 0, 0));
        details.addView(ratingRow, Ui.margins(host, Ui.matchWrap(), 0, 5, 0, 0));
        details.addView(Ui.text(host, PRICES[r][index], 18, Ui.INK, true),
                Ui.margins(host, Ui.matchWrap(), 0, 4, 0, 0));

        LinearLayout delivery = Ui.row(host);
        delivery.addView(Ui.text(host, "prime", 11, Ui.AMZ_PRIME, true));
        delivery.addView(Ui.text(host, "FREE delivery Thu, Aug 13", 11, Ui.INK_SOFT),
                Ui.margins(host, Ui.wrap(), 6, 0, 0, 0));
        details.addView(delivery, Ui.margins(host, Ui.matchWrap(), 0, 4, 0, 0));

        row.addView(details, Ui.weight(1f));
        card.addView(row, Ui.matchWrap());

        final int position = index + 1;
        card.setOnClickListener(view -> {
            if (position == TARGETS[r]) {
                didOpen[r] = true;
                showProduct();
            }
        });
        return card;
    }

    // ---- product --------------------------------------------------------

    private void showProduct() {
        host.setBackAction(this::showResults);
        host.setPhase("amazon_product");
        host.setChrome(Ui.AMZ_NAVY, false);
        scrollBase = host.scrolls;
        final int r = round;
        final int target = TARGETS[r] - 1;

        LinearLayout screen = Ui.col(host);
        screen.setBackgroundColor(Color.WHITE);
        screen.addView(queryHeader(), Ui.matchWrap());
        screen.addView(newBanner(), TaskBanner.params(host));

        ScrollView scroll = new ScrollView(host);
        LinearLayout page = Ui.col(host);
        page.setPadding(Ui.dp(host, 14), Ui.dp(host, 6), Ui.dp(host, 14), Ui.dp(host, 28));

        ZoomPanel zoom = new ZoomPanel(host);
        zoom.setImage(PHOTOS[r][target]);
        page.addView(zoom, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(host, 220)));
        page.addView(thumbnailStrip(r, zoom), Ui.margins(host, Ui.matchWrap(), 0, 8, 0, 12));

        page.addView(Ui.text(host, TITLES[r][target], 19, Ui.INK), Ui.matchWrap());
        page.addView(Ui.text(host, "Visit the " + STORES[r], 13, Ui.AMZ_LINK),
                Ui.margins(host, Ui.matchWrap(), 0, 5, 0, 0));

        LinearLayout ratingRow = Ui.row(host);
        ratingRow.addView(Ui.stars(host, RATINGS[r][target], 15));
        ratingRow.addView(Ui.text(host, REVIEWS[r][target] + " ratings", 13, Ui.AMZ_LINK),
                Ui.margins(host, Ui.wrap(), 6, 0, 0, 0));
        page.addView(ratingRow, Ui.margins(host, Ui.matchWrap(), 0, 7, 0, 10));
        page.addView(Ui.divider(host, Color.parseColor("#E7E7E7")), Ui.matchWrap());

        LinearLayout priceRow = Ui.row(host);
        priceRow.addView(Ui.text(host, "-23%", 16, Ui.AMZ_RED, true));
        priceRow.addView(Ui.text(host, PRICES[r][target], 26, Ui.INK, true),
                Ui.margins(host, Ui.wrap(), 8, 0, 0, 0));
        page.addView(priceRow, Ui.margins(host, Ui.matchWrap(), 0, 12, 0, 0));

        LinearLayout delivery = Ui.row(host);
        delivery.addView(Ui.icon(host, R.drawable.ic_truck, 16, Ui.INK_SOFT));
        delivery.addView(Ui.text(host, "FREE delivery Thursday, August 13", 13, Ui.INK),
                Ui.margins(host, Ui.wrap(), 6, 0, 0, 0));
        page.addView(delivery, Ui.margins(host, Ui.matchWrap(), 0, 9, 0, 0));
        page.addView(Ui.text(host, "In Stock", 15, Ui.AMZ_GREEN, true),
                Ui.margins(host, Ui.matchWrap(), 0, 8, 0, 14));

        addToCart = pillButton("Add to Cart", Ui.AMZ_YELLOW, Ui.INK, view -> {
            if (!didGesture[round]) {
                return;
            }
            didAdd[round] = true;
            host.addToCart();
            if (round + 1 < ROUNDS) {
                round++;
                showHome();
            } else {
                showCart();
            }
        });
        page.addView(addToCart, Ui.matchWrap());
        page.addView(Ui.text(host, "The button unlocks once the step above is ticked off.",
                        12, Ui.INK_SOFT),
                Ui.margins(host, Ui.matchWrap(), 2, 7, 0, 16));

        page.addView(Ui.divider(host, Color.parseColor("#E7E7E7")), Ui.matchWrap());
        page.addView(Ui.text(host, "About this item", 17, Ui.INK, true),
                Ui.margins(host, Ui.matchWrap(), 0, 14, 0, 8));
        for (String bullet : BULLETS[r]) {
            LinearLayout row = Ui.row(host);
            row.setGravity(Gravity.TOP);
            row.addView(Ui.text(host, "•", 14, Ui.INK));
            TextView copy = Ui.text(host, bullet, 14, Ui.INK);
            copy.setLineSpacing(0f, 1.2f);
            row.addView(copy, Ui.margins(host, Ui.weight(1f), 8, 0, 0, 0));
            page.addView(row, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 7));
        }

        page.addView(Ui.divider(host, Color.parseColor("#E7E7E7")),
                Ui.margins(host, Ui.matchWrap(), 0, 10, 0, 0));
        page.addView(Ui.text(host, "Customer reviews", 17, Ui.INK, true),
                Ui.margins(host, Ui.matchWrap(), 0, 14, 0, 10));
        for (String[] rv : REVIEW_TEXT[r]) {
            page.addView(review(rv[0], Double.parseDouble(rv[1]), rv[2]),
                    Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 12));
        }

        scroll.addView(page);
        screen.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        host.show(screen);
        recompute();
    }

    private View thumbnailStrip(int r, ZoomPanel target) {
        LinearLayout strip = Ui.row(host);
        for (int repeat = 0; repeat < 2; repeat++) {
            for (int i = 0; i < PHOTOS[r].length; i++) {
                final int resource = PHOTOS[r][i];
                ImageView thumb = new ImageView(host);
                thumb.setImageResource(resource);
                thumb.setScaleType(ImageView.ScaleType.FIT_CENTER);
                thumb.setBackground(Ui.rounded(host, Color.WHITE, 6,
                        Color.parseColor("#D5D9D9"), 1));
                thumb.setOnClickListener(view -> target.setImage(resource));
                strip.addView(thumb, Ui.margins(host, Ui.size(host, 76, 76), 0, 0, 8, 0));
            }
        }
        HorizontalScrollView scroll = new HorizontalScrollView(host);
        scroll.setHorizontalScrollBarEnabled(false);
        scroll.addView(strip);
        return scroll;
    }

    private LinearLayout review(String title, double rating, String body) {
        LinearLayout block = Ui.col(host);
        LinearLayout head = Ui.row(host);
        head.addView(Ui.stars(host, rating, 13));
        head.addView(Ui.text(host, title, 14, Ui.INK, true),
                Ui.margins(host, Ui.wrap(), 7, 0, 0, 0));
        block.addView(head, Ui.matchWrap());
        TextView copy = Ui.text(host, body, 13, Ui.INK_SOFT);
        copy.setLineSpacing(0f, 1.2f);
        block.addView(copy, Ui.margins(host, Ui.matchWrap(), 0, 5, 0, 0));
        return block;
    }

    // ---- cart -----------------------------------------------------------

    private void showCart() {
        host.setBackAction(this::showResults);
        host.setPhase("amazon_cart");
        host.setChrome(Ui.AMZ_NAVY, false);
        addToCart = null;

        LinearLayout screen = Ui.col(host);
        screen.setBackgroundColor(Ui.AMZ_BG);
        screen.addView(queryHeader(), Ui.matchWrap());
        screen.addView(newBanner(), TaskBanner.params(host));

        ScrollView scroll = new ScrollView(host);
        LinearLayout page = Ui.col(host);
        page.setPadding(Ui.dp(host, 12), Ui.dp(host, 6), Ui.dp(host, 12), Ui.dp(host, 24));

        LinearLayout addedCard = card();
        LinearLayout added = Ui.row(host);
        added.addView(Ui.icon(host, R.drawable.ic_check, 22, Ui.AMZ_GREEN));
        added.addView(Ui.text(host, "All " + ROUNDS + " items added", 18, Ui.AMZ_GREEN, true),
                Ui.margins(host, Ui.wrap(), 8, 0, 0, 0));
        addedCard.addView(added, Ui.matchWrap());
        page.addView(addedCard, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 10));

        double total = 0;
        for (int r = 0; r < ROUNDS; r++) {
            int target = TARGETS[r] - 1;
            total += Double.parseDouble(PRICES[r][target].substring(1));
            LinearLayout item = card();
            LinearLayout itemRow = Ui.row(host);
            itemRow.setGravity(Gravity.TOP);
            ImageView photo = new ImageView(host);
            photo.setImageResource(PHOTOS[r][target]);
            photo.setScaleType(ImageView.ScaleType.FIT_CENTER);
            itemRow.addView(photo, Ui.size(host, 92, 92));
            LinearLayout d = Ui.col(host);
            d.setPadding(Ui.dp(host, 10), 0, 0, 0);
            d.addView(Ui.text(host, TITLES[r][target], 14, Ui.INK), Ui.matchWrap());
            d.addView(Ui.text(host, PRICES[r][target], 17, Ui.INK, true),
                    Ui.margins(host, Ui.matchWrap(), 0, 5, 0, 0));
            d.addView(Ui.text(host, "Qty: 1", 13, Ui.INK_SOFT),
                    Ui.margins(host, Ui.matchWrap(), 0, 5, 0, 0));
            itemRow.addView(d, Ui.weight(1f));
            item.addView(itemRow, Ui.matchWrap());
            page.addView(item, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 10));
        }

        LinearLayout totals = card();
        totals.addView(Ui.text(host,
                        String.format(java.util.Locale.US,
                                "Subtotal (%d items): $%.2f", ROUNDS, total),
                        17, Ui.INK, true), Ui.matchWrap());
        totals.addView(Ui.text(host, "Delivery: FREE", 13, Ui.INK_SOFT),
                Ui.margins(host, Ui.matchWrap(), 0, 4, 0, 0));
        page.addView(totals, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 16));

        page.addView(pillButton("Finish task", Color.parseColor("#1F2933"), Color.WHITE,
                view -> host.finishIfValid()), Ui.matchWrap());

        scroll.addView(page);
        screen.addView(scroll, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        host.show(screen);
        recompute();
    }

    // ---- chrome ---------------------------------------------------------

    private LinearLayout searchHeader(EditText input, Runnable onSearch) {
        final int r = round;
        LinearLayout header = Ui.col(host);
        header.setBackgroundColor(Ui.AMZ_NAVY);
        header.setPadding(Ui.dp(host, 12), Ui.dp(host, 10), Ui.dp(host, 12), Ui.dp(host, 10));

        LinearLayout brandRow = Ui.row(host);
        ImageView logo = new ImageView(host);
        logo.setImageResource(R.drawable.ic_cart);
        logo.setAdjustViewBounds(true);
        logo.setScaleType(ImageView.ScaleType.FIT_START);
        brandRow.addView(logo, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, Ui.dp(host, 30)));
        brandRow.addView(Ui.flexSpacer(host));
        brandRow.addView(cartIcon());
        ImageView exit = Ui.icon(host, R.drawable.ic_close, 22, Color.WHITE);
        exit.setContentDescription("Close task");
        exit.setOnClickListener(view -> host.confirmExit());
        brandRow.addView(exit, Ui.margins(host, Ui.size(host, 22, 22), 18, 0, 0, 0));
        header.addView(brandRow, Ui.margins(host, Ui.matchWrap(), 0, 0, 0, 10));

        LinearLayout searchRow = Ui.row(host);
        searchRow.setBackground(Ui.rounded(host, Color.WHITE, 8));
        input.setBackground(null);
        input.setTextSize(15);
        input.setTextColor(Ui.INK);
        input.setHintTextColor(Color.parseColor("#8C8C8C"));
        input.setPadding(Ui.dp(host, 12), Ui.dp(host, 10), Ui.dp(host, 8), Ui.dp(host, 10));
        searchRow.addView(input, Ui.weight(1f));

        LinearLayout searchButton = Ui.row(host);
        searchButton.setGravity(Gravity.CENTER);
        searchButton.addView(Ui.icon(host, R.drawable.ic_search, 22, Ui.AMZ_NAVY));
        searchButton.setContentDescription("Search");
        Runnable submit = () -> {
            if (Ui.matches(input, QUERIES[r])) {
                onSearch.run();
            }
        };
        searchButton.setOnClickListener(view -> submit.run());
        // The keyboard's own action key now submits too.  It did nothing at all
        // before, so the only way in was the on-screen button -- which every
        // user, and every agent, has to discover by failing at Enter first.
        host.submitOn(input, submit);
        Runnable gate = () -> {
            boolean ready = Ui.matches(input, QUERIES[r]);
            searchButton.setEnabled(ready);
            searchButton.setBackground(Ui.rounded(host,
                    ready ? Ui.AMZ_ORANGE : Color.parseColor("#E7E9EA"), 8));
        };
        Ui.onTextChange(input, gate);
        gate.run();
        searchRow.addView(searchButton, Ui.size(host, 50, 44));
        header.addView(searchRow, Ui.matchWrap());

        LinearLayout locationRow = Ui.row(host);
        locationRow.addView(Ui.icon(host, R.drawable.ic_location, 15, Color.WHITE));
        locationRow.addView(Ui.text(host, "Deliver to Study Room 214", 12,
                Color.parseColor("#D8DDE1")), Ui.margins(host, Ui.wrap(), 5, 0, 0, 0));
        header.addView(locationRow, Ui.margins(host, Ui.matchWrap(), 2, 8, 0, 0));
        return header;
    }

    private LinearLayout queryHeader() {
        LinearLayout header = Ui.col(host);
        header.setBackgroundColor(Ui.AMZ_NAVY);
        header.setPadding(Ui.dp(host, 10), Ui.dp(host, 10), Ui.dp(host, 12), Ui.dp(host, 10));

        LinearLayout row = Ui.row(host);
        ImageView exit = Ui.icon(host, R.drawable.ic_close, 22, Color.WHITE);
        exit.setContentDescription("Close task");
        exit.setOnClickListener(view -> host.confirmExit());
        row.addView(exit);

        LinearLayout pill = Ui.row(host);
        pill.setBackground(Ui.rounded(host, Color.WHITE, 8));
        pill.setPadding(Ui.dp(host, 10), Ui.dp(host, 9), Ui.dp(host, 10), Ui.dp(host, 9));
        pill.addView(Ui.icon(host, R.drawable.ic_search, 18, Color.parseColor("#8C8C8C")));
        pill.addView(Ui.text(host, QUERIES[round], 15, Ui.INK),
                Ui.margins(host, Ui.wrap(), 8, 0, 0, 0));
        row.addView(pill, Ui.margins(host, Ui.weight(1f), 8, 0, 8, 0));
        row.addView(cartIcon());
        header.addView(row, Ui.matchWrap());
        return header;
    }

    /** Cart icon carrying the running item count, so progress shows in-app. */
    private LinearLayout cartIcon() {
        LinearLayout holder = Ui.row(host);
        holder.addView(Ui.icon(host, R.drawable.ic_cart, 24, Color.WHITE));
        if (host.cartCount() > 0) {
            TextView badge = Ui.text(host, String.valueOf(host.cartCount()), 11,
                    Ui.AMZ_NAVY, true);
            badge.setGravity(Gravity.CENTER);
            badge.setBackground(Ui.circle(Ui.AMZ_ORANGE));
            badge.setPadding(Ui.dp(host, 6), Ui.dp(host, 1), Ui.dp(host, 6), Ui.dp(host, 1));
            holder.addView(badge, Ui.margins(host, Ui.wrap(), 3, 0, 0, 0));
        }
        return holder;
    }

    private LinearLayout card() {
        LinearLayout card = Ui.col(host);
        card.setBackgroundColor(Color.WHITE);
        card.setPadding(Ui.dp(host, 12), Ui.dp(host, 12), Ui.dp(host, 12), Ui.dp(host, 12));
        return card;
    }

    private TextView pillButton(String label, int background, int ink,
            View.OnClickListener click) {
        TextView button = Ui.text(host, label, 15, ink, true);
        button.setTypeface(Typeface.DEFAULT_BOLD);
        button.setGravity(Gravity.CENTER);
        button.setBackground(Ui.rounded(host, background, 22));
        button.setPadding(Ui.dp(host, 16), Ui.dp(host, 13), Ui.dp(host, 16), Ui.dp(host, 13));
        button.setOnClickListener(click);
        return button;
    }
}
