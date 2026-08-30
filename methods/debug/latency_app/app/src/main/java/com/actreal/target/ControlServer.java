package com.actreal.target;

import android.os.SystemClock;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/**
 * The host's line to the app: newline-delimited JSON over loopback.
 *
 * <p>Reached with {@code adb forward tcp:PORT tcp:PORT}, so nothing here needs
 * root, a debugger, or a repackaged third-party app. The app is ours, so the
 * host simply tells it what inertia to deliver -- which is what the threat
 * model already assumes an attacker at this layer can do.
 */
public final class ControlServer {

    public static final int DEFAULT_PORT = 8129;

    public interface Host {
        JSONObject describe() throws JSONException;

        void setImuMode(int mode);

        int imuMode();

        void scheduleImu(long startElapsedNs, double periodMs, float[][] frames, String bundleId);

        void setBackground(float[][] frames, double periodMs);

        void playTouch(long startUptimeMs, java.util.List<TouchInjector.Point> points);

        void clearRecordings();

        void clearScheduled();

        JSONObject dump(boolean includeRows) throws JSONException;

        JSONObject stats() throws JSONException;
    }

    private final int port;
    private final Host host;
    private Thread thread;
    private ServerSocket server;
    private volatile boolean running;
    private volatile String lastError = "";
    private volatile int connections;

    public ControlServer(int port, Host host) {
        this.port = port;
        this.host = host;
    }

    public int port() {
        return port;
    }

    public boolean isRunning() {
        return running;
    }

    public int connectionCount() {
        return connections;
    }

    public String lastError() {
        return lastError;
    }

    public synchronized void start() {
        if (running) {
            return;
        }
        running = true;
        thread = new Thread(this::accept, "actreal-control");
        thread.start();
    }

    public synchronized void stop() {
        running = false;
        try {
            if (server != null) {
                server.close();
            }
        } catch (IOException ignored) {
            // Closing the socket is how the accept loop is woken.
        }
    }

    private void accept() {
        try {
            server = new ServerSocket(port, 4, InetAddress.getByName("127.0.0.1"));
        } catch (IOException error) {
            lastError = "bind: " + error;
            running = false;
            return;
        }
        while (running) {
            try (Socket socket = server.accept()) {
                connections++;
                socket.setTcpNoDelay(true);
                serve(socket);
            } catch (IOException error) {
                if (running) {
                    lastError = "accept: " + error;
                }
            }
        }
    }

    private void serve(Socket socket) throws IOException {
        BufferedReader reader =
                new BufferedReader(
                        new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
        BufferedWriter writer =
                new BufferedWriter(
                        new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
        String line;
        while (running && (line = reader.readLine()) != null) {
            if (line.isEmpty()) {
                continue;
            }
            String reply;
            try {
                reply = handle(new JSONObject(line)).toString();
            } catch (JSONException | RuntimeException error) {
                reply = errorReply(error).toString();
            }
            writer.write(reply);
            writer.write('\n');
            writer.flush();
        }
    }

    private static JSONObject errorReply(Throwable error) {
        JSONObject out = new JSONObject();
        try {
            out.put("ok", false);
            out.put("error", error.getClass().getSimpleName() + ": " + error.getMessage());
        } catch (JSONException ignored) {
            // A JSONObject with two string values cannot fail to build.
        }
        return out;
    }

    private JSONObject handle(JSONObject request) throws JSONException {
        String cmd = request.optString("cmd", "");
        JSONObject out = new JSONObject();
        out.put("ok", true);
        out.put("cmd", cmd);

        switch (cmd) {
            case "ping": {
                long[] both = Clocks.sampleBoth();
                out.put("uptime_ms", both[0]);
                out.put("elapsed_ns", both[1]);
                out.put("read_window_ns", both[2]);
                return out;
            }
            case "hello": {
                out.put("device", host.describe());
                long[] both = Clocks.sampleBoth();
                out.put("uptime_ms", both[0]);
                out.put("elapsed_ns", both[1]);
                out.put("read_window_ns", both[2]);
                out.put("clock_offset_ns", both[1] - both[0] * 1_000_000L);
                out.put("protocol", "actreal_control_v1");
                return out;
            }
            case "mode": {
                String imu = request.optString("imu", "real");
                int mode = "injected".equals(imu) ? ImuBus.MODE_INJECTED : ImuBus.MODE_REAL;
                host.setImuMode(mode);
                out.put("imu", host.imuMode() == ImuBus.MODE_INJECTED ? "injected" : "real");
                return out;
            }
            case "background": {
                double periodMs = request.optDouble("period_ms", 10.0);
                host.setBackground(readFrames(request.optJSONArray("frames")), periodMs);
                out.put("period_ms", periodMs);
                return out;
            }
            case "imu": {
                float[][] frames = readFrames(request.optJSONArray("frames"));
                if (frames.length == 0) {
                    throw new IllegalArgumentException("imu needs at least one frame");
                }
                double periodMs = request.optDouble("period_ms", 10.0);
                long startElapsedNs = resolveStart(request);
                host.scheduleImu(
                        startElapsedNs, periodMs, frames, request.optString("bundle_id", ""));
                out.put("frames", frames.length);
                out.put("start_elapsed_ns", startElapsedNs);
                out.put("now_elapsed_ns", SystemClock.elapsedRealtimeNanos());
                return out;
            }
            case "touch": {
                JSONArray array = request.optJSONArray("points");
                if (array == null || array.length() == 0) {
                    throw new IllegalArgumentException("touch needs at least one point");
                }
                java.util.List<TouchInjector.Point> points =
                        new java.util.ArrayList<>(array.length());
                for (int i = 0; i < array.length(); i++) {
                    JSONObject p = array.getJSONObject(i);
                    points.add(
                            new TouchInjector.Point(
                                    p.getDouble("t_ms"),
                                    (float) p.getDouble("x"),
                                    (float) p.getDouble("y"),
                                    (float) p.optDouble("pressure", 1.0),
                                    (float) p.optDouble("size", 0.0),
                                    p.optInt("pointer_id", 0),
                                    p.optString("action", "MOVE")));
                }
                long startUptimeMs =
                        request.has("start_uptime_ms")
                                ? request.getLong("start_uptime_ms")
                                : SystemClock.uptimeMillis()
                                        + Math.round(request.optDouble("start_in_ms", 0.0));
                host.playTouch(startUptimeMs, points);
                out.put("points", points.size());
                out.put("start_uptime_ms", startUptimeMs);
                out.put("now_uptime_ms", SystemClock.uptimeMillis());
                return out;
            }
            case "clear": {
                host.clearRecordings();
                if (request.optBoolean("scheduled", false)) {
                    host.clearScheduled();
                }
                return out;
            }
            case "dump": {
                out.put("data", host.dump(request.optBoolean("rows", true)));
                return out;
            }
            case "stats": {
                out.put("data", host.stats());
                return out;
            }
            default:
                throw new IllegalArgumentException("unknown cmd " + cmd);
        }
    }

    /**
     * Resolve when a window should start, on the sensor clock.
     *
     * <p>The host plans on uptime because that is the clock a MotionEvent
     * carries, so a plan expressed that way is converted here with an offset
     * measured on the device rather than one the host guessed.
     */
    private long resolveStart(JSONObject request) throws JSONException {
        if (request.has("start_elapsed_ns")) {
            return request.getLong("start_elapsed_ns");
        }
        long offsetNs = Clocks.uptimeToElapsedOffsetNs();
        if (request.has("start_uptime_ms")) {
            return Clocks.uptimeMsToElapsedNs(request.getLong("start_uptime_ms"), offsetNs);
        }
        double delayMs = request.optDouble("start_in_ms", 0.0);
        return SystemClock.elapsedRealtimeNanos() + Math.round(delayMs * 1_000_000.0);
    }

    private static float[][] readFrames(JSONArray array) throws JSONException {
        if (array == null) {
            return new float[0][];
        }
        float[][] frames = new float[array.length()][];
        for (int i = 0; i < array.length(); i++) {
            JSONArray row = array.getJSONArray(i);
            if (row.length() != 6) {
                throw new IllegalArgumentException(
                        "frame " + i + " has " + row.length() + " channels, expected 6");
            }
            float[] values = new float[6];
            for (int c = 0; c < 6; c++) {
                values[c] = (float) row.getDouble(c);
            }
            frames[i] = values;
        }
        return frames;
    }
}
