package com.sensorworldmodel.collector;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.WindowManager;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;

/**
 * Newline-delimited JSON over a loopback socket, published by {@code adb forward}.
 *
 * <p>The wire format is not invented here: it is the protocol the host's
 * {@code actreal.control.ControlClient} already speaks, specified command for
 * command by {@code actreal/simulator.py}, which implements the same thing in
 * Python so the host side can be exercised without a phone. Matching it means
 * the host needs no changes at all to drive this app.
 *
 * <p>Only the commands this app can honour are implemented. There is no
 * {@code touch}: touches arrive through the real Android input pipeline from a
 * virtual input device, so they are already MotionEvents by the time this
 * process sees them, and an app that dispatched its own would be reproducing
 * strictly less -- no device id, no source flags, no driver batching.
 *
 * <p>Loopback only. The socket binds to 127.0.0.1 and is reachable from this
 * device alone; a host reaches it through {@code adb forward tcp:8129}, which
 * requires USB debugging already trusted for that machine.
 */
final class ControlServer {

    static final int PORT = 8129;
    private static final String TAG = "ActRealControl";
    private static final String PROTOCOL = "actreal_control_v1";

    private final Context context;
    private Thread thread;
    private ServerSocket serverSocket;
    private volatile boolean running;
    private volatile int connections;
    private volatile String lastError = "";

    ControlServer(Context context) {
        this.context = context.getApplicationContext();
    }

    synchronized void start() {
        if (running) {
            return;
        }
        running = true;
        thread = new Thread(new Runnable() {
            @Override
            public void run() {
                serve();
            }
        }, "actreal-control");
        thread.setDaemon(true);
        thread.start();
    }

    synchronized void stop() {
        running = false;
        try {
            if (serverSocket != null) {
                serverSocket.close();
            }
        } catch (IOException ignored) {
            // Closing the listener is how the accept loop is woken to exit.
        }
        serverSocket = null;
    }

    int connections() {
        return connections;
    }

    String lastError() {
        return lastError;
    }

    private void serve() {
        try {
            serverSocket = new ServerSocket(PORT, 4, InetAddress.getByName("127.0.0.1"));
            serverSocket.setReuseAddress(true);
        } catch (IOException error) {
            lastError = "bind: " + error;
            Log.e(TAG, "cannot bind " + PORT, error);
            return;
        }
        while (running) {
            try {
                final Socket socket = serverSocket.accept();
                connections++;
                // One thread per client.  Handling inline meant the accept loop
                // was blocked for as long as a client stayed connected, so a
                // single process that died without closing its socket -- a
                // crashed script, a retry that left one half-open -- wedged the
                // channel for every later caller.  adb forward still accepts on
                // the host side and then closes, which surfaces as "the target
                // app closed the control channel" and looks like the app is
                // down when it is merely busy with a peer that has gone away.
                Thread worker = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        handle(socket);
                    }
                }, "actreal-control-client");
                worker.setDaemon(true);
                worker.start();
            } catch (IOException error) {
                if (running) {
                    lastError = "accept: " + error;
                }
            }
        }
    }

    private void handle(Socket socket) {
        try {
            socket.setTcpNoDelay(true);
            BufferedReader reader =
                    new BufferedReader(new InputStreamReader(socket.getInputStream(), "UTF-8"));
            BufferedWriter writer =
                    new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), "UTF-8"));
            String line;
            while (running && (line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                JSONObject reply;
                try {
                    reply = dispatch(new JSONObject(line));
                } catch (Exception error) {
                    // Surfaced to the caller the way the app does it, rather
                    // than dropping the connection and leaving the host to
                    // guess whether the command was even understood.
                    reply = new JSONObject();
                    reply.put("ok", false);
                    reply.put("error", error.getClass().getSimpleName() + ": " + error.getMessage());
                }
                writer.write(reply.toString());
                writer.write('\n');
                writer.flush();
            }
        } catch (Exception error) {
            lastError = "session: " + error;
        } finally {
            try {
                socket.close();
            } catch (IOException ignored) {
                // The peer went away; nothing left to report to.
            }
        }
    }

    private synchronized JSONObject dispatch(JSONObject request) throws Exception {
        String command = request.optString("cmd", "");
        JSONObject reply = new JSONObject();
        reply.put("cmd", command);
        reply.put("ok", true);

        ImuBus bus = ImuBus.get();

        if ("ping".equals(command)) {
            Clocks.Sample sample = Clocks.read();
            reply.put("uptime_ms", sample.uptimeMs);
            reply.put("elapsed_ns", sample.elapsedNs);
            reply.put("read_window_ns", sample.readWindowNs);
            return reply;
        }

        if ("hello".equals(command)) {
            return hello(reply);
        }

        if ("mode".equals(command)) {
            reply.put("imu", bus.setMode(request.optString("imu", ImuBus.MODE_REAL)));
            return reply;
        }

        if ("background".equals(command)) {
            double periodMs = request.optDouble("period_ms", 10.0);
            bus.setBackground(frames(request.optJSONArray("frames")), periodMs);
            reply.put("period_ms", periodMs);
            return reply;
        }

        if ("imu".equals(command)) {
            return scheduleImu(request, reply, bus);
        }

        if ("clear".equals(command)) {
            if (request.optBoolean("scheduled", false)) {
                bus.clearScheduled();
            }
            bus.clearRows();
            TouchLog.get().clear();
            return reply;
        }

        if ("stats".equals(command)) {
            reply.put("data", stats(bus));
            return reply;
        }

        if ("dump".equals(command)) {
            reply.put("data", dump(bus, request.optBoolean("rows", true)));
            return reply;
        }

        reply.put("ok", false);
        reply.put("error", "unknown cmd " + command);
        return reply;
    }

    private JSONObject hello(JSONObject reply) throws Exception {
        Clocks.Sample sample = Clocks.best(9);
        reply.put("protocol", PROTOCOL);
        reply.put("uptime_ms", sample.uptimeMs);
        reply.put("elapsed_ns", sample.elapsedNs);
        reply.put("read_window_ns", sample.readWindowNs);
        reply.put("clock_offset_ns", sample.offsetNs());
        // Who is on the other end.  The host reports a configuration's reach
        // from this, instead of naming a target app it merely assumed it was
        // talking to.
        reply.put("package", context.getPackageName());

        WindowManager windows = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        DisplayMetrics metrics = new DisplayMetrics();
        int width = 0;
        int height = 0;
        int density = 0;
        if (windows != null) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                android.graphics.Rect bounds =
                        windows.getCurrentWindowMetrics().getBounds();
                width = bounds.width();
                height = bounds.height();
            }
            metrics = context.getResources().getDisplayMetrics();
            if (width == 0) {
                width = metrics.widthPixels;
                height = metrics.heightPixels;
            }
            density = metrics.densityDpi;
        }

        JSONObject device = new JSONObject();
        device.put("model", Build.MODEL);
        device.put("manufacturer", Build.MANUFACTURER);
        device.put("display_w", width);
        device.put("display_h", height);
        device.put("density_dpi", density);
        // This app draws its study pages over the whole screen, so the mapped
        // rectangle is the screen; the field exists because the host's mapping
        // expects it and a letterboxed target would need it.
        device.put("source_w", width);
        device.put("source_h", height);
        JSONObject rect = new JSONObject();
        rect.put("left", 0);
        rect.put("top", 0);
        rect.put("right", width);
        rect.put("bottom", height);
        device.put("usable_rect", rect);

        SensorManager sensors = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        device.put("accel", sensorInfo(sensors, Sensor.TYPE_ACCELEROMETER));
        device.put("gyro", sensorInfo(sensors, Sensor.TYPE_GYROSCOPE));
        device.put("real_rate_hz", 100.0);
        reply.put("device", device);
        return reply;
    }

    private JSONObject sensorInfo(SensorManager sensors, int type) throws Exception {
        JSONObject info = new JSONObject();
        Sensor sensor = sensors == null ? null : sensors.getDefaultSensor(type);
        info.put("present", sensor != null);
        info.put("min_delay_us", sensor == null ? 0 : sensor.getMinDelay());
        return info;
    }

    private JSONObject scheduleImu(JSONObject request, JSONObject reply, ImuBus bus)
            throws Exception {
        if (!ImuBus.MODE_INJECTED.equals(bus.mode())) {
            throw new IllegalStateException("imu scheduled while the bus is in real mode");
        }
        float[][] frames = frames(request.optJSONArray("frames"));
        if (frames.length == 0) {
            throw new IllegalArgumentException("imu needs at least one frame");
        }
        double periodMs = request.optDouble("period_ms", 10.0);
        String bundleId = request.optString("bundle_id", "");

        long startElapsedNs;
        if (request.has("start_elapsed_ns")) {
            startElapsedNs = request.getLong("start_elapsed_ns");
        } else if (request.has("start_uptime_ms")) {
            // The host names the instant on the touch clock; this device owns
            // the conversion, because it measured the offset itself.
            startElapsedNs = Clocks.elapsedNsAtUptimeMs(request.getLong("start_uptime_ms"));
        } else {
            double startInMs = request.optDouble("start_in_ms", 0.0);
            startElapsedNs = SystemClock.elapsedRealtimeNanos()
                    + Math.round(startInMs * 1_000_000.0);
        }

        int scheduled = bus.schedule(frames, periodMs, startElapsedNs, bundleId);
        reply.put("frames", scheduled);
        reply.put("start_elapsed_ns", startElapsedNs);
        reply.put("now_elapsed_ns", SystemClock.elapsedRealtimeNanos());
        return reply;
    }

    private JSONObject stats(ImuBus bus) throws Exception {
        JSONObject data = new JSONObject();
        data.put("imu_mode", bus.mode());
        data.put("attached", bus.attached());
        data.put("injected_frames", bus.injectedFrames());
        data.put("real_frames", bus.realFrames());
        data.put("injected_max_lateness_ms", round(bus.maxLatenessMs()));
        data.put("injected_mean_lateness_ms", round(bus.meanLatenessMs()));
        data.put("background_frames", bus.backgroundFrames());
        data.put("background_running", bus.backgroundRunning());
        data.put("background_delivered", bus.backgroundDelivered());
        data.put("imu_dropped", bus.droppedRows());
        data.put("connections", connections);
        data.put("control_error", lastError);
        return data;
    }

    private JSONObject dump(ImuBus bus, boolean withRows) throws Exception {
        JSONObject data = new JSONObject();
        List<ImuBus.Row> rows = bus.recent();
        List<TouchLog.Row> touches = TouchLog.get().recent();
        data.put("imu_rows", rows.size());
        data.put("imu_dropped", bus.droppedRows());
        data.put("touch_rows", touches.size());
        data.put("touch_dropped", TouchLog.get().dropped());
        data.put("clock_offset_ns", Clocks.best(5).offsetNs());
        if (withRows) {
            JSONArray imu = new JSONArray();
            for (ImuBus.Row row : rows) {
                // Column order matches the host's reader in actreal/session.py:
                // seq, timestamp, six channels, origin, bundle, frame index.
                JSONArray out = new JSONArray();
                out.put(row.seq);
                out.put(row.timestampNs);
                for (int i = 0; i < ImuBus.CHANNELS; i++) {
                    out.put(row.values[i]);
                }
                out.put(row.origin);
                out.put(row.bundleId);
                out.put(row.frameIndex);
                imu.put(out);
            }
            data.put("imu", imu);
            JSONArray touch = new JSONArray();
            for (TouchLog.Row row : touches) {
                // Column order is the host's reader in actreal/session.py:
                // seq, uptime, elapsed, action, batched, pointer count, index,
                // pointer id, x, y, pressure, size, then the provenance an
                // in-app dispatch cannot produce.
                JSONArray out = new JSONArray();
                out.put(row.seq);
                out.put(row.uptimeNs);
                out.put(row.elapsedNs);
                out.put(row.action);
                out.put(row.historical ? 1 : 0);
                out.put(row.pointerCount);
                out.put(0);
                out.put(row.pointerId);
                out.put(row.x);
                out.put(row.y);
                out.put(row.pressure);
                out.put(row.size);
                out.put(1);
                out.put(row.deviceId);
                out.put(row.source);
                out.put(0);
                touch.put(out);
            }
            data.put("touch", touch);
        }
        return data;
    }

    private static float[][] frames(JSONArray array) throws Exception {
        if (array == null) {
            return new float[0][];
        }
        float[][] out = new float[array.length()][];
        for (int i = 0; i < array.length(); i++) {
            JSONArray row = array.getJSONArray(i);
            if (row.length() != ImuBus.CHANNELS) {
                throw new IllegalArgumentException(
                        "frame " + i + " has " + row.length() + " channels, expected "
                                + ImuBus.CHANNELS);
            }
            float[] values = new float[ImuBus.CHANNELS];
            for (int c = 0; c < ImuBus.CHANNELS; c++) {
                values[c] = (float) row.getDouble(c);
            }
            out[i] = values;
        }
        return out;
    }

    private static double round(double value) {
        return Math.round(value * 1000.0) / 1000.0;
    }
}
