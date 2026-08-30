package com.actreal.target;

import android.view.MotionEvent;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * What the app actually received, kept so the host can check it against the plan.
 *
 * <p>Touch rows include the historical points Android batches into a single
 * dispatch. Dropping them would silently thin the trajectory: a fast swipe can
 * carry several samples per delivered event, and those samples are the ones
 * that carry its speed.
 */
public final class Recorder implements ImuBus.Listener {

    public static final String TOUCH_HEADER =
            "seq,uptime_ns,elapsed_ns,action,batched,pointer_count,pointer_index,"
                    + "pointer_id,x,y,pressure,size,tool_type,device_id,source,edge_flags";

    public static final String IMU_HEADER =
            "seq,timestamp_ns,ax,ay,az,gx,gy,gz,origin,bundle_id,frame_index";

    private static final class TouchRow {
        long seq;
        long uptimeNs;
        long elapsedNs;
        String action;
        boolean batched;
        int pointerCount;
        int pointerIndex;
        int pointerId;
        float x;
        float y;
        float pressure;
        float size;
        int toolType;
        int deviceId;
        int source;
        int edgeFlags;
    }

    private static final class ImuRow {
        long seq;
        long timestampNs;
        float ax;
        float ay;
        float az;
        float gx;
        float gy;
        float gz;
        String origin;
        String bundleId;
        int frameIndex;
    }

    private final List<TouchRow> touch = new ArrayList<>();
    private final List<ImuRow> imu = new ArrayList<>();
    private final int maxRows;

    private long touchSeq;
    private long imuSeq;
    private long touchDropped;
    private long imuDropped;
    private volatile long clockOffsetNs;

    public Recorder(int maxRows) {
        this.maxRows = maxRows;
        this.clockOffsetNs = Clocks.uptimeToElapsedOffsetNs();
    }

    public void refreshClockOffset() {
        clockOffsetNs = Clocks.uptimeToElapsedOffsetNs();
    }

    public long clockOffsetNs() {
        return clockOffsetNs;
    }

    public synchronized void clear() {
        touch.clear();
        imu.clear();
        touchSeq = 0;
        imuSeq = 0;
        touchDropped = 0;
        imuDropped = 0;
    }

    public synchronized int touchRowCount() {
        return touch.size();
    }

    public synchronized int imuRowCount() {
        return imu.size();
    }

    public synchronized long touchDropped() {
        return touchDropped;
    }

    public synchronized long imuDropped() {
        return imuDropped;
    }

    /** Record one dispatched MotionEvent, batched history included. */
    public void observe(MotionEvent event) {
        int pointers = event.getPointerCount();
        int history = event.getHistorySize();
        String action = MotionEvent.actionToString(event.getActionMasked());
        for (int h = 0; h < history; h++) {
            long uptimeNs = event.getHistoricalEventTimeNanos(h);
            for (int p = 0; p < pointers; p++) {
                add(
                        event,
                        p,
                        uptimeNs,
                        "MOVE_BATCHED",
                        true,
                        event.getHistoricalX(p, h),
                        event.getHistoricalY(p, h),
                        event.getHistoricalPressure(p, h),
                        event.getHistoricalSize(p, h));
            }
        }
        long uptimeNs = event.getEventTimeNanos();
        for (int p = 0; p < pointers; p++) {
            add(
                    event,
                    p,
                    uptimeNs,
                    action,
                    false,
                    event.getX(p),
                    event.getY(p),
                    event.getPressure(p),
                    event.getSize(p));
        }
    }

    private synchronized void add(
            MotionEvent event,
            int pointer,
            long uptimeNs,
            String action,
            boolean batched,
            float x,
            float y,
            float pressure,
            float size) {
        if (touch.size() >= maxRows) {
            touchDropped++;
            return;
        }
        TouchRow row = new TouchRow();
        row.seq = touchSeq++;
        row.uptimeNs = uptimeNs;
        row.elapsedNs = uptimeNs + clockOffsetNs;
        row.action = action;
        row.batched = batched;
        row.pointerCount = event.getPointerCount();
        row.pointerIndex = pointer;
        row.pointerId = event.getPointerId(pointer);
        row.x = x;
        row.y = y;
        row.pressure = pressure;
        row.size = size;
        row.toolType = event.getToolType(pointer);
        row.deviceId = event.getDeviceId();
        row.source = event.getSource();
        row.edgeFlags = event.getEdgeFlags();
        touch.add(row);
    }

    @Override
    public synchronized void onImuSample(ImuSample sample) {
        if (imu.size() >= maxRows) {
            imuDropped++;
            return;
        }
        ImuRow row = new ImuRow();
        row.seq = imuSeq++;
        row.timestampNs = sample.timestampNs;
        row.ax = sample.ax;
        row.ay = sample.ay;
        row.az = sample.az;
        row.gx = sample.gx;
        row.gy = sample.gy;
        row.gz = sample.gz;
        row.origin = sample.origin;
        row.bundleId = sample.bundleId;
        row.frameIndex = sample.frameIndex;
        imu.add(row);
    }

    public synchronized JSONObject dump(boolean includeRows) throws JSONException {
        JSONObject out = new JSONObject();
        out.put("touch_rows", touch.size());
        out.put("imu_rows", imu.size());
        out.put("touch_dropped", touchDropped);
        out.put("imu_dropped", imuDropped);
        out.put("clock_offset_ns", clockOffsetNs);
        if (!includeRows) {
            return out;
        }
        JSONArray touchArray = new JSONArray();
        for (TouchRow row : touch) {
            JSONArray r = new JSONArray();
            r.put(row.seq);
            r.put(row.uptimeNs);
            r.put(row.elapsedNs);
            r.put(row.action);
            r.put(row.batched ? 1 : 0);
            r.put(row.pointerCount);
            r.put(row.pointerIndex);
            r.put(row.pointerId);
            r.put((double) row.x);
            r.put((double) row.y);
            r.put((double) row.pressure);
            r.put((double) row.size);
            r.put(row.toolType);
            r.put(row.deviceId);
            r.put(row.source);
            r.put(row.edgeFlags);
            touchArray.put(r);
        }
        JSONArray imuArray = new JSONArray();
        for (ImuRow row : imu) {
            JSONArray r = new JSONArray();
            r.put(row.seq);
            r.put(row.timestampNs);
            r.put((double) row.ax);
            r.put((double) row.ay);
            r.put((double) row.az);
            r.put((double) row.gx);
            r.put((double) row.gy);
            r.put((double) row.gz);
            r.put(row.origin);
            r.put(row.bundleId);
            r.put(row.frameIndex);
            imuArray.put(r);
        }
        out.put("touch_header", TOUCH_HEADER);
        out.put("imu_header", IMU_HEADER);
        out.put("touch", touchArray);
        out.put("imu", imuArray);
        return out;
    }
}
