// Configuration two, the inertial half.
//
// WHY THIS HOOK IS NATIVE
//
// Replacing events below ART keeps the delivery path independent of Java
// instrumentation inside the target. The hook attaches to the platform sensor
// queue and edits each batch after it is read.
//
// WHERE
//
//     android::SensorEventQueue::read(ASensorEvent* events, size_t numEvents)
//
// is the single point every delivery passes through: the queue reads a batch
// out of the kernel socket into the caller's array and returns how many it
// filled. Everything above it -- the JNI receiver, dispatchSensorEvent,
// onSensorChanged -- is downstream of that buffer. Rewriting it on the way out
// changes what the app receives without ART being involved at all.
//
// WHAT IS AND IS NOT REPRODUCED:
//   - values are ours exactly
//   - the rate, the batching and the delivery instants stay the device's,
//     because this edits a read that already happened and cannot invent one
//   - timestamps are never rewritten; they are the query, not the output

'use strict';

var PLAN_PATH = '/data/local/tmp/actreal/imu_plan.json';
var PLAN_SCHEMA = 'actreal_imu_plan_v1';

var TYPE_ACCELEROMETER = 1;
var TYPE_GYROSCOPE = 4;

// ASensorEvent, arm64. Confirmed against the event's own `version` field at
// runtime rather than assumed: version is defined as sizeof(ASensorEvent), so
// a mismatch is detectable instead of silently shifting every field.
var OFF_VERSION = 0;
var OFF_SENSOR = 4;      // handle
var OFF_TYPE = 8;
var OFF_TIMESTAMP = 16;  // int64, elapsedRealtimeNanos
var OFF_DATA = 24;       // float[16]; data[0..2] is the vector
var EVENT_SIZE = 104;

var plan = null;
var mode = 'real';
var sizeChecked = false;

var stats = {
    reads: 0,
    events: 0,
    replaced: 0,
    passed: 0,
    from_window: 0,
    from_background: 0,
    no_frame: 0,
    size_mismatch: 0,
    first_elapsed_ns: 0,
    last_elapsed_ns: 0
};
var counters = {};

function log(message) { console.log('[actreal-native] ' + message); }

function toNum(value) {
    if (value === null || value === undefined) { return 0; }
    if (typeof value === 'number') { return value; }
    if (typeof value.toNumber === 'function') { return value.toNumber(); }
    return parseInt(value.toString(), 10);
}

// ---------------------------------------------------------------------------
// The replacement plan.
// ---------------------------------------------------------------------------

function installPlan(parsed) {
    if (parsed.schema_version !== PLAN_SCHEMA) {
        throw new Error('unknown plan schema ' + parsed.schema_version);
    }
    var periodNs = Math.round(parsed.period_ms * 1e6);
    if (!(periodNs > 0)) { throw new Error('plan period must be positive'); }
    var windows = (parsed.windows || []).map(function (w) {
        var frames = w.frames || [];
        return {
            bundle_id: w.bundle_id || '',
            start_elapsed_ns: toNum(w.start_elapsed_ns),
            end_elapsed_ns: toNum(w.start_elapsed_ns) + frames.length * periodNs,
            frames: frames
        };
    });
    windows.sort(function (a, b) { return a.start_elapsed_ns - b.start_elapsed_ns; });
    for (var i = 1; i < windows.length; i++) {
        if (windows[i].start_elapsed_ns < windows[i - 1].end_elapsed_ns) {
            throw new Error('windows ' + windows[i - 1].bundle_id + ' and ' +
                            windows[i].bundle_id + ' overlap');
        }
    }
    plan = {
        period_ns: periodNs,
        background: parsed.background || [],
        background_start_elapsed_ns: toNum(parsed.background_start_elapsed_ns),
        windows: windows
    };
    log('plan: ' + windows.length + ' window(s), ' + plan.background.length +
        ' background frames @ ' + parsed.period_ms + ' ms');
    return status();
}

function loadPlanFile() {
    var text = null;
    if (typeof File.readAllText === 'function') {
        text = File.readAllText(PLAN_PATH);
    } else {
        var f = new File(PLAN_PATH, 'r');
        text = f.readText();
        f.close();
    }
    return installPlan(JSON.parse(text));
}

function frameAt(elapsedNs) {
    if (plan === null) { return null; }
    var windows = plan.windows;
    for (var i = 0; i < windows.length; i++) {
        var w = windows[i];
        if (elapsedNs < w.start_elapsed_ns) { break; }
        if (elapsedNs < w.end_elapsed_ns) {
            var index = Math.floor((elapsedNs - w.start_elapsed_ns) / plan.period_ns);
            if (index < 0) { index = 0; }
            if (index >= w.frames.length) { index = w.frames.length - 1; }
            return { frame: w.frames[index], origin: 'window' };
        }
    }
    var n = plan.background.length;
    if (n === 0) { return null; }
    var span = n * plan.period_ns;
    var delta = (elapsedNs - plan.background_start_elapsed_ns) % span;
    if (delta < 0) { delta += span; }
    return { frame: plan.background[Math.floor(delta / plan.period_ns) % n],
             origin: 'background' };
}

// ---------------------------------------------------------------------------
// The hook
// ---------------------------------------------------------------------------

var READ_SYMBOL = '_ZN7android16SensorEventQueue4readEP12ASensorEventm';

function attach() {
    var module = Process.getModuleByName('libsensor.so');
    var target = null;
    module.enumerateExports().forEach(function (e) {
        if (e.name === READ_SYMBOL) { target = e.address; }
    });
    if (target === null) {
        throw new Error('SensorEventQueue::read not exported by libsensor.so');
    }

    Interceptor.attach(target, {
        onEnter: function (args) {
            // args[0] is `this`; the buffer is the first real argument.
            this.events = args[1];
        },
        onLeave: function (retval) {
            // read() returns how many events it filled, or negative on error.
            // Trusting the caller's numEvents instead would rewrite whatever
            // stale bytes happened to be past the end of the batch.
            var count = retval.toInt32();
            stats.reads += 1;
            if (count <= 0 || this.events === null || mode !== 'injected' || plan === null) {
                if (count > 0) { stats.passed += count; stats.events += count; }
                return;
            }

            for (var i = 0; i < count; i++) {
                var event = this.events.add(i * EVENT_SIZE);
                stats.events += 1;

                if (!sizeChecked) {
                    // `version` is defined as sizeof(ASensorEvent). If the
                    // struct on this build is not the 104 bytes assumed, every
                    // field after the first is being read from the wrong place
                    // and the values written would land in another event -- so
                    // it is checked once and injection stops rather than
                    // corrupting the stream silently.
                    var version = event.add(OFF_VERSION).readS32();
                    sizeChecked = true;
                    if (version !== EVENT_SIZE) {
                        stats.size_mismatch = version;
                        log('ASensorEvent is ' + version + ' bytes, not ' +
                            EVENT_SIZE + '; refusing to rewrite');
                        mode = 'real';
                        return;
                    }
                }

                var type = event.add(OFF_TYPE).readS32();
                if (type !== TYPE_ACCELEROMETER && type !== TYPE_GYROSCOPE) {
                    stats.passed += 1;
                    continue;
                }

                var ts = event.add(OFF_TIMESTAMP).readS64().toNumber();
                if (stats.first_elapsed_ns === 0) { stats.first_elapsed_ns = ts; }
                stats.last_elapsed_ns = ts;
                counters[type] = (counters[type] || 0) + 1;

                var hit = frameAt(ts);
                if (hit === null) {
                    stats.no_frame += 1;
                    stats.passed += 1;
                    continue;
                }

                // The plan carries accelerometer and gyroscope in one six-wide
                // frame; which half belongs to this event is decided by its own
                // type field, which is why no handle-to-type resolution is
                // needed down here.
                var base = (type === TYPE_ACCELEROMETER) ? 0 : 3;
                var data = event.add(OFF_DATA);
                for (var k = 0; k < 3; k++) {
                    data.add(k * 4).writeFloat(hit.frame[base + k]);
                }
                stats.replaced += 1;
                if (hit.origin === 'window') { stats.from_window += 1; }
                else { stats.from_background += 1; }
            }
        }
    });
    log('hooked SensorEventQueue::read at ' + target);
}

// ---------------------------------------------------------------------------
// Clock -- read natively from the same process as the sensor queue.
// ---------------------------------------------------------------------------

function clock() {
    // CLOCK_BOOTTIME is what SystemClock.elapsedRealtimeNanos returns and what
    // sensor timestamps use; CLOCK_MONOTONIC is uptimeMillis, which
    // MotionEvent uses. Reading both here relates the two Android clocks at the
    // point where the replacement plan is installed.
    var clock_gettime = new NativeFunction(
        Module.getGlobalExportByName('clock_gettime'), 'int', ['int', 'pointer']);
    var CLOCK_MONOTONIC = 1, CLOCK_BOOTTIME = 7;
    var ts = Memory.alloc(16);

    clock_gettime(CLOCK_BOOTTIME, ts);
    var before = ts.readS64().toNumber() * 1e9 + ts.add(8).readS64().toNumber();
    clock_gettime(CLOCK_MONOTONIC, ts);
    var uptimeNs = ts.readS64().toNumber() * 1e9 + ts.add(8).readS64().toNumber();
    clock_gettime(CLOCK_BOOTTIME, ts);
    var after = ts.readS64().toNumber() * 1e9 + ts.add(8).readS64().toNumber();

    return {
        uptime_ms: Math.round(uptimeNs / 1e6),
        elapsed_ns: Math.round((before + after) / 2),
        read_window_ns: after - before
    };
}

function status() {
    var rates = {};
    var span_s = (stats.last_elapsed_ns - stats.first_elapsed_ns) / 1e9;
    Object.keys(counters).forEach(function (type) {
        rates[type] = {
            events: counters[type],
            observed_hz: span_s > 0 ? Math.round(counters[type] / span_s * 100) / 100 : 0
        };
    });
    return {
        backend: 'native',
        mode: mode,
        plan_loaded: plan !== null,
        windows: plan ? plan.windows.length : 0,
        background_frames: plan ? plan.background.length : 0,
        period_ms: plan ? plan.period_ns / 1e6 : 0,
        stats: stats,
        rates: rates
    };
}

var startup = { attempted: true, attached: false, error: '' };
try {
    try { loadPlanFile(); } catch (e) { log('no plan yet (' + e + ')'); }
    attach();
    startup.attached = true;
} catch (e) {
    startup.error = String(e && e.message ? e.message : e);
    log('install failed: ' + startup.error);
}

rpc.exports = {
    diagnose: function () {
        return {
            backend: 'native',
            startup: startup,
            frida: Frida.version,
            runtime: Script.runtime,
            // Deliberately never touches Java: the whole point of this file is
            // that the bridge aborts this process.
            uses_java_bridge: false,
            verdict: startup.attached
                ? 'native hook installed on SensorEventQueue::read'
                : 'not installed: ' + startup.error
        };
    },
    setMode: function (m) {
        if (m !== 'real' && m !== 'injected') { throw new Error('bad mode ' + m); }
        mode = m;
        log('mode=' + m);
        return status();
    },
    loadPlan: function (text) { return installPlan(JSON.parse(text)); },
    reload: function () { return loadPlanFile(); },
    clock: function () { return clock(); },
    status: function () { return status(); },
    resetStats: function () {
        Object.keys(stats).forEach(function (k) { stats[k] = 0; });
        counters = {};
        sizeChecked = false;
        return status();
    }
};
