// Measure how late an injected touch actually arrives, without asking the app.
//
// WHY
//
// A gesture's two halves are scheduled against one instant. The inertial half
// lands on it exactly, because that hook is passive: sensor events arrive
// carrying their own timestamps and it only decides which frame belongs at each
// one. The touch half is active -- written to uinput from the host, then
// forwarded, then scheduled -- so it arrives some delay later. Configuration
// one measured that delay by asking the target app when the contact landed,
// through a control channel added to it. Configuration two removed the channel
// on purpose, and with it the measurement: the correction has been zero ever
// since, and every touch trails its own inertia by roughly 6ms.
//
// This gets the number back without putting anything into the app. It reads
// the arrival instant from inside the process, the same way the inertial hook
// reads the clock, and reports it out over RPC.
//
// WHERE
//
//     android::MotionEvent::initialize(...)
//
// Every touch that reaches the application is constructed through it. Hooking
// it is read-only: nothing is modified, no argument is rewritten, no return
// value is touched. The gesture that arrives is exactly the gesture that was
// sent -- which matters, because the whole point of the project is that the
// signal is the victim's and not ours to reshape.
//
// WHAT COMES OUT
//
// A list of arrival instants on CLOCK_MONOTONIC, the clock MotionEvent uses.
// The host knows when it intended each contact to land; the difference is the
// delay, and its median is what the scheduler should subtract next time.

'use strict';

var SYMBOL = '_ZN7android11MotionEvent10initializeEiij';

var arrivals = [];       // uptime nanoseconds, newest last
var limit = 4096;
var installed = false;
var error = '';

function log(message) { console.log('[actreal-touch] ' + message); }

// CLOCK_MONOTONIC is what uptimeMillis and MotionEvent's own timestamps run on.
// Reading it here rather than converting from something else keeps the arrival
// instant on the same clock as the intended instant the host holds.
var clock_gettime = null;
var scratch = null;

function nowNs() {
    if (clock_gettime === null) {
        clock_gettime = new NativeFunction(
            Module.getGlobalExportByName('clock_gettime'), 'int', ['int', 'pointer']);
        scratch = Memory.alloc(16);
    }
    clock_gettime(1, scratch);   // CLOCK_MONOTONIC
    return scratch.readS64().toNumber() * 1e9 + scratch.add(8).readS64().toNumber();
}

function attach() {
    var module = Process.getModuleByName('libinput.so');
    var target = null;
    module.enumerateExports().forEach(function (e) {
        if (target === null && e.name.indexOf(SYMBOL) === 0) { target = e.address; }
    });
    if (target === null) {
        throw new Error('MotionEvent::initialize not exported by libinput.so');
    }
    Interceptor.attach(target, {
        onEnter: function () {
            // Deliberately nothing but a timestamp. Reading the arguments would
            // mean decoding a C++ signature that differs across builds, and the
            // question here is *when*, not what.
            arrivals.push(nowNs());
            if (arrivals.length > limit) { arrivals.shift(); }
        }
    });
    log('watching MotionEvent::initialize at ' + target);
}

try {
    attach();
    installed = true;
} catch (e) {
    error = String(e && e.message ? e.message : e);
    log('install failed: ' + error);
}

rpc.exports = {
    diagnose: function () {
        return { backend: 'touch-probe', installed: installed, error: error,
                 seen: arrivals.length };
    },
    // Everything since the last call, and clear. Polling between actions keeps
    // each gesture's arrivals separable without timestamping on the host side.
    drain: function () {
        var out = arrivals;
        arrivals = [];
        return out;
    },
    peek: function () { return arrivals.length; },
    clock: function () { return { uptime_ns: nowNs() }; }
};
