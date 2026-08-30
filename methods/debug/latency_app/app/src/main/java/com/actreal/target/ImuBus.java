package com.actreal.target;

import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The single point where inertial data reaches the app.
 *
 * <p>Everything downstream of this bus is written the way an ordinary app
 * reads sensors, and cannot tell where a sample came from -- which is the
 * point. Exactly one producer is attached at a time: either the real
 * SensorManager or the injected stream. Running both would splice two
 * different hands together and the seam would be the strongest signal in the
 * recording.
 */
public final class ImuBus {

    public interface Listener {
        void onImuSample(ImuSample sample);
    }

    public static final int MODE_REAL = 0;
    public static final int MODE_INJECTED = 1;

    private final CopyOnWriteArrayList<Listener> listeners = new CopyOnWriteArrayList<>();
    private final AtomicLong delivered = new AtomicLong();
    private volatile int mode = MODE_REAL;

    public void addListener(Listener listener) {
        listeners.add(listener);
    }

    public void removeListener(Listener listener) {
        listeners.remove(listener);
    }

    public int mode() {
        return mode;
    }

    public void setMode(int value) {
        mode = value;
    }

    public long deliveredCount() {
        return delivered.get();
    }

    /**
     * Hand a sample to the app.
     *
     * <p>Samples whose origin does not match the current mode are dropped
     * rather than delivered late: a real sample arriving during an injected
     * action is exactly the contamination this bus exists to prevent.
     */
    public void publish(ImuSample sample) {
        boolean wanted =
                (mode == MODE_INJECTED)
                        ? ImuSample.ORIGIN_INJECTED.equals(sample.origin)
                        : ImuSample.ORIGIN_REAL.equals(sample.origin);
        if (!wanted) {
            return;
        }
        delivered.incrementAndGet();
        for (Listener listener : listeners) {
            listener.onImuSample(sample);
        }
    }
}
