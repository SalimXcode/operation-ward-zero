"""
Operation Ward Zero - Vitals Simulator
Publishes synthetic vitals every 5 s to wardzero/<patient>/vitals (QoS 1).
Cycles: NORMAL (45 s) -> DETERIORATING ramp (15 s) -> CRITICAL (30 s) -> recovery (15 s)

Run:  python simulator.py            (auto cycle)
      python simulator.py critical   (force critical right away, handy for demos)
Press  c  + Enter for critical,  n + Enter for normal while it runs.
"""
import json
import random
import sys
import threading
import time
import uuid

import paho.mqtt.client as mqtt

BROKER, PORT = "broker.hivemq.com", 1883
PATIENT = "bed01"
TOPIC = f"wardzero/{PATIENT}/vitals"
INTERVAL = 5

NORMAL = dict(hr=76, spo2=97.5, sys=118, dia=76, temp=36.8)
CRITICAL = dict(hr=138, spo2=86.5, sys=172, dia=104, temp=39.1)

force = {"mode": None}   # manual override from the keyboard
if len(sys.argv) > 1 and sys.argv[1].lower().startswith("crit"):
    force["mode"] = "critical"

def lerp(a, b, t):
    return {k: a[k] + (b[k] - a[k]) * t for k in a}

def phase(elapsed):
    """Return (label, blend 0..1 toward CRITICAL) for the auto cycle."""
    t = elapsed % 105
    if t < 45:  return "NORMAL", 0.0
    if t < 60:  return "DETERIORATING", (t - 45) / 15
    if t < 90:  return "CRITICAL", 1.0
    return "RECOVERING", 1 - (t - 90) / 15

def reading(blend):
    base = lerp(NORMAL, CRITICAL, blend)
    return {
        "hr": round(base["hr"] + random.gauss(0, 2 + 3 * blend)),
        "spo2": round(min(100, base["spo2"] + random.gauss(0, 0.4 + 0.6 * blend)), 1),
        "sys": round(base["sys"] + random.gauss(0, 3 + 4 * blend)),
        "dia": round(base["dia"] + random.gauss(0, 2 + 3 * blend)),
        "temp": round(base["temp"] + random.gauss(0, 0.1), 1),
        "ts": time.time(),
    }

def keyboard():
    for line in sys.stdin:
        c = line.strip().lower()
        force["mode"] = {"c": "critical", "n": "normal"}.get(c, None)
        print(f">> manual mode: {force['mode'] or 'auto'}")

def main():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"wardzero-sim-{uuid.uuid4().hex[:6]}")
    except AttributeError:
        client = mqtt.Client(client_id=f"wardzero-sim-{uuid.uuid4().hex[:6]}")
    client.reconnect_delay_set(1, 10)
    client.connect(BROKER, PORT, keepalive=30)
    client.loop_start()
    threading.Thread(target=keyboard, daemon=True).start()
    start = time.time()
    print(f"Publishing to {TOPIC} every {INTERVAL}s  (c=critical, n=normal, Ctrl+C to stop)")
    try:
        while True:
            if force["mode"] == "critical": label, blend = "CRITICAL (manual)", 1.0
            elif force["mode"] == "normal": label, blend = "NORMAL (manual)", 0.0
            else: label, blend = phase(time.time() - start)
            data = reading(blend)
            info = client.publish(TOPIC, json.dumps(data), qos=1)
            print(f"[{label:<18}] HR {data['hr']:>3}  SpO2 {data['spo2']:>5}  BP {data['sys']}/{data['dia']}  T {data['temp']}  (rc={info.rc})")
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
