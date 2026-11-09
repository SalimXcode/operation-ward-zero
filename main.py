"""
Operation Ward Zero - Backend (FastAPI)
Sense -> Understand -> Predict -> Alert -> Respond
"""
import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import numpy as np
import paho.mqtt.client as mqtt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wardzero")

# ----------------------------- CONFIG ---------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ward.env"))
except ImportError:
    pass

MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = "wardzero/+/vitals"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

RISK_THRESHOLD = 0.6
ALERT_COOLDOWN_S = 20
DB_TIMEOUT_S = 5
BUFFER_MAX = 5000

# ----------------------------- STATE ----------------------------------------
class State:
    loop: asyncio.AbstractEventLoop = None
    mqtt_connected = False
    db_connected = False
    db_client = None
    last_error = ""
    history = deque(maxlen=120)
    alerts = {}
    last_alert_ts = {}
    retry_buffer = deque(maxlen=BUFFER_MAX)
    current_notes = "" # Caregiver Notes store karne ke liye

S = State()

# ----------------------------- ML MODEL -------------------------------------
def _components(hr, spo2, sys_, temp):
    return np.array([
        np.maximum((hr - 100) / 40, (55 - hr) / 25),
        (95 - spo2) / 8,
        np.maximum((sys_ - 140) / 40, (95 - sys_) / 25),
        np.maximum((temp - 37.8) / 1.5, (36.0 - temp) / 1.5),
    ]).clip(min=0)

def train_model() -> RandomForestRegressor:
    rng = np.random.default_rng(42)
    n = 6000
    hr = rng.normal(80, 25, n).clip(35, 190)
    spo2 = rng.normal(95, 4.5, n).clip(70, 100)
    sys_ = rng.normal(120, 28, n).clip(60, 220)
    dia = (sys_ * 0.65 + rng.normal(0, 6, n)).clip(35, 130)
    temp = rng.normal(37.0, 0.9, n).clip(34, 41.5)
    c = _components(hr, spo2, sys_, temp)
    y = (0.6 * c.max(axis=0) + 0.4 * c.sum(axis=0) + rng.normal(0, 0.02, n)).clip(0, 1)
    X = np.column_stack([hr, spo2, sys_, dia, temp])
    model = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=1)
    model.fit(X, y)
    log.info("Risk model trained on %d synthetic samples", n)
    return model

MODEL = train_model()

def predict_risk(v: dict) -> float:
    try:
        x = np.array([[v["hr"], v["spo2"], v["sys"], v["dia"], v["temp"]]], dtype=float)
        return float(np.clip(MODEL.predict(x)[0], 0, 1))
    except Exception as e:
        log.warning("predict failed: %s", e)
        return 0.0

def explain(v: dict) -> list:
    r = []
    if v["hr"] > 100: r.append({"code": "tachy", "value": round(v["hr"])})
    if v["hr"] < 55: r.append({"code": "brady", "value": round(v["hr"])})
    if v["spo2"] < 90: r.append({"code": "spo2_crit", "value": round(v["spo2"], 1)})
    elif v["spo2"] < 95: r.append({"code": "spo2_low", "value": round(v["spo2"], 1)})
    if v["sys"] > 140: r.append({"code": "bp_high", "value": f'{round(v["sys"])}/{round(v["dia"])}'})
    if v["sys"] < 95: r.append({"code": "bp_low", "value": f'{round(v["sys"])}/{round(v["dia"])}'})
    if v["temp"] > 37.8: r.append({"code": "fever", "value": round(v["temp"], 1)})
    if v["temp"] < 36.0: r.append({"code": "hypo", "value": round(v["temp"], 1)})
    return r

# ----------------------------- WEBSOCKETS -----------------------------------
class Manager:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def drop(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await asyncio.wait_for(ws.send_text(json.dumps(msg)), timeout=3)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.drop(ws)

manager = Manager()

def status_msg() -> dict:
    return {"type": "status", "mqtt": S.mqtt_connected, "db": S.db_connected,
            "buffer": len(S.retry_buffer), "db_error": S.last_error}

# ----------------------------- DATABASE + RETRY BUFFER ----------------------
def note_db_error(e: Exception):
    msg = f"{type(e).__name__}: {e}"[:400]
    if msg != S.last_error:
        log.error("DB error -> %s", msg)
        S.last_error = msg

def _init_db():
    if not (SUPABASE_URL and SUPABASE_KEY):
        log.warning("Supabase not configured - running with in-memory buffer only")
        return
    try:
        from supabase import create_client
        S.db_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        note_db_error(e)

def _db_write(table: str, row: dict, op: str = "insert"):
    if S.db_client is None:
        raise ConnectionError("no db client")
    if op == "insert":
        S.db_client.table(table).insert(row).execute()
    else:
        S.db_client.table(table).update(row["set"]).eq("id", row["id"]).execute()

async def persist(table: str, row: dict, op: str = "insert"):
    try:
        await asyncio.wait_for(asyncio.to_thread(_db_write, table, row, op), DB_TIMEOUT_S)
        S.last_error = ""
        if not S.db_connected:
            S.db_connected = True
            await manager.broadcast(status_msg())
    except Exception as e:
        note_db_error(e)
        S.retry_buffer.append((table, row, op))
        if S.db_connected:
            S.db_connected = False
        await manager.broadcast(status_msg())

async def flush_loop():
    while True:
        try:
            await asyncio.sleep(5)
            if S.db_client is None: continue
            if not S.retry_buffer:
                if not S.db_connected:
                    try:
                        await asyncio.wait_for(asyncio.to_thread(
                            lambda: S.db_client.table("vitals").select("id").limit(1).execute()), DB_TIMEOUT_S)
                        S.db_connected = True
                        await manager.broadcast(status_msg())
                    except Exception: pass
                continue
            flushed = 0
            while S.retry_buffer:
                table, row, op = S.retry_buffer[0]
                try:
                    await asyncio.wait_for(asyncio.to_thread(_db_write, table, row, op), DB_TIMEOUT_S)
                    S.retry_buffer.popleft()
                    flushed += 1
                except Exception as e:
                    note_db_error(e)
                    break
            if flushed:
                log.info("Flushed %d buffered rows", flushed)
                S.db_connected = not S.retry_buffer
                await manager.broadcast(status_msg())
        except Exception as e:
            log.error("flush_loop error: %s", e)

# ----------------------------- GROQ AI ------------------------
def _groq_opts(max_out: int) -> dict:
    o = {"model": GROQ_MODEL, "temperature": 0.4, "max_completion_tokens": max_out + 700}
    if "gpt-oss" in GROQ_MODEL:
        o["reasoning_effort"] = "low"
    return o

def _groq_err(e: Exception) -> str:
    r = getattr(e, "response", None)
    return (r.text[:300] if r is not None else str(e)[:200])

def fallback_advice(reasons: list) -> str:
    codes = {r["code"] for r in reasons}
    tips = []
    if codes & {"spo2_low", "spo2_crit"}: tips.append("Help the patient sit upright, check the finger sensor, and encourage slow deep breaths.")
    if codes & {"tachy", "brady", "bp_high", "bp_low"}: tips.append("Keep the patient calm and resting; re-check pulse and blood pressure in 5 minutes.")
    if codes & {"fever", "hypo"}: tips.append("Adjust blankets and room temperature and offer fluids if the patient is awake.")
    tips.append("If symptoms persist or worsen, contact the doctor or emergency services immediately.")
    return "\n".join(f"- {t}" for t in tips[:2])

async def groq_advice(v: dict, risk: float, reasons: list) -> str:
    if not GROQ_API_KEY: return fallback_advice(reasons)
    
    # NAYA: Caregiver notes ko AI prompt me bhejo
    notes_context = f"Caregiver Notes: {S.current_notes}." if S.current_notes else "No caregiver notes."
    
    prompt = (f"Home patient vitals: HR {v['hr']} bpm, SpO2 {v['spo2']}%, BP {v['sys']}/{v['dia']} mmHg, "
              f"Temp {v['temp']} C. AI risk score {risk:.2f}. Flags: {[r['code'] for r in reasons]}. {notes_context} "
              "Give EXACTLY 2 short bullet points (start each with '- ') of highly actionable, "
              "NON-diagnostic advice for the family caregiver. No diagnosis, no medication doses.")
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                             headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                             json={**_groq_opts(300),
                                   "messages": [{"role": "system", "content": "You are a cautious home-care assistant."},
                                                {"role": "user", "content": prompt}]})
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
            return text or fallback_advice(reasons)
    except Exception as e:
        log.warning("Groq failed: %s %s", type(e).__name__, _groq_err(e))
        return fallback_advice(reasons)

# ----------------------------- AI CHAT ASSISTANT ----------------------------
def _vitals_text(v) -> str:
    if not v: return "No sensor data received yet."
    return (f"HR {v['hr']:.0f} bpm, SpO2 {v['spo2']}%, BP {v['sys']:.0f}/{v['dia']:.0f} mmHg, "
            f"Temp {v['temp']} C, AI risk score {v['risk']:.2f}")

def fallback_chat(v, reasons) -> str:
    if not v: return "I have not received any sensor data yet. Please check that the simulator or device is running."
    lines = [f"Current readings: {_vitals_text(v)}."]
    if reasons:
        lines.append("Flags: " + ", ".join(r["code"].replace("_", " ") for r in reasons) + ".")
        lines.append(fallback_advice(reasons))
    else:
        lines.append("All vitals are in the safe range. Keep monitoring.")
    return "\n".join(lines)

async def chat_reply(text: str, history: list, lang: str) -> str:
    v = S.history[-1] if S.history else None
    reasons = v.get("reasons", []) if v else []
    if not GROQ_API_KEY:
        return fallback_chat(v, reasons) + "\n[GROQ_API_KEY not loaded]"
    langname = {"hi": "Hindi", "hinglish": "Hinglish"}.get(lang, "English")
    
    notes_context = f"Caregiver notes: {S.current_notes}." if S.current_notes else ""
    
    system = ("You are Ward Zero AI, a calm home-healthcare assistant for a family caregiver. "
              "Never diagnose and never give medication doses. Be concise (max 80 words). "
              f"Reply in {langname}. Live patient data: {_vitals_text(v)}. {notes_context} "
              f"Active flags: {[r['code'] for r in reasons]}. Unacknowledged alerts: "
              f"{sum(1 for a in S.alerts.values() if not a['acknowledged'])}.")
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                             headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                             json={**_groq_opts(350),
                                   "messages": [{"role": "system", "content": system}] + history[-8:]
                                               + [{"role": "user", "content": text}]})
            r.raise_for_status()
            out = (r.json()["choices"][0]["message"]["content"] or "").strip()
            return out or fallback_chat(v, reasons)
    except Exception as e:
        log.warning("Groq chat failed: %s", e)
        return fallback_chat(v, reasons) + "\n[Groq error]"

async def handle_chat(ws: WebSocket, history: list, text: str, lang: str):
    try:
        reply = await chat_reply(text[:500], history, lang)
        history += [{"role": "user", "content": text[:500]}, {"role": "assistant", "content": reply}]
        del history[:-12]
        await asyncio.wait_for(ws.send_text(json.dumps({"type": "chat", "text": reply})), timeout=3)
    except Exception as e:
        log.info("chat send failed: %s", e)

# ----------------------------- PIPELINE -------------------------------------
async def create_alert(pid: str, v: dict, risk: float, reasons: list):
    try:
        advice = await groq_advice(v, risk, reasons)
        # NAYA: Response time aur Caregiver notes backend me save karo
        alert = {"id": str(uuid.uuid4()), "patient_id": pid, "risk": round(risk, 3),
                 "reasons": reasons, "advice": advice, "acknowledged": False,
                 "caregiver_notes": S.current_notes, "ts": datetime.now(timezone.utc).isoformat()}
        S.alerts[alert["id"]] = alert
        await manager.broadcast({"type": "alert", "data": alert})
        await persist("alerts", alert)
    except Exception as e:
        log.error("create_alert error: %s", e)

async def process(topic: str, payload: bytes):
    try:
        d = json.loads(payload)
        pid = topic.split("/")[1]
        v = {k: float(d[k]) for k in ("hr", "spo2", "sys", "dia", "temp")}
        risk = predict_risk(v)
        reasons = explain(v)
        ts = datetime.now(timezone.utc).isoformat()
        rec = {"patient_id": pid, **v, "risk": round(risk, 3), "ts": ts}
        S.history.append({**rec, "reasons": reasons})
        await manager.broadcast({"type": "vitals", "data": rec, "reasons": reasons})
        asyncio.create_task(persist("vitals", rec))
        
        now = time.time()
        if risk > RISK_THRESHOLD and now - S.last_alert_ts.get(pid, 0) > ALERT_COOLDOWN_S:
            S.last_alert_ts[pid] = now
            asyncio.create_task(create_alert(pid, v, risk, reasons))
    except Exception as e:
        log.warning("Bad MQTT message on %s: %s", topic, e)

# ----------------------------- MQTT -----------------------------------------
def _on_connect(client, userdata, *args):
    S.mqtt_connected = True
    client.subscribe(MQTT_TOPIC, qos=1)
    log.info("MQTT connected, subscribed to %s", MQTT_TOPIC)
    asyncio.run_coroutine_threadsafe(manager.broadcast(status_msg()), S.loop)

def _on_disconnect(client, userdata, *args):
    S.mqtt_connected = False
    log.warning("MQTT disconnected")
    asyncio.run_coroutine_threadsafe(manager.broadcast(status_msg()), S.loop)

def _on_message(client, userdata, msg):
    asyncio.run_coroutine_threadsafe(process(msg.topic, msg.payload), S.loop)

def start_mqtt() -> mqtt.Client:
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"wardzero-api-{uuid.uuid4().hex[:6]}")
    except AttributeError:
        client = mqtt.Client(client_id=f"wardzero-api-{uuid.uuid4().hex[:6]}")
    client.on_connect, client.on_disconnect, client.on_message = _on_connect, _on_disconnect, _on_message
    client.reconnect_delay_set(1, 10)
    try:
        client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=30)
        client.loop_start()
    except Exception as e:
        log.error("MQTT start failed: %s", e)
    return client

# ----------------------------- APP ------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    S.loop = asyncio.get_running_loop()
    _init_db()
    S.db_connected = S.db_client is not None
    client = start_mqtt()
    task = asyncio.create_task(flush_loop())
    yield
    task.cancel()
    try:
        client.loop_stop(); client.disconnect()
    except Exception:
        pass

app = FastAPI(title="Operation Ward Zero", lifespan=lifespan)

@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.get("/health")
async def health():
    return status_msg()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "snapshot", "history": list(S.history)[-60:],
                                       "alerts": sorted(S.alerts.values(), key=lambda a: a["ts"], reverse=True)[:30]}))
        await ws.send_text(json.dumps(status_msg()))
        chat_hist: list = []
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "ack" and msg.get("id") in S.alerts:
                a = S.alerts[msg["id"]]
                a["acknowledged"] = True
                
                # NAYA: Response time calculate karo
                sent_time = datetime.fromisoformat(a["ts"].replace('Z', '+00:00'))
                ack_time = datetime.now(timezone.utc)
                response_sec = int((ack_time - sent_time).total_seconds())
                a["response_time"] = response_sec
                
                await manager.broadcast({"type": "ack", "id": a["id"], "response_time": response_sec})
                
                asyncio.create_task(persist("alerts", {
                    "id": a["id"], 
                    "set": {"acknowledged": True, "response_time": response_sec}
                }, "update"))
                
            elif msg.get("type") == "note":
                # NAYA: Caregiver note receive karo
                S.current_notes = str(msg.get("text", ""))[:200]
                log.info(f"Caregiver Note: {S.current_notes}")
                
            elif msg.get("type") == "chat" and str(msg.get("text", "")).strip():
                asyncio.create_task(handle_chat(ws, chat_hist, str(msg["text"]), str(msg.get("lang", "en"))))
            elif msg.get("type") == "ping":
                await ws.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.info("WS closed: %s", e)
    finally:
        manager.drop(ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)