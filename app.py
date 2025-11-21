import os
import requests
from flask import Flask, request, jsonify
from openai import OpenAI

# -------- CONFIG --------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

OPENF1_BASE = "https://api.openf1.org/v1"

app = Flask(__name__)

# -------- HELPERS: OPENF1 --------

def get_race_session(year: int, country_name: str):
    params = {
        "year": year,
        "country_name": country_name,
        "session_type": "Race"
    }
    resp = requests.get(f"{OPENF1_BASE}/sessions", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError("No race session found. Check year/country spelling.")
    data.sort(key=lambda x: x["date_end"])
    return data[-1]


def get_stints(session_key: int, driver_number: int):
    params = {"session_key": session_key, "driver_number": driver_number}
    resp = requests.get(f"{OPENF1_BASE}/stints", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_weather(meeting_key: int):
    params = {"meeting_key": meeting_key}
    resp = requests.get(f"{OPENF1_BASE}/weather", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return data[len(data) // 2]


def get_driver_info(session_key: int, driver_number: int):
    params = {"session_key": session_key, "driver_number": driver_number}
    resp = requests.get(f"{OPENF1_BASE}/drivers", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return data[0]


def summarise_stints(stints, pit_lap: int):
    if not stints:
        return "Stints: no data found.", "unknown", "unknown"

    lines = []
    before_comp = "unknown"
    after_comp = "unknown"

    stints_sorted = sorted(stints, key=lambda s: s["stint_number"])
    for stint in stints_sorted:
        comp = stint["compound"]
        ls = stint["lap_start"]
        le = stint["lap_end"]
        lines.append(f"- Stint {stint['stint_number']}: {comp} (laps {ls}–{le})")

        if ls <= pit_lap <= le:
            before_comp = comp
        if ls == pit_lap + 1:
            after_comp = comp

    return "\n".join(lines), before_comp, after_comp

# -------- RACEMETA PROMPT --------

RACEMETA_SYSTEM_PROMPT = """
You are RaceMeta, an F1 pit-wall meta strategist that gives sharp, Twitter-ready verdicts on race strategy calls.

Output format (ALWAYS keep it tight, no extra sentences):

<verdict_line>
- Bullet 1 (max ~12 words)
- Bullet 2
- Bullet 3
Alt: One-line alternative call.
Take: One-line general lesson, punchy, like a tweet.

Rules:
- Verdict line: 2–5 words, strong opinion (Optimal / Suboptimal / Mixed, etc.).
- Each bullet MUST be short, one idea, no comma chains if possible.
- No hashtags, no emojis.
- Whole answer should be ~240 characters, hard cap 280-ish.
- If data is missing, lean on typical F1 strategy logic but stay grounded.
"""

def build_context_block(session, driver_info, weather, stint_text,
                        before_comp, after_comp, pit_lap, user_question):
    track_name = session.get("circuit_short_name") or session.get("location") or "Unknown circuit"
    country = session.get("country_name", "Unknown country")
    year = session.get("year", "Unknown year")

    driver_name = driver_info.get("full_name") if driver_info else "Unknown driver"
    team_name = driver_info.get("team_name") if driver_info else "Unknown team"

    track_temp = None
    wind = None
    if weather:
        track_temp = weather.get("track_temperature")
        wind = weather.get("wind_speed")

    lines = [
        f"Race: {country} {year} ({track_name})",
        f"Driver: {driver_name}",
        f"Car: {team_name}",
        "",
        "Tyre & stint data:",
        stint_text,
        "",
        f"Pit of interest: lap {pit_lap} (from {before_comp} to {after_comp})"
    ]

    if track_temp is not None:
        lines.append(f"Approx track temp: {track_temp:.1f}°C")
    if wind is not None:
        lines.append(f"Wind speed sample: {wind:.1f} m/s")

    lines.append("")
    lines.append(f"User question: {user_question}")

    return "\n".join(lines)


def call_racemeta(context_block: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": RACEMETA_SYSTEM_PROMPT},
            {"role": "user", "content": context_block}
        ],
        temperature=0.4,
        max_completion_tokens=180,
    )
    return completion.choices[0].message.content.strip()

# -------- HTTP ENDPOINT --------

from flask import Flask, request, jsonify  # re-import to be safe

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        year = int(data["year"])
        country = str(data["country"])
        driver_number = int(data["driver_number"])
        pit_lap = int(data["pit_lap"])
        question = str(data["question"])
    except Exception as e:
        return jsonify({"error": f"Invalid payload: {e}"}), 400

    try:
        session = get_race_session(year, country)
        session_key = session["session_key"]
        meeting_key = session["meeting_key"]

        driver_info = get_driver_info(session_key, driver_number)
        stints = get_stints(session_key, driver_number)
        stint_text, before_comp, after_comp = summarise_stints(stints, pit_lap)
        weather = get_weather(meeting_key)

        context_block = build_context_block(
            session=session,
            driver_info=driver_info,
            weather=weather,
            stint_text=stint_text,
            before_comp=before_comp,
            after_comp=after_comp,
            pit_lap=pit_lap,
            user_question=question,
        )

        verdict = call_racemeta(context_block)

        return jsonify({
            "context": context_block,
            "verdict": verdict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return "RaceMeta OpenF1 backend is running.", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
