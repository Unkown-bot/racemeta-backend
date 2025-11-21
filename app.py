import os
import requests
import re
from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS


# -------- CONFIG --------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

OPENF1_BASE = "https://api.openf1.org/v1"

app = Flask(__name__)
CORS(app)

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

def get_latest_race_session():
    """
    Get the most recent completed Race session from OpenF1.
    Used when the user just asks about 'Lewis' etc without specifying race.
    """
    params = {"session_type": "Race"}
    resp = requests.get(f"{OPENF1_BASE}/sessions", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError("No race sessions found in OpenF1.")
    # sort by end date and take latest
    data.sort(key=lambda x: x["date_end"])
    return data[-1]

# Mapping from words in the question -> (country_name, track_hint)
TRACK_KEYWORDS = {
    # Italy
    "monza": ("Italy", "Monza"),
    "imola": ("Italy", "Imola"),

    # UK
    "silverstone": ("Great Britain", "Silverstone"),
    "british gp": ("Great Britain", "Silverstone"),

    # Belgium
    "spa": ("Belgium", "Spa"),

    # Netherlands
    "zandvoort": ("Netherlands", "Zandvoort"),

    # Austria
    "red bull ring": ("Austria", "Red Bull Ring"),
    "austria gp": ("Austria", "Red Bull Ring"),

    # Hungary
    "hungaroring": ("Hungary", "Hungaroring"),

    # Spain
    "barcelona": ("Spain", "Barcelona"),
    "catalunya": ("Spain", "Barcelona"),

    # Monaco
    "monaco": ("Monaco", "Monaco"),

    # Middle East / Asia
    "bahrain": ("Bahrain", "Bahrain"),
    "sakhir": ("Bahrain", "Sakhir"),
    "jeddah": ("Saudi Arabia", "Jeddah"),
    "saudi": ("Saudi Arabia", "Jeddah"),
    "yas marina": ("Abu Dhabi", "Yas Marina"),
    "abu dhabi": ("Abu Dhabi", "Yas Marina"),
    "suzuka": ("Japan", "Suzuka"),
    "qatar": ("Qatar", "Qatar"),
    "losail": ("Qatar", "Losail"),

    # Americas (all in USA/Canada/Brazil/Mexico)
    "interlagos": ("Brazil", "Interlagos"),
    "brazil": ("Brazil", "Interlagos"),   # rough but ok
    "cota": ("USA", "Americas"),
    "austin": ("USA", "Americas"),
    "miami": ("USA", "Miami"),
    "vegas": ("USA", "Vegas"),
    "las vegas": ("USA", "Vegas"),
    "montreal": ("Canada", "Montreal"),
    "canadian gp": ("Canada", "Montreal"),
    "mexico": ("Mexico", "Mexico"),

    # Australia
    "melbourne": ("Australia", "Melbourne"),
    "albert park": ("Australia", "Albert Park"),
}

def choose_session_for_question(question: str):
    """
    Decide which race session to use based on the user's wording.

    Priority:
    1) If the question mentions a known track/GP keyword (e.g. 'Monza', 'Imola'),
       pick the latest Race at that circuit within its country.
    2) Otherwise, fall back to the absolute latest race (get_latest_race_session).
    """
    q = question.lower()

    for keyword, (country_name, track_hint) in TRACK_KEYWORDS.items():
        if keyword in q:
            # Fetch all races in that country
            params = {"session_type": "Race", "country_name": country_name}
            resp = requests.get(f"{OPENF1_BASE}/sessions", params=params, timeout=10)
            resp.raise_for_status()
            sessions = resp.json()
            if not sessions:
                continue

            track_hint_lower = track_hint.lower()

            # Prefer sessions whose meeting/location/circuit names contain the track_hint
            def has_track_hint(s):
                text = " ".join([
                    str(s.get("meeting_name", "")),
                    str(s.get("location", "")),
                    str(s.get("circuit_short_name", "")),
                ]).lower()
                return track_hint_lower in text

            track_sessions = [s for s in sessions if has_track_hint(s)]

            candidates = track_sessions or sessions
            candidates.sort(key=lambda x: x["date_end"])
            return candidates[-1]

    # No track keyword matched -> global latest race
    return get_latest_race_session()


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

import re  # at top of file if not already imported

def detect_driver_number_from_question(question: str, session_key: int):
    """
    Try to figure out which driver the user is talking about
    based on the text, using OpenF1 driver list for that session.
    """
    q = question.lower()

    # Fetch all drivers for this session
    params = {"session_key": session_key}
    resp = requests.get(f"{OPENF1_BASE}/drivers", params=params, timeout=10)
    resp.raise_for_status()
    drivers = resp.json()

    # First, try to match by any of first/last/full/broadcast name appearing in the question
    for d in drivers:
        candidates = set()
        if d.get("full_name"):
            candidates.add(d["full_name"].lower())
        if d.get("first_name"):
            candidates.add(d["first_name"].lower())
        if d.get("last_name"):
            candidates.add(d["last_name"].lower())
        if d.get("broadcast_name"):
            candidates.add(d["broadcast_name"].lower())

        for name in candidates:
            if not name:
                continue
            # Simple containment check
            if name in q:
                return d["driver_number"]

    # Fallback: some common short-name nicknames
    # (this handles 'lewis', 'max', 'checo', 'lando' etc if above didn't catch)
    nickname_map = {
        "lewis": 44,
        "hamilton": 44,
        "max": 1,
        "verstappen": 1,
        "checo": 11,
        "perez": 11,
        "charles": 16,
        "leclerc": 16,
        "lando": 4,
        "norris": 4,
        "george": 63,
        "russell": 63,
        "carlos": 55,
        "sainz": 55,
    }

    for nick, num in nickname_map.items():
        if nick in q:
            return num

    return None

def detect_lap_from_question(question: str):
    """
    Look for patterns like 'lap 16' in the question.
    Returns an int lap number or None if not found.
    """
    q = question.lower()
    match = re.search(r"lap\s+(\d+)", q)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None



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


@app.route("/analyze_latest", methods=["POST"])
def analyze_latest():
    """
    Natural-language endpoint.

    Expects JSON:
    {
      "question": "Was it right to pit Lewis or extend his stint?"
    }

    Logic:
    - Use latest race from OpenF1.
    - Detect driver from question.
    - Detect lap from question if present.
    - If no lap, choose first pit stop (between stint 1 and 2).
    - Reuse existing RaceMeta pipeline.
    """
    try:
        data = request.get_json(force=True)
        question = str(data["question"])
    except Exception as e:
        return jsonify({"error": f"Invalid payload: {e}"}), 400

    try:
        # 1) Choose race session based on question (track-aware)
        session = choose_session_for_question(question)
        session_key = session["session_key"]
        meeting_key = session["meeting_key"]


        # 2) Detect driver
        driver_number = detect_driver_number_from_question(question, session_key)
        if driver_number is None:
            return jsonify({"error": "Could not detect driver from question for latest race."}), 400

        # 3) Fetch stints for this driver
        stints = get_stints(session_key, driver_number)
        if not stints:
            return jsonify({"error": "No stint data for this driver in latest race."}), 400

        # 4) Detect lap from question, else infer first stop from stints
        pit_lap = detect_lap_from_question(question)
        stints_sorted = sorted(stints, key=lambda s: s["stint_number"])

        if pit_lap is None:
            if len(stints_sorted) > 1:
                pit_lap = stints_sorted[1]["lap_start"] - 1
            else:
                pit_lap = (stints_sorted[0]["lap_start"] + stints_sorted[0]["lap_end"]) // 2

        # 5) Driver + weather
        driver_info = get_driver_info(session_key, driver_number)
        weather = get_weather(meeting_key)

        # 6) Build context + call RaceMeta
        stint_text, before_comp, after_comp = summarise_stints(stints, pit_lap)

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
