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

# Threads config (for @race.meta bot)
THREADS_USER_TOKEN = os.environ.get("THREADS_USER_TOKEN")
THREADS_VERIFY_TOKEN = os.environ.get("THREADS_VERIFY_TOKEN", "changeme")
THREADS_API_BASE = "https://graph.threads.net/v1.0"

app = Flask(__name__)
CORS(app)

# -------- HELPERS: OPENF1 --------


def get_race_session(year: int, country_name: str):
    params = {
        "year": year,
        "country_name": country_name,
        "session_type": "Race",
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

    # Americas
    "interlagos": ("Brazil", "Interlagos"),
    "brazil": ("Brazil", "Interlagos"),
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

            def has_track_hint(s):
                text = " ".join(
                    [
                        str(s.get("meeting_name", "")),
                        str(s.get("location", "")),
                        str(s.get("circuit_short_name", "")),
                    ]
                ).lower()
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

    # Try matching by name variants
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
            if name and name in q:
                return d["driver_number"]

    # Fallback nicknames
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

def compute_metrics_from_openf1(stints, pit_lap: int, driver_info):
    """
    Build a small metrics dict using only real OpenF1 data.

    - Tyre life: how many laps of the current stint were still unused at the pit lap
    - Final position: finishing position from OpenF1 driver info
    """
    metrics = {}

    # --- Tyre life ---
    if stints:
        stints_sorted = sorted(stints, key=lambda s: s["stint_number"])
        for stint in stints_sorted:
            ls = stint["lap_start"]
            le = stint["lap_end"]
            if ls <= pit_lap <= le:
                stint_len = (le - ls + 1)
                laps_used = pit_lap - ls + 1
                laps_remaining = max(0, le - pit_lap)

                # Normalised 0–100 score for the progress bar
                if stint_len > 0:
                    life_score = int(max(0, min(100, (laps_remaining / stint_len) * 100)))
                else:
                    life_score = 0

                metrics["tyre_life"] = {
                    "label": "Tyre life",
                    "value": f"{laps_remaining} laps",
                    "description": "Estimated usable life left in this stint when the stop was made.",
                    "score": life_score,
                    # If you still had a decent margin, that’s generally positive
                    "impact": "positive" if laps_remaining >= 3 else "neutral",
                }
                break

    # --- Final position ---
    if driver_info:
        # Different OpenF1 dumps sometimes call this 'position' or similar
        finishing_pos = (
            driver_info.get("position")
            or driver_info.get("position_over_line")
            or driver_info.get("classification_position")
        )

        try:
            if finishing_pos is not None:
                pos_int = int(finishing_pos)
                # Simple mapping: P1 = 100, P2 = 90, P3 = 80, etc.
                score = max(0, min(100, 110 - pos_int * 10))

                metrics["final_position"] = {
                    "label": "Final position",
                    "value": f"P{pos_int}",
                    "description": "Classified finishing position in this race.",
                    "score": score,
                    "impact": "positive" if pos_int <= 3 else "negative",
                }
        except (TypeError, ValueError):
            pass

    return metrics



# -------- RACEMETA PROMPT --------

RACEMETA_SYSTEM_PROMPT = """
You are RaceMeta, an F1 pit-wall meta strategist. You explain race strategy calls
(pit timing, tyre choice, stint length) for fans who follow F1 closely.

You ALWAYS answer in this format:

Verdict: <2–8 words, strong opinion – e.g. "Optimal cover", "Costly overcut", "Mixed – defendable">
Why:
- Point 1 (max ~18 words, ONE clear idea)
- Point 2
- Point 3
Alt: One-line alternative call you’d have made.
Take: One-line general lesson that works as a tweet.

Style rules:
- Tone: confident, conversational, slightly spicy, but never disrespectful to teams or drivers.
- No hashtags, no emojis.
- Total length ~280–350 characters if possible (OK to go a bit over if needed for clarity).
- Each bullet = one idea. Avoid comma chains like "did X, Y and Z" – split into separate bullets.
- Use concrete race logic: tyre state, track position, undercut/overcut power, safety car risk, traffic, etc.
- If data is missing or noisy, say so briefly and lean on typical F1 strategy logic rather than inventing facts.

Clarification behaviour (very important):
- If the question is TOO vague for a fair verdict (e.g. no clear driver, race, or situation),
  reply like this instead:

Verdict: Need more info
Why:
- Say exactly what’s missing (driver? race? lap? tyre question?)
Alt: Ask the user for 1–2 specific details.
Take: Clear inputs = sharper strategy calls.

Never add anything outside this template.
"""


def build_context_block(
    session,
    driver_info,
    weather,
    stint_text,
    before_comp,
    after_comp,
    pit_lap,
    user_question,
):
    track_name = (
        session.get("circuit_short_name")
        or session.get("location")
        or "Unknown circuit"
    )
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
        f"Pit of interest: lap {pit_lap} (from {before_comp} to {after_comp})",
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
            {"role": "user", "content": context_block},
        ],
        temperature=0.4,
        max_completion_tokens=180,
    )
    return completion.choices[0].message.content.strip()


# -------- NEW HELPERS: PARSE & RACE CONTEXT --------


def classify_verdict_type(text: str) -> str:
    """
    Map summary/verdict text into 'optimal', 'suboptimal' or 'neutral'.
    """
    t = (text or "").lower()
    if any(x in t for x in ["suboptimal", "too early", "too late", "costly", "bad call"]):
        return "suboptimal"
    if any(x in t for x in ["optimal", "good call", "right call", "smart", "strong"]):
        return "optimal"
    if "need more info" in t:
        return "neutral"
    return "neutral"


def parse_racemeta_verdict(verdict_text: str):
    """
    Parse the formatted RaceMeta answer into structured pieces:
    summary, verdict_type, reasoning list, recommended_strategy.
    """
    if not verdict_text:
        return {
            "summary": "",
            "verdict_type": "neutral",
            "reasoning": [],
            "recommended_strategy": "",
        }

    text = verdict_text.strip()

    # Summary = line after "Verdict:"
    summary = ""
    m_verdict = re.search(r"Verdict:\s*(.*)", text, re.IGNORECASE)
    if m_verdict:
        summary = m_verdict.group(1).strip()

    # Why-section (between "Why:" and the next "Alt:" or "Take:" line or end)
    reasoning = []
    m_why = re.search(
        r"Why:\s*([\s\S]*?)(?:\nAlt:|\nTake:|$)",
        text,
        re.IGNORECASE,
    )
    if m_why:
        why_part = m_why.group(1)
        for line in why_part.splitlines():
            l = line.strip()
            if not l:
                continue
            # Strip leading "- " or "• "
            l = re.sub(r"^[-•]\s*", "", l)
            if l:
                reasoning.append(l)

    # Alt: recommended_strategy (between "Alt:" and "Take:" or end)
    recommended_strategy = ""
    m_alt = re.search(
        r"Alt:\s*([\s\S]*?)(?:\nTake:|$)",
        text,
        re.IGNORECASE,
    )
    if m_alt:
        recommended_strategy = m_alt.group(1).strip()

    verdict_type = classify_verdict_type(summary or verdict_text)

    return {
        "summary": summary or verdict_text,
        "verdict_type": verdict_type,
        "reasoning": reasoning,
        "recommended_strategy": recommended_strategy,
    }



def extract_race_context_from_context_block(context_block: str, fallback_pit_lap=None):
    """
    Parse 'Race:' and 'Driver:' lines + pit lap from the text context block.
    This lets us build a race_context object for the frontend.
    """
    driver = None
    race = None
    lap = None

    # Race: Country Year (Track)
    m_race = re.search(r"^Race:\s*(.+)$", context_block, re.IGNORECASE | re.MULTILINE)
    if m_race:
        race = m_race.group(1).strip()

    # Driver: Name
    m_driver = re.search(r"^Driver:\s*(.+)$", context_block, re.IGNORECASE | re.MULTILINE)
    if m_driver:
        driver = m_driver.group(1).strip()

    # Pit of interest: lap N
    m_lap = re.search(r"Pit of interest:\s*lap\s+(\d+)", context_block, re.IGNORECASE)
    if m_lap:
        try:
            lap = int(m_lap.group(1))
        except ValueError:
            lap = None

    if lap is None and fallback_pit_lap is not None:
        lap = fallback_pit_lap

    # Build label
    label_parts = []
    if driver:
        label_parts.append(driver)
    if race:
        label_parts.append(race)
    if lap is not None:
        label_parts.append(f"lap {lap}")
    label = " – ".join(label_parts) if label_parts else "Race context"

    return {
        "driver": driver or "Unknown driver",
        "race": race or "Unknown race",
        "lap": lap,
        "position": None,  # we can wire this later from OpenF1 laps data
        "label": label,
    }


# -------- THREADS HELPER --------


def threads_post_text_reply(text: str, reply_to_id: str):
    """
    Post a text reply on Threads as the authenticated user (@race.meta).
    """
    if not THREADS_USER_TOKEN:
        print("THREADS_USER_TOKEN not set, skipping Threads reply.")
        return None

    params = {
        "text": text,
        "media_type": "TEXT",
        "reply_to_id": reply_to_id,
        "auto_publish_text": "true",
        "access_token": THREADS_USER_TOKEN,
    }

    try:
        resp = requests.post(
            f"{THREADS_API_BASE}/me/threads", params=params, timeout=15
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print("Error posting reply to Threads:", e)
        return None


# -------- HTTP ENDPOINTS --------


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Structured endpoint for a specific race/driver/lap.
    Kept mostly for completeness; your playground currently uses /analyze_latest.
    """
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

        parsed = parse_racemeta_verdict(verdict)
        race_context = extract_race_context_from_context_block(context_block, fallback_pit_lap=pit_lap)

        # Placeholder metrics – we will fill these from OpenF1 later
        metrics = compute_metrics_from_openf1(stints, pit_lap, driver_info)


      return jsonify({
    "context": context_block,
    "verdict": verdict_text,
    "summary": summary,
    "verdict_type": verdict_type,
    "reasoning": reasoning,
    "recommended_strategy": recommended_strategy,
    "confidence": confidence,
    "race_context": race_context,
    "metrics": metrics,
})


    except Exception as e:
        return jsonify({"error": str(e)}), 500


def analyze_latest_core(question: str):
    """
    Core logic for 'latest race' natural-language analysis.
    Returns (context_block, verdict).
    """
    # 1) Choose race session based on question (track-aware)
    session = choose_session_for_question(question)
    session_key = session["session_key"]
    meeting_key = session["meeting_key"]

    # 2) Detect driver
    driver_number = detect_driver_number_from_question(question, session_key)
    if driver_number is None:
        raise ValueError("Could not detect driver from question for latest race.")

    # 3) Fetch stints for this driver
    stints = get_stints(session_key, driver_number)
    if not stints:
        raise ValueError("No stint data for this driver in latest race.")

    # 4) Detect lap from question, else infer first stop from stints
    pit_lap = detect_lap_from_question(question)
    stints_sorted = sorted(stints, key=lambda s: s["stint_number"])

    if pit_lap is None:
        if len(stints_sorted) > 1:
            pit_lap = stints_sorted[1]["lap_start"] - 1
        else:
            pit_lap = (
                stints_sorted[0]["lap_start"] + stints_sorted[0]["lap_end"]
            ) // 2

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

    return context_block, verdict


@app.route("/analyze_latest", methods=["POST"])
def analyze_latest():
    """
    Natural-language endpoint used by the playground.

    Expects JSON:
    {
      "question": "Was it right to pit Lewis or extend his stint?"
    }

    Returns both legacy fields (context, verdict) and structured fields
    (summary, verdict_type, reasoning, recommended_strategy, race_context, metrics, confidence).
    """
    try:
        data = request.get_json(force=True)
        question = str(data["question"])
    except Exception as e:
        return jsonify({"error": f"Invalid payload: {e}"}), 400

    try:
        context_block, verdict = analyze_latest_core(question)

        parsed = parse_racemeta_verdict(verdict)
        race_context = extract_race_context_from_context_block(context_block)

        metrics = {}

        return jsonify(
            {
                "context": context_block,
                "verdict": verdict,
                "summary": parsed["summary"],
                "verdict_type": parsed["verdict_type"],
                "reasoning": parsed["reasoning"],
                "recommended_strategy": parsed["recommended_strategy"],
                "race_context": race_context,
                "metrics": metrics,
                "confidence": 80,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/threads_webhook", methods=["GET", "POST"])
def threads_webhook():
    """
    Webhook for Threads mentions.

    - GET: verification handshake with Meta (hub.challenge).
    - POST: handle mention events, run RaceMeta, and reply.
    """
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if mode == "subscribe" and token == THREADS_VERIFY_TOKEN:
            return challenge, 200
        return "Verification failed", 403

    # POST = actual events
    try:
        payload = request.get_json(force=True)
    except Exception:
        return "Invalid payload", 400

    try:
        entries = payload.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})

                mention_text = value.get("text") or value.get("message")
                media_id = value.get("media_id") or value.get("id")

                if not mention_text or not media_id:
                    continue

                # Strip '@race.meta' from the text to get the actual question
                question = re.sub(
                    r"@race\.meta", "", mention_text, flags=re.IGNORECASE
                ).strip()
                if not question:
                    continue

                try:
                    _, verdict = analyze_latest_core(question)
                    threads_post_text_reply(verdict, reply_to_id=str(media_id))
                except Exception as e:
                    print("Error handling mention:", e)

    except Exception as e:
        print("Error parsing Threads webhook payload:", e)

    # Always 200 so Meta doesn't keep retrying forever
    return "OK", 200


@app.route("/", methods=["GET"])
def health():
    return "RaceMeta OpenF1 backend is running.", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)


