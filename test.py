from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
from neo4j import GraphDatabase
from PyPDF2 import PdfReader
import uuid
import os
from copy import deepcopy
import mysql.connector
# from payment_service import PaymentService
from typing import Optional, Dict,List
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional
import torch
from diffusers import DiffusionPipeline
import imageio
import uuid
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
from fastapi.responses import StreamingResponse
import io
import replicate

load_dotenv()

conn=None
cursor=None

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NEO4J_URI = "."
NEO4J_USER = "."
NEO4J_PASSWORD = "."
LLM_ENDPOINT = "https://godfatherpersonalcomputer.shop/asklaptop"
LLM_MODEL = "gpt-oss:120b-cloud"

class ImageRequest(BaseModel):
    prompt: str


# -------- Route --------
@app.post("/hacks/image")
def generate_image(payload: ImageRequest):

    input_data = {
        "prompt": payload.prompt,
        "aspect_ratio": "16:9",
        "safety_filter_level": "block_medium_and_above"
    }

    # Call Replicate model
    output = replicate.run(
        "google/imagen-4",
        input=input_data
    )

    # Replicate returns a URL → fetch image bytes
    image_url = output.url
    response = requests.get(image_url)

    image_bytes = io.BytesIO(response.content)

    return StreamingResponse(
        image_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": "inline; filename=output.png"
        },
    )


class TTSRequest(BaseModel):
    text: str


# -------- ElevenLabs Client --------
client = ElevenLabs(
    api_key="sk_4ae4e39a3e868f3bfb011377e6b902c65c2fe23a3e2f06d4"   # replace or load from env
)


# -------- Route --------
@app.post("/hacks/tts")
def generate_tts(payload: TTSRequest):

    audio_stream = client.text_to_speech.convert(
        text=payload.text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    return StreamingResponse(
        audio_stream,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline; filename=tts.mp3"
        },
    )




# -------- Request Schema --------
class VideoRequest(BaseModel):
    prompt: str


# -------- Load model ONCE at startup --------
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe.to(device)

OUTPUT_DIR = "generated_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------- Route --------
@app.post("/hacks/video")
def generate_video(payload: VideoRequest):

    prompt = payload.prompt

    result = pipe(
        prompt,
        num_inference_steps=30,
        num_frames=24
    )

    frames = result.frames[0]

    filename = f"{uuid.uuid4()}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    imageio.mimsave(output_path, frames, fps=8)

    return {
        "status": "success",
        "file": filename,
        "path": output_path
    }




driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Allowlists ────────────────────────────────────────────────────────────────

ALLOWED_LABELS = {"Character", "Location", "Organization", "Object", "Event", "Creature"}

ALLOWED_RELATIONS = {
    "VISITS", "TRAVELS_TO", "ARRIVES_AT", "ESCAPES_FROM", "RETURNS_TO",
    "ENTERS", "LEAVES", "EXPLORES", "HIDES_IN",
    "KNOWS", "FRIEND_OF", "ALLY_OF", "ENEMY_OF", "MENTORS", "STUDENT_OF",
    "WORKS_WITH", "LEADS", "FOLLOWS", "SERVES", "BETRAYS", "TRUSTS",
    "DISTRUSTS", "PROTECTS", "RESCUES", "ABANDONS", "LOVES", "HATES", "FEARS",
    "FIGHTS", "ATTACKS", "DEFENDS", "CHASES", "CAPTURES", "INTERROGATES",
    "THREATENS", "KILLS", "INJURES", "DESTROYS", "DEFEATS", "SURVEILS", "AMBUSHES",
    "OWNS", "POSSESSES", "STEALS", "LOSES", "FINDS", "USES", "WIELDS",
    "GUARDS", "GIVES", "RECEIVES",
    "KNOWS_ABOUT", "DISCOVERS", "INVESTIGATES", "REVEALS", "LEARNS",
    "TEACHES", "REMEMBERS", "OBSERVES",
    "MEMBER_OF", "LEADS_ORGANIZATION", "REBELS_AGAINST", "SERVES_UNDER",
    "CONTROLS", "RULES", "COMMANDS", "SUPPORTS", "OPPOSES",
    "LOCATED_IN", "PART_OF", "NEAR", "CONNECTED_TO", "HIDDEN_IN",
    "ORIGINATES_FROM", "OCCURS_IN", "HAPPENS_AT",
    "CAUSES", "TRIGGERS", "PREVENTS", "ENABLES", "RESULTS_IN",
    "LEADS_TO", "CONFLICTS_WITH", "RESOLVES",
    "ACTIVATES", "DEACTIVATES", "CREATES", "MODIFIES", "REPAIRS",
    "BREAKS", "SUMMONS", "UNLOCKS",
}


# ── Conflict dictionaries ────────────────────────────────────────────────────

POSITIVE_RELS = {
    "ALLY_OF", "FRIEND_OF", "TRUSTS", "PROTECTS", "LOVES", "RESCUES",
    "SUPPORTS", "SERVES", "FOLLOWS", "MEMBER_OF", "MENTORS", "STUDENT_OF",
    "WORKS_WITH", "GUARDS", "GIVES", "CREATES", "REPAIRS",
}
NEGATIVE_RELS = {
    "ENEMY_OF", "BETRAYS", "HATES", "DISTRUSTS", "ATTACKS", "FIGHTS",
    "KILLS", "INJURES", "THREATENS", "CAPTURES", "CHASES", "DESTROYS",
    "DEFEATS", "AMBUSHES", "ABANDONS", "STEALS", "OPPOSES",
    "REBELS_AGAINST",
}
STRUCTURAL_CONFLICTS = {
    "LEADS": {"SERVES_UNDER", "FOLLOWS"},
    "CONTROLS": {"ESCAPES_FROM", "REBELS_AGAINST"},
    "RULES": {"ESCAPES_FROM", "REBELS_AGAINST", "DEFEATS"},
    "CAPTURES": {"ESCAPES_FROM", "RESCUES"},
    "ESCAPES_FROM": {"CAPTURES", "SERVES", "FOLLOWS"},
    "DESTROYS": {"REPAIRS", "CREATES"},
    "CREATES": {"DESTROYS"},
}

STATE_TERMINAL_RELS = {"DESTROYS", "KILLS"}
STATE_REQUIRES_ALIVE = {
    "VISITS", "ENTERS", "EXPLORES", "ARRIVES_AT", "RETURNS_TO",
    "GUARDS", "PROTECTS", "LEADS", "CONTROLS", "RULES", "COMMANDS",
    "FIGHTS", "ATTACKS", "CAPTURES", "SERVES", "FOLLOWS",
    "TRAVELS_TO", "WORKS_WITH", "MENTORS",
}

EXCLUSIVE_ROLES = {"LEADS", "CONTROLS", "RULES", "COMMANDS", "LEADS_ORGANIZATION"}

DOMINANT_RELATIONS = {"LEADS", "CONTROLS", "DESTROYS", "COMMANDS", "TRIGGERS", "KILLS"}


def _get_polarity(rel: str) -> str:
    if rel in POSITIVE_RELS:
        return "POSITIVE"
    if rel in NEGATIVE_RELS:
        return "NEGATIVE"
    return "NEUTRAL"


def _relations_conflict(rel_a: str, rel_b: str) -> bool:
    pol_a, pol_b = _get_polarity(rel_a), _get_polarity(rel_b)
    if pol_a != "NEUTRAL" and pol_b != "NEUTRAL" and pol_a != pol_b:
        return True
    if rel_b in STRUCTURAL_CONFLICTS.get(rel_a, set()):
        return True
    if rel_a in STRUCTURAL_CONFLICTS.get(rel_b, set()):
        return True
    return False


# ── LLM prompts ──────────────────────────────────────────────────────────────

EXTRACT_SYSTEM_PROMPT = """You are an exhaustive narrative knowledge-graph extractor.
Given a chapter of a story, extract EVERY entity and relationship into structured triples.
Be extremely thorough — capture every detail that could matter for story continuity.

## EXTRACTION DEPTH REQUIREMENTS

For EVERY sentence, extract ALL of the following that apply:

1. CHARACTER RELATIONSHIPS: Who knows whom, alliances, enmities, loyalties, betrayals,
romantic bonds, mentorships, rivalries, family ties, trust, distrust.
2. MOVEMENT & PRESENCE: Every travel, arrival, departure, entry, escape. Where is each
character at each point in the chapter? Track location changes carefully.
3. ACTIONS & CONFLICT: Every fight, attack, defense, chase, capture, killing, injury,
threat, ambush. Who did what to whom?
4. POSSESSION & OBJECTS: Who owns, carries, wields, uses, steals, loses, finds, gives,
receives what object? Track every object interaction.
5. KNOWLEDGE & DISCOVERY: Who learns, discovers, reveals, investigates, observes, or
remembers what? Track information flow between characters.
6. POWER & POLITICS: Leadership, control, commands, service, rebellion, organizational
membership, hierarchy changes, power shifts.
7. LOCATION STRUCTURE: Which locations are part of which, connected to which, near which,
hidden in which? Capture the geography.
8. CAUSALITY & EVENTS: What triggers, causes, prevents, enables, leads to what?
Name events explicitly and link their causes and effects.
9. EMOTIONAL STATES: Fears, loves, hates — capture every expressed emotion as a relation.
10. OBJECT INTERACTIONS: Activations, deactivations, creation, destruction, modification,
    repair, summoning, unlocking.

## ENTITY IDENTIFICATION RULES

- Extract EVERY named entity: characters, places, objects, groups, events, creatures.
- Also extract UNNAMED but important entities by giving them descriptive names
(e.g. "Prison Guard", "Self-Destruct Sequence", "Hidden Portal").
- For groups acting as one ("the guards", "the council"), create an Organization or
Character entity with a descriptive name.
- An entity can appear in MULTIPLE triples — do not skip repeated appearances.

## ALLOWED NODE LABELS
Character, Location, Organization, Object, Event, Creature

## ALLOWED RELATIONSHIP TYPES (use ONLY these, uppercase)
VISITS, TRAVELS_TO, ARRIVES_AT, ESCAPES_FROM, RETURNS_TO, ENTERS, LEAVES,
EXPLORES, HIDES_IN, KNOWS, FRIEND_OF, ALLY_OF, ENEMY_OF, MENTORS,
STUDENT_OF, WORKS_WITH, LEADS, FOLLOWS, SERVES, BETRAYS, TRUSTS,
DISTRUSTS, PROTECTS, RESCUES, ABANDONS, LOVES, HATES, FEARS, FIGHTS,
ATTACKS, DEFENDS, CHASES, CAPTURES, INTERROGATES, THREATENS, KILLS,
INJURES, DESTROYS, DEFEATS, SURVEILS, AMBUSHES, OWNS, POSSESSES,
STEALS, LOSES, FINDS, USES, WIELDS, GUARDS, GIVES, RECEIVES,
KNOWS_ABOUT, DISCOVERS, INVESTIGATES, REVEALS, LEARNS, TEACHES,
REMEMBERS, OBSERVES, MEMBER_OF, LEADS_ORGANIZATION, REBELS_AGAINST,
SERVES_UNDER, CONTROLS, RULES, COMMANDS, SUPPORTS, OPPOSES,
LOCATED_IN, PART_OF, NEAR, CONNECTED_TO, HIDDEN_IN, ORIGINATES_FROM,
OCCURS_IN, HAPPENS_AT, CAUSES, TRIGGERS, PREVENTS, ENABLES,
RESULTS_IN, LEADS_TO, CONFLICTS_WITH, RESOLVES, ACTIVATES,
DEACTIVATES, CREATES, MODIFIES, REPAIRS, BREAKS, SUMMONS, UNLOCKS

## STRICT RULES
- Output STRICT JSON only. No markdown, no explanations, no commentary.
- Normalize similar verbs to the closest allowed relation.
- Remove exact duplicate triples.
- Prefer MORE triples over fewer — err on the side of capturing too much.
- Every character should have at least one LOCATED_IN, VISITS, or movement relation.
- Every mentioned object should have at least one OWNS, USES, POSSESSES, or WIELDS relation.
- Every event should have at least one CAUSES, TRIGGERS, or RESULTS_IN relation.
- If a character performs an action somewhere, produce BOTH the action triple AND a location triple.

## OUTPUT FORMAT
{"triples":[{"source":"...","source_type":"...","relation":"...","target":"...","target_type":"..."}]}
"""

NARRATOR_PROMPT = """You are a story continuity checker. You receive a list of detected issues from an automated graph comparison.

Your ONLY job: convert those issues into one concise natural language paragraph.

STRICT RULES:
- ONLY mention facts that appear in the DETECTED ISSUES list. Never invent information.
- Never repeat or summarize the full graph. Never describe new events that aren't issues.
- Do not use diagrams, arrows, technical notation, or bullet points.
- Use character and entity names naturally in sentences.
- If issues exist, start your paragraph with "This is not according to the story..."
- If DETECTED ISSUES is empty or says 'none', respond with EXACTLY: "No inconsistencies found. The new chapter aligns with the existing story."
- One paragraph maximum.
"""

import re

def safe_json_parse(raw: str):
    # extract first JSON array or object
    match = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM output")

    return json.loads(match.group(1))


# ── Core functions ────────────────────────────────────────────────────────────

def _extract_triples(chapter_text: str) -> list[dict]:
    query = f"{EXTRACT_SYSTEM_PROMPT}\n\n{chapter_text}"
    response = requests.post(
        LLM_ENDPOINT,
        headers={"Content-Type": "application/json"},
        json={
            "query": query,
            "model_name": LLM_MODEL,
            "stream": False
        }
    )
    response.raise_for_status()
    raw = response.json().get("response", "")
    raw = raw.strip()
    if raw.startswith(""):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = safe_json_parse(raw)
    return [
        t for t in data.get("triples", [])
        if t.get("source_type") in ALLOWED_LABELS
        and t.get("target_type") in ALLOWED_LABELS
        and t.get("relation") in ALLOWED_RELATIONS
    ]


def _check_triple_conflict(session, t: dict) -> Optional[str]:
    """
    Traverse Neo4j neighbourhood for ONE triple.
    Returns a conflict description string, or None if clean.
    """
    src, rel, tgt = t["source"], t["relation"], t["target"]
    src_type, tgt_type = t["source_type"], t["target_type"]

    # ── 1. Type conflict — check existing labels for source and target ────
    for name, expected_type in [(src, src_type), (tgt, tgt_type)]:
        records = session.run(
            "MATCH (n {name: $name}) RETURN labels(n)[0] AS label",
            name=name
        ).data()
        if records:
            existing_label = records[0]["label"]
            if existing_label != expected_type:
                return (f"Role shift: '{name}' was a {existing_label} "
                        f"but is now described as {expected_type}")

    # ── 2. Gather outgoing rels from source toward target ─────────────────
    outgoing = session.run(
        "MATCH (a {name: $source})-[r]->(b {name: $target}) RETURN type(r) AS rel",
        source=src, target=tgt
    ).data()
    outgoing_rels = {r["rel"] for r in outgoing}

    # ── 3. Gather reverse rels (target → source) ─────────────────────────
    reverse = session.run(
        "MATCH (a {name: $target})-[r]->(b {name: $source}) RETURN type(r) AS rel",
        source=src, target=tgt
    ).data()
    reverse_rels = {r["rel"] for r in reverse}

    # ── 4. Semantic conflict (polarity + structural, both directions) ─────
    for existing_rel in outgoing_rels | reverse_rels:
        if _relations_conflict(rel, existing_rel):
            return (f"'{src}' was previously '{existing_rel}' with '{tgt}', "
                    f"but the new chapter introduces '{rel}' — these contradict each other.")

    # ── 5. State conflict — acting on destroyed / killed entity ───────────
    if rel in STATE_REQUIRES_ALIVE:
        # Check if target has been destroyed/killed
        terminal_on_tgt = session.run(
            "MATCH (a)-[r]->(b {name: $name}) "
            "WHERE type(r) IN $rels RETURN type(r) AS rel, a.name AS by",
            name=tgt, rels=list(STATE_TERMINAL_RELS)
        ).data()
        if terminal_on_tgt:
            return (f"'{tgt}' was previously destroyed/killed, "
                    f"but the new chapter has '{src}' performing '{rel}' on it.")

        # Check if source has been destroyed/killed
        terminal_on_src = session.run(
            "MATCH (a)-[r]->(b {name: $name}) "
            "WHERE type(r) IN $rels RETURN type(r) AS rel, a.name AS by",
            name=src, rels=list(STATE_TERMINAL_RELS)
        ).data()
        if terminal_on_src:
            return (f"'{src}' was previously destroyed/killed, "
                    f"but the new chapter has it performing '{rel}' on '{tgt}'.")

    # ── 6. Exclusive role conflict ────────────────────────────────────────
    if rel in EXCLUSIVE_ROLES:
        holders = session.run(
            "MATCH (a)-[r:" + rel + "]->(b {name: $target}) "
            "RETURN a.name AS holder",
            target=tgt
        ).data()
        for h in holders:
            if h["holder"] != src:
                return (f"'{h['holder']}' already '{rel}' '{tgt}', "
                        f"but now '{src}' also claims to — only one should.")

    # ── 7. Causal conflict — two different causes for same event ──────────
    if rel in ("TRIGGERS", "CAUSES"):
        existing_causes = session.run(
            "MATCH (a)-[r]->(b {name: $target}) "
            "WHERE type(r) IN ['TRIGGERS', 'CAUSES'] "
            "RETURN a.name AS cause, type(r) AS rel",
            target=tgt
        ).data()
        for ec in existing_causes:
            if ec["cause"] != src:
                return (f"'{ec['cause']}' already '{ec['rel']}' '{tgt}', "
                        f"but now '{src}' also claims to cause it.")

    # ── 8. Dominant new entity warning ────────────────────────────────────
    if rel in DOMINANT_RELATIONS:
        exists = session.run(
            "MATCH (n {name: $name}) RETURN n LIMIT 1",
            name=src
        ).data()
        if not exists:
            return (f"New entity '{src}' immediately '{rel}' '{tgt}' "
                    f"— high-impact introduction.")

    return None

class very(BaseModel):
    input:str

@app.post('/hacks/verify')
def verify(payload:very):
    prompt = f"""
        You are a PROFESSIONAL STORY EDITOR specializing in detecting inconsistencies.

        Your task is to carefully analyze the provided story and identify problems related to:

        - Character consistency (behavior, motivations, dialogue, knowledge)
        - World or environment details (rules, setting logic, continuity)
        - Timeline and causality issues
        - Emotional or logical contradictions
        - Unclear or conflicting narrative information

        Guidelines:
        - Focus ONLY on inconsistencies or continuity problems.
        - Base every observation strictly on the provided text.
        - Do NOT rewrite the story.
        - Do NOT suggest fixes or improvements.
        - Do NOT invent missing details.

        Output Requirements:
        Return STRICT JSON in the following structure:

        {{
        "issues": [
            {{
            "type": "character | environment | timeline | logic | continuity",
            "section": "short identifier or reference",
            "problem": "clear description of the inconsistency",
            "evidence": "brief quote or explanation grounded in the text",
            "severity": 0.0
            }}
        ]
        }}

        The provided story is {payload.input} 
        """
    try:
        # 🔥 LOCAL OLLAMA CALL
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "audience-agent:latest",
                "prompt": prompt,
                "stream": False,
                "format": "json"   # VERY important for local models
            },
            timeout=300
        )

        ollama_response.raise_for_status()

        # extract model text safely
        response_text = ollama_response.json().get("response", "").strip()

        if not response_text:
            raise HTTPException(
                status_code=500,
                detail="Model returned empty output"
            )

        response_json = json.loads(response_text)

        return {
            "status": "success",
            "response": response_json
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Verify model returned invalid JSON"
        )

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    





def _insert_triples(triples: list[dict]) -> dict:
    """MERGE all triples into Neo4j. No validation."""
    nodes_created = 0
    rels_created = 0

    with driver.session() as session:
        for t in triples:
            query = (
                f"MERGE (a:{t['source_type']} {{name: $source}}) "
                f"MERGE (b:{t['target_type']} {{name: $target}}) "
                f"MERGE (a)-[r:{t['relation']}]->(b)"
            )
            summary = session.run(
                query,
                source=t["source"],
                target=t["target"]
            ).consume()

            nodes_created += summary.counters.nodes_created
            rels_created += summary.counters.relationships_created

    return {
        "nodes_created": nodes_created,
        "relationships_created": rels_created,
        "triples_processed": len(triples),
    }


# ── Request model ─────────────────────────────────────────────────────────────

class ChapterRequest(BaseModel):
    text: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/hacks/validate-chapter")
def validate_chapter(req: ChapterRequest):
    """
    Read-only. Extract triples, traverse Neo4j per-triple,
    fail-fast on first conflict, return narrator report.
    """
    triples = _extract_triples(req.text)

    with driver.session() as session:
        for t in triples:
            issue = _check_triple_conflict(session, t)
            if issue:
                # Narrator LLM call
                user_content = f"DETECTED ISSUES:\n- {issue}\n"
                query = f"{NARRATOR_PROMPT}\n\n{user_content}"
                response = requests.post(
                    LLM_ENDPOINT,
                    headers={"Content-Type": "application/json"},
                    json={
                        "query": query,
                        "model_name": LLM_MODEL,
                        "stream": False
                    }
                )
                response.raise_for_status()
                report = response.json().get("response", "").strip()
                return {"consistent": False, "report": report}

    return {"consistent": True}


@app.post("/hacks/insert-chapter")
def insert_chapter(req: ChapterRequest):
    """
    Write-only. Extract triples, MERGE into Neo4j, return counts.
    """
    triples = _extract_triples(req.text)
    return _insert_triples(triples)

class suggestor(BaseModel):
    input:str


@app.post("/hacks/suggest")
def suggest(payload:suggestor):
    prompt = f"""
    You are a writer's creative assistant helping overcome writer's block.

    Your task:
    Analyze the author's current work and suggest POSSIBLE DIRECTIONS the story could take next.

    IMPORTANT:
    - Do NOT write scenes.
    - Do NOT continue the story.
    - Only suggest high-level narrative directions in which the story could possibly take a turn to.
    - Each suggestion must be short (1–2 sentences maximum).

    OUTPUT RULES:
    - Return STRICT VALID JSON only.
    - No markdown.
    - No explanations outside JSON.
    - Provide AT MOST 3 suggestions.
    - If fewer than 3 strong ideas exist, return fewer.

    JSON FORMAT:
    {{
    "suggestions": [
        "idea 1",
        "idea 2",
        "idea 3"
    ]
    }}
    The authors last response was {payload.input}
    """
    try:
        response = call_llm(
            prompt=prompt,
            model_name="gpt-oss:20b-cloud",
            stream=False
        )

        return {
            "status": "success",
            "response": response
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

def login():
    global cursor,conn
    try:
        conn = mysql.connector.connect(
    host="127.0.0.1",
    user="apiuser",
    password="StrongPassword123!",
    database="hacks"
    )

        cursor = conn.cursor(dictionary=True)
        print('Connection to DB successfull')
        return conn
    except mysql.connector.Error as e:
        print('Error: ', e)

class getdata(BaseModel):
    login_id:str
    password:str
    mode:str

@app.post("/hacks/get_data")
def call_data(payload:getdata):
    conn = None
    cursor = None

    try:
        conn = login()
        cursor = conn.cursor(dictionary=True)

        # ⭐ check if user already exists
        check_query = """
        SELECT id, mode, summary
        FROM login_info
        WHERE login = %s AND password_hash = %s
        """
        cursor.execute(check_query, (payload.login_id, payload.password))
        existing_user = cursor.fetchone()

        if existing_user:
            existing_mode = existing_user["mode"]
            summary = existing_user["summary"]

            if existing_mode == payload.mode:
                return {"result":False,"summary":"Data already upto the mark"}
            else:
                return {"result":True,"summary":summary}
            
    except HTTPException:
        raise

    except Exception as e:
        print("ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()



class fetchsummary(BaseModel):
    login_id:str
    password:str
    summary:str
    mode:str


class fetchsummary(BaseModel):
    login_id: str
    session_id: str
    summary: str
    mode: str


@app.post("/mindicator/nativesummary")
def native_summary(payload: fetchsummary):

    conn = None
    cursor = None

    try:
        conn = login()
        cursor = conn.cursor(dictionary=True)

        check_query = """
        SELECT mode, chats
        FROM customer_data
        WHERE login_id = %s AND session_id = %s
        """

        cursor.execute(check_query, (payload.login_id, payload.session_id))
        existing_user = cursor.fetchone()

        if existing_user:

            previous_mode = existing_user["mode"]
            old_chats = existing_user["chats"] or ""

            # ⭐ same device → append chats
            if previous_mode == payload.mode:
                new_chats = old_chats + " " + payload.summary

            # ⭐ device changed → wipe chats
            else:
                new_chats = payload.summary

            update_query = """
            UPDATE customer_data
            SET chats = %s,
                summary = CONCAT(IFNULL(summary,''), ' ', %s),
                mode = %s
            WHERE login_id = %s AND session_id = %s
            """

            cursor.execute(
                update_query,
                (
                    new_chats,
                    payload.summary,
                    payload.mode,
                    payload.login_id,
                    payload.session_id
                )
            )

        else:

            insert_query = """
            INSERT INTO customer_data
            (login_id, session_id, mode, chats, summary)
            VALUES (%s, %s, %s, %s, %s)
            """

            cursor.execute(
                insert_query,
                (
                    payload.login_id,
                    payload.session_id,
                    payload.mode,
                    payload.summary,
                    payload.summary
                )
            )

        conn.commit()

        return {"status": "success"}

    except HTTPException:
        raise

    except Exception as e:
        print("ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

class PreviousContext(BaseModel):
    user_query: str
    ai_response: str


class StoryPayload(BaseModel):
    user_qry : str
    braindump: str
    genre: Dict
    style: Dict
    characters: List
    synopsis: str
    worldbuilding: str
    outline: str
    chapters: List
    previous_context: PreviousContext


def build_prompt(payload: StoryPayload) -> str:
    current_query = payload.user_qry

    return f"""
You are an AI assistant answering a user's CURRENT question.

Answer the current question using all available context.
Do NOT continue the story unless explicitly asked.
Be precise and concise.

====================
CURRENT USER QUESTION
====================
{current_query}

====================
PREVIOUS CONTEXT
====================
Previous User Question:
{payload.previous_context.user_query}

Previous AI Response:
{payload.previous_context.ai_response}

====================
STORY & WORLD CONTEXT
====================
Braindump:
{payload.braindump}

Genre:
{payload.genre}

Style:
{payload.style}

Characters:
{payload.characters}

Synopsis:
{payload.synopsis}

Worldbuilding:
{payload.worldbuilding}

Outline:
{payload.outline}

Chapters:
{payload.chapters}

====================
TASK
====================
Answer ONLY the current user question.
Keep the response short and relevant.
"""


# -------------------- API Endpoint --------------------

@app.post("/hacks/convo")
def generate_story(payload: StoryPayload):
    try:
        prompt = build_prompt(payload)

        response = call_llm(
            prompt=prompt,
            model_name="gpt-oss:20b-cloud",
            stream=False
        )

        return {
            "status": "success",
            "static_query": payload.user_qry,
            "response": response
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get('/hacks/health')
async def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'service': 'payment-gateway'}

class summar(BaseModel):
    chapters:str

@app.post("/hacks/summarize")
def summarizer(payload:summar):
    prompt=f"""
        Summarize this without loosing a single sense of logic. Maintain the entire logic, characters and the main theme.
        The text is: {payload.chapters}
    """
    summarized_text = call_llm(
            prompt=prompt,
            model_name="gpt-oss:120b-cloud",
            stream=False
        )

    summarized_json={
        "summary":summarized_text
    }

    return summarized_json


class councilRequest(BaseModel):
    input: str

@app.post("/hacks/council")
def judge(payload: councilRequest):
    answer={}

    try:
        EDITOR_PROMPT_TEMPLATE = f"""
            You are an editor performing a FAST TRIAGE pass on the following script.

            Your job is NOT to rewrite or critique style.
            Your job is to DETECT RISK FLAGS, IDENTIFY WHERE they occur in the script,
            and recommend which specialist models (if any) should review the script further.

            Additionally:
            If — and ONLY IF — the story strongly suggests that a previously deceased or removed character
            being alive again could meaningfully improve narrative clarity or continuity,
            you may flag it. This is OPTIONAL. Do NOT invent such cases.

            Analyze the script and return STRICT JSON with the following schema:

            {{
            "unresolved_promises": true | false,
            "entity_inconsistencies": true | false,
            "tone_instability": true | false,
            "logic_gaps": true | false,
            "missing_context": true | false,

            "revival_suggestion": true | false,
            "revival_note": "short optional note explaining why a previously lost/dead character might need reconsideration, or empty string",

            "risk_flags": [
                {{
                "section": "short location identifier such as 'opening', 'paragraph 3', 'midpoint', 'final section'",
                "risk_type": "unresolved_promise | entity_inconsistency | tone_instability | logic_gap | missing_context",
                "severity": "low | medium | high"
                }}
            ],

            "recommended_models": [],

            "notes": "one short sentence per triggered flag, or empty string"
            }}

            Rules for revival_suggestion:
            - Default MUST be false
            - Only set to true if the script clearly implies that a previously removed/dead character
            creates structural problems that could be resolved by their presence
            - Do NOT speculate or invent characters
            - Do NOT force this field to trigger

            Rules for risk_flags:
            - Only include entries when a real risk is detected
            - Use SHORT location identifiers, not long quotes from the script
            - Prefer structural markers like:
            "opening", "early section", "paragraph 2", "midpoint", "late section", "ending"
            - Do NOT invent sections that do not exist

            Rules for recommended_models:
            - Add "director" ONLY if there are issues with sequencing, structure, flow, or context continuity
            - Add "dop" ONLY if scene descriptions are weak, unclear, visually inconsistent, or lack spatial grounding
            - Do NOT invent other model names
            - If no specialist review is needed, return an empty list

            General Rules:
            - Do NOT suggest fixes
            - Do NOT explain reasoning
            - Do NOT return markdown
            - Do NOT return anything except valid JSON

            SCRIPT:
            <<<
            {payload.input}
            >>>
            """

        editor_response = call_llm(
            prompt=EDITOR_PROMPT_TEMPLATE,
            model_name="gpt-oss:120b-cloud",
            stream=False
        )

        editor_response_json = json.loads(editor_response)

        editor_response_json["recommended_models"].append("audience")

        sections = extract_audience_sections(
            payload.input,
            editor_response_json["notes"]
            )

        for i in editor_response_json["recommended_models"]:
            if i == "director":
                answer["director"]=director(payload.input,editor_response_json["notes"])
            elif i == "dop":
                answer["dop"]=dop(payload.input,editor_response_json["notes"])
        
        positive_result = audience_positive(sections)
        negative_result = audience_negative(sections)

        answer["audience"] = audience(positive_result, negative_result)

        return {
            "role": "editor",
            "verdict": editor_response_json,
            "answer":answer
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Editor model returned invalid JSON"
        )

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))

class EditorRequest(BaseModel):
    input: str

def call_llm(prompt: str, model_name: str, stream: bool = False):
    response = requests.post(
        "https://godfatherpersonalcomputer.shop/asklaptop",
        json={
            "query": prompt,
            "model_name": model_name,
            "stream": stream
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()

@app.post("/hacks/editor")
def editor_judge(payload: EditorRequest):
    input_text = payload.input
    EDITOR_PROMPT_TEMPLATE = f"""
        You are an editor performing a FAST TRIAGE pass on the following script.

        Your job is NOT to rewrite or critique style.
        Your job is to DETECT RISK FLAGS only.

        Analyze the script and return STRICT JSON with the following schema:

        {{
        "unresolved_promises": true | false,
        "entity_inconsistencies": true | false,
        "tone_instability": true | false,
        "logic_gaps": true | false,
        "missing_context": true | false,
        "notes": "one short sentence per triggered flag, or empty string"
        }}

        Rules:
        - Do NOT suggest fixes
        - Do NOT explain reasoning
        - Do NOT return markdown
        - Do NOT return anything except valid JSON

        SCRIPT:
        <<<
        {input_text}
        >>>
        """   

    try:
        prompt = EDITOR_PROMPT_TEMPLATE

        editor_response = call_llm(
            prompt=prompt,
            model_name="gpt-oss:120b-cloud",   # you can swap models later
            stream=False
        )

        editor_response_json=json.loads(editor_response)

        return {
            "role": "editor",
            "verdict": editor_response_json
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))
    
def extract_audience_sections(script: str, notes: str, max_chars=1500):
    """
    Extracts script portions likely relevant to audience perception.
    Currently naive: uses first, last, and any paragraph hinted by notes.
    """

    paragraphs = script.split("\n\n")

    selected = []

    # Always include opening and ending
    if paragraphs:
        selected.append(paragraphs[0])
        if len(paragraphs) > 1:
            selected.append(paragraphs[-1])

    # Heuristic: include paragraphs containing keywords from notes
    keywords = [w for w in notes.split() if len(w) > 4]

    for p in paragraphs:
        if any(k.lower() in p.lower() for k in keywords):
            selected.append(p)

    # Deduplicate and trim
    unique = list(dict.fromkeys(selected))
    joined = "\n\n".join(unique)

    return joined[:max_chars]

def director(input,remarks):

    DIRECTOR_PROMPT_TEMPLATE = """
        You are a DIRECTOR reviewing a script for STRUCTURE and FLOW.

        Your responsibility:
        - Verify that the script progresses logically from start to end
        - Ensure context is not lost between sections
        - Detect abrupt jumps, missing transitions, or broken narrative flow
        - Detect sections that contradict earlier established context

        You are NOT allowed to:
        - Rewrite the script
        - Suggest stylistic changes
        - Judge tone or emotional quality unless it breaks flow
        - Fix issues

        Return STRICT JSON with the following schema:

        {{
        "flow_consistent": true | false,
        "context_drift_detected": true | false,
        "structural_issues": true | false,
        "issues": [
            {{
            "section": "short identifier or description",
            "problem": "one sentence describing the structural or flow issue"
            }}
        ]

        Rules:
        - Only report issues if they break structure or context
        - If no issues exist, return an empty issues array
        - Do NOT include explanations outside JSON
        - Do NOT include markdown

        SCRIPT:
        <<<
        {script}
        >>>

        EDITOR REMARKS (for context only, do not repeat):
        <<<
        {remarks}
        >>>
        """
    try:
        prompt = DIRECTOR_PROMPT_TEMPLATE.format(
            script=input,
            remarks=remarks
        )

        response = call_llm(
            prompt=prompt,
            model_name="gpt-oss:20b-cloud",
            stream=False
        )

        # Director MUST return JSON
        response_json = json.loads(response)

        return {
            "role": "director",
            "verdict": response_json
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Director model returned invalid JSON"
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=str(e)
        )
    
def dop(input, remarks):
    DOP_PROMPT_TEMPLATE = """
    You are a DIRECTOR OF PHOTOGRAPHY (DOP) reviewing a script for VISUAL CLARITY
    and SCENE DESCRIPTION QUALITY.

    Your responsibility:
    - Check whether scenes are visually clear and well grounded
    - Detect vague, abstract, or hard-to-visualize descriptions
    - Detect inconsistent or confusing spatial layouts
    - Detect scenes that lack sensory detail where it is expected

    You are NOT allowed to:
    - Rewrite the script
    - Judge story structure or sequencing
    - Fix issues
    - Add creative suggestions

    Return STRICT JSON with the following schema:

    {{
    "visual_clarity_good": true | false,
    "scene_definition_issues": true | false,
    "issues": [
        {{
        "section": "short identifier or description",
        "problem": "one sentence describing the visual or scene clarity issue"
        }}
    ]
    }}

    Rules:
    - Only report issues related to visual or scene clarity
    - If no issues exist, return an empty issues array
    - Do NOT include explanations outside JSON
    - Do NOT include markdown

    SCRIPT:
    <<<
    {script}
    >>>

    EDITOR REMARKS (for context only, do not repeat):
    <<<
    {remarks}
    >>>
    """

    try:
        prompt = DOP_PROMPT_TEMPLATE.format(
            script=input,
            remarks=remarks
        )

        response = call_llm(
            prompt=prompt,
            model_name="gpt-oss:20b-cloud",  # smaller model
            stream=False
        )

        response_json = json.loads(response)

        return {
            "role": "dop",
            "verdict": response_json
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="DOP model returned invalid JSON"
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=str(e)
        )

def audience_positive(sections):
    POSITIVE_PROMPT_TEMPLATE = """
    You are an AUDIENCE MEMBER identifying POSITIVE engagement factors.

    For each section provided:
    - Identify why a typical user would LIKE or ENGAGE with it
    - Base your reasoning strictly on the content provided

    Return STRICT JSON with the following schema:

    {{
    "positives": [
        {{
        "section": "section identifier",
        "appeal_factors": ["clarity", "emotion", "imagery", "pace", "relatability"],
        "why_it_works": "short explanation grounded in the text"
        }}
    ]
    }}

    Rules:
    - Do NOT invent new sections
    - Do NOT critique
    - Do NOT suggest fixes
    - Do NOT include markdown

    SECTIONS:
    <<<
    {sections}
    >>>
    """

    try:
        prompt = POSITIVE_PROMPT_TEMPLATE.format(sections=sections)

        # 🔥 Local Ollama call
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "audience-agent:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        ollama_response.raise_for_status()

        # Ollama wraps output inside "response"
        response_text = ollama_response.json()["response"]

        return json.loads(response_text)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Audience positive model returned invalid JSON"
        )

def audience_negative(sections):
    NEGATIVE_PROMPT_TEMPLATE = """
    You are an AUDIENCE MEMBER identifying NEGATIVE engagement factors.

    For each section provided:
    - Identify why a typical user may DISLIKE, feel confused, or disengage
    - Base your reasoning strictly on the content provided

    Return STRICT JSON with the following schema:

    {{
    "negatives": [
        {{
        "section": "section identifier",
        "friction_factors": ["confusion", "abruptness", "weak imagery", "low stakes"],
        "why_it_fails": "short explanation grounded in the text"
        }}
    ]
    }}

    Rules:
    - Do NOT invent new sections
    - Do NOT suggest fixes
    - Do NOT repeat positive points
    - Do NOT include markdown

    SECTIONS:
    <<<
    {sections}
    >>>
    """

    try:
        prompt = NEGATIVE_PROMPT_TEMPLATE.format(sections=sections)

        # 🔥 Call local Ollama
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "audience-agent:latest",
                "prompt": prompt,
                "stream": False,
                "format": "json"   # forces structured output (important for GGUF)
            },
            timeout=300
        )

        ollama_response.raise_for_status()

        response_text = ollama_response.json()["response"]

        return json.loads(response_text)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Audience negative model returned invalid JSON"
        )
    
class chatter(BaseModel):
    text: str


@app.post("/hacks/chatbot")
def chatbot(payload: chatter):

    prompt = f"""
        You are a PROFESSIONAL ASSISTANT named OSCAR whose role is to generate accurate, grounded responses.
        Your parent app is called OSCARIFY.AI which is used to create/enhance scripts and books. 
        Your parent app can help authors write award winning scripts by helping them in enhancing their writing perspective rather than replacing their ideas.

        Rules:
        - Use ONLY the information present in the provided DATA.
        - Do NOT invent facts or external knowledge.
        - Keep responses concise and professional.
        - Dont refrain from answering any general questions.

        Return STRICT JSON ONLY in this format:

        {{
        "answer": "your grounded response"
        }}

        DATA:
        <<<
        {payload.text}
        >>>
"""

    try:
        response = call_llm(
            prompt=prompt,
            model_name="gpt-oss:20b-cloud",
            stream=False
        )

        # 🔥 Guard against empty output (your current code assumes success)
        if not response or not response.strip():
            raise HTTPException(
                status_code=500,
                detail="Model returned empty output"
            )

        response_json = json.loads(response)

        return {
            "response": response_json
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Chatbot model returned invalid JSON"
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=str(e)
        )


def audience(positive, negative):
    SYNTHESIS_PROMPT_TEMPLATE = """
    You are synthesizing audience perception into an EXPLAINABLE SUMMARY.

    Given positive and negative audience feedback for the same sections:
    - Explain why users are likely to like or dislike each section
    - Balance both perspectives
    - Do NOT suggest fixes

    Return STRICT JSON with the following schema:

    {{
    "analysis": [
        {{
        "section": "section identifier",
        "why_users_like_it": ["reason"],
        "why_users_dislike_it": ["reason"],
        "net_effect": "short balanced summary",
        "confidence": 0.0
        }}
    ]
    }}

    Rules:
    - Ground all reasoning in provided inputs
    - Do NOT invent new issues
    - Do NOT include markdown

    POSITIVE FEEDBACK:
    <<<
    {positive}
    >>>

    NEGATIVE FEEDBACK:
    <<<
    {negative}
    >>>
    """

    try:
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            positive=json.dumps(positive),
            negative=json.dumps(negative)
        )

        # 🔥 Local Ollama call
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "audience-agent:latest",
                "prompt": prompt,
                "stream": False,
                "format": "json"   # helps prevent malformed JSON
            },
            timeout=300
        )

        ollama_response.raise_for_status()

        response_text = ollama_response.json()["response"]

        return json.loads(response_text)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Audience synthesis model returned invalid JSON"
        )
    
    
class SynopsisPayload(BaseModel):
    dump: str
    genre: str
    style: str
    characters: list = []

@app.post("/hacks/synopsis")
def synopsis(payload: SynopsisPayload):
    prompt = f"""
Write a concise third-person synopsis capturing conflict, goal, and stakes.
STRICTLY FOLLOW THE NAME OF THE CHARACTERS GIVEN, STYLE, GENRE and CORE.

Genre: {payload.genre}
Style: {payload.style}
Characters:{payload.characters}
Story Core:
{payload.dump}
"""
    return {"synopsis": call_llm(prompt, "gpt-oss:120b-cloud")}

class WorldbuildingPayload(BaseModel):
    dump: str
    genre: str
    style: str
    synopsis: str

@app.post("/hacks/worldbuilding")
def worldbuilding(payload: WorldbuildingPayload):
    prompt = f"""
    Write concise worldbuilding documentation in clean, valid Markdown suitable for direct rendering with react-markdown.

    Do NOT advance the plot.
    Do NOT narrate events or scenes.

    Use EXACTLY this structure:

    # Worldbuilding

    ## Core Rules

    Describe immutable laws governing the world using short bullet points.

    ## Culture & Society

    Describe beliefs, norms, and social hierarchy.

    ## Technology / Power Systems

    Explain how technology, magic, or power functions and its limitations.

    ## Political Structure

    Describe governing bodies, factions, or authority systems.

    ## Environmental Constraints

    Describe physical or cosmic limitations affecting the world.

    Rules:

    * Use valid Markdown only
    * Use "-" for bullet lists
    * Leave one blank line between sections
    * Keep paragraphs short
    * DO NOT generate tables
    * DO NOT include examples
    * DO NOT use HTML, emojis, or code blocks

    Genre: {payload.genre}
    Style: {payload.style}

    Core:
    {payload.dump}

    Context:
    {payload.synopsis}

    Return ONLY Markdown.
"""
    return {"worldbuilding": call_llm(prompt, "gpt-oss:120b-cloud")}

class ChapterPayload(BaseModel):
    dump: str
    genre: str
    style: str
    synopsis: str
    worldbuilding: str
    chapternumber: int
    characters:list=[]

@app.post("/hacks/chapters")
def chapters(payload: ChapterPayload):
    chapters = []
    for i in range(payload.chapternumber):
        prompt = f"""
            Generate a structured chapter outline in clean, valid Markdown suitable for direct rendering with react-markdown.

            The synopsis of the story is: {payload.synopsis}

            Do NOT resolve the main conflict.
            Do NOT write prose or full scenes.

            Use EXACTLY this structure:

            # Chapter {i+1} Outline

            ## Chapter Goal

            Brief description of the narrative objective.

            ## Key Plot Beats

            * List major story developments in order.
            * Each point should be concise.

            ## Stakes

            Explain what can be gained or lost in this chapter.

            ## Setup for Next Chapter

            Describe narrative momentum carried forward without resolving the central conflict.

            Rules:

            * Use valid Markdown only
            * Use "-" for bullet lists
            * Leave one blank line between sections
            * Keep points concise
            * DO NOT generate tables
            * DO NOT include examples
            * DO NOT use HTML, emojis, or code blocks

            Return ONLY Markdown.
            The list of characters is as follows {payload.characters}.

            """
        chapters.append(call_llm(prompt, "gpt-oss:120b-cloud"))
    return {"chapters": chapters}