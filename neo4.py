import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from neo4j import GraphDatabase
from pydantic import BaseModel
from typing import Optional
import requests

app=FastAPI()

load_dotenv()
NEO4J_URI = "url"
NEO4J_USER = "user"
NEO4J_PASSWORD = "password"
LLM_ENDPOINT = "https://godfatherpersonalcomputer.shop/asklaptop"
LLM_MODEL = "gpt-oss:120b-cloud"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

_cached_snapshot: Optional[dict] = None


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
    # Strip markdown fences if the model wraps JSON in 
    raw = raw.strip()
    if raw.startswith(""):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)
    return [
        t for t in data.get("triples", [])
        if t.get("source_type") in ALLOWED_LABELS
        and t.get("target_type") in ALLOWED_LABELS
        and t.get("relation") in ALLOWED_RELATIONS
    ]


def _upsert_graph(triples: list[dict], idx: dict, snapshot: dict) -> dict:
    nodes_created = 0
    rels_created = 0

    with driver.session() as session:
        for t in triples:

            # 🧠 Run analysis ONLY for this triple
            analysis = _analyse_against_snapshot(snapshot, [t])
            issues = analysis["all_conflict_issues"]

            # 🚨 IF CONFLICT EXISTS → RETURN ENGLISH REASON IMMEDIATELY
            if issues:
                user_content = f"""DETECTED ISSUES:
{chr(10).join(f'- {issue}' for issue in issues)}
"""

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
                raw = response.json().get("response", "").strip()

                # ⛔ STOP UPSERT AND RETURN EXPLANATION
                return {
                    "status": "conflict",
                    "conflict_report": raw,
                    "conflicting_triple": t
                }

            # ✅ SAFE → WRITE TO GRAPH
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
        "status": "success",
        "nodes_created": nodes_created,
        "relationships_created": rels_created
    }

class ChapterRequest(BaseModel):
    text: str


@app.post("/hacks/process-chapter")
def process_chapter(req: ChapterRequest):
    global _cached_snapshot

    triples = _extract_triples(req.text)

    snapshot = _fetch_snapshot()
    idx = _build_indexes(snapshot)

    result = _upsert_graph(triples, idx, snapshot)

    # 🔥 If conflict happened, return immediately
    if result.get("status") == "conflict":
        return result

    _cached_snapshot = _fetch_snapshot()

    result["triples_extracted"] = len(triples)
    return result


@app.get("/hacks/snapshot")
def get_snapshot():
    """Return the cached snapshot, or fetch fresh if none cached yet."""
    global _cached_snapshot
    if _cached_snapshot is None:
        _cached_snapshot = _fetch_snapshot()
    return _cached_snapshot


@app.post("/hacks/refresh-snapshot")
def refresh_snapshot():
    """Force-refresh the cached snapshot from Neo4j."""
    global _cached_snapshot
    _cached_snapshot = _fetch_snapshot()
    return {"status": "refreshed", "entities": len(_cached_snapshot["entities"]), "relationships": len(_cached_snapshot["relationships"])}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTE 2 — snapshot graph before & after, diff what changed
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_snapshot() -> dict:
    """Inline of graph_export.py — fetch current graph from Neo4j as a snapshot dict."""
    with driver.session() as session:
        node_records = session.run(
            "MATCH (n) WHERE n.name IS NOT NULL RETURN labels(n)[0] AS type, n.name AS name"
        ).data()
        rel_records = session.run(
            "MATCH (a)-[r]->(b) WHERE a.name IS NOT NULL AND b.name IS NOT NULL "
            "RETURN a.name AS source, type(r) AS relation, b.name AS target"
        ).data()

    seen_e = set()
    entities = []
    for r in node_records:
        k = (r["name"], r["type"])
        if k not in seen_e:
            seen_e.add(k)
            entities.append({"name": r["name"], "type": r["type"]})

    seen_r = set()
    relationships = []
    for r in rel_records:
        k = (r["source"], r["relation"], r["target"])
        if k not in seen_r:
            seen_r.add(k)
            relationships.append({"source": r["source"], "relation": r["relation"], "target": r["target"]})

    return {
        "entities": sorted(entities, key=lambda e: e["name"].lower()),
        "relationships": sorted(relationships, key=lambda r: (r["source"].lower(), r["relation"], r["target"].lower())),
    }


# ── Semantic polarity groups ─────────────────────────────────────────────────
# Instead of enumerating every pair, group relations by sentiment.
# Any POSITIVE relation automatically conflicts with any NEGATIVE relation
# on the same entity pair (either direction). No manual keys needed.

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
# Structural exclusives: relations that can't coexist on the same pair
STRUCTURAL_CONFLICTS = {
    "LEADS": {"SERVES_UNDER", "FOLLOWS"},
    "CONTROLS": {"ESCAPES_FROM", "REBELS_AGAINST"},
    "RULES": {"ESCAPES_FROM", "REBELS_AGAINST", "DEFEATS"},
    "CAPTURES": {"ESCAPES_FROM", "RESCUES"},
    "ESCAPES_FROM": {"CAPTURES", "SERVES", "FOLLOWS"},
    "DESTROYS": {"REPAIRS", "CREATES"},
    "CREATES": {"DESTROYS"},
}


def _get_polarity(rel: str) -> str:
    if rel in POSITIVE_RELS:
        return "POSITIVE"
    if rel in NEGATIVE_RELS:
        return "NEGATIVE"
    return "NEUTRAL"


def _relations_conflict(rel_a: str, rel_b: str) -> bool:
    """Return True if two relations semantically contradict each other."""
    # Polarity clash: positive vs negative = conflict
    pol_a, pol_b = _get_polarity(rel_a), _get_polarity(rel_b)
    if pol_a != "NEUTRAL" and pol_b != "NEUTRAL" and pol_a != pol_b:
        return True
    # Structural clash: explicit incompatible pairs
    if rel_b in STRUCTURAL_CONFLICTS.get(rel_a, set()):
        return True
    if rel_a in STRUCTURAL_CONFLICTS.get(rel_b, set()):
        return True
    return False

# ── State contradiction pairs ────────────────────────────────────────────────
# If target entity has been DESTROYED, it cannot later be VISITED, ENTERED, etc.
STATE_TERMINAL_RELS = {"DESTROYS", "KILLS"}  # these make the target "gone"
STATE_REQUIRES_ALIVE = {
    "VISITS", "ENTERS", "EXPLORES", "ARRIVES_AT", "RETURNS_TO",
    "GUARDS", "PROTECTS", "LEADS", "CONTROLS", "RULES", "COMMANDS",
    "FIGHTS", "ATTACKS", "CAPTURES", "SERVES", "FOLLOWS",
    "TRAVELS_TO", "WORKS_WITH", "MENTORS",
}

# ── Exclusive role relations ─────────────────────────────────────────────────
# Only one entity should hold these over a given target at the same time
EXCLUSIVE_ROLES = {"LEADS", "CONTROLS", "RULES", "COMMANDS", "LEADS_ORGANIZATION"}

# ── High-impact relations ────────────────────────────────────────────────────
DOMINANT_RELATIONS = {"LEADS", "CONTROLS", "DESTROYS", "COMMANDS", "TRIGGERS", "KILLS"}


def _build_indexes(snapshot: dict) -> dict:
    """Pre-index snapshot for O(n) comparison in _analyse_against_snapshot."""
    entity_types = {e["name"]: e["type"] for e in snapshot["entities"]}
    rels_set = {(r["source"], r["relation"], r["target"]) for r in snapshot["relationships"]}

    # (source, target) → set of rels  AND  (target, source) → set of rels
    pair_rels: dict[tuple, set[str]] = {}
    # relation → set of (source, target)
    by_relation: dict[str, set[tuple]] = {}
    # target → set of (source, relation)
    by_target: dict[str, set[tuple]] = {}
    # entities that have been destroyed/killed (terminal state)
    dead_entities: set[str] = set()

    for r in snapshot["relationships"]:
        s, rel, tg = r["source"], r["relation"], r["target"]
        pair_rels.setdefault((s, tg), set()).add(rel)
        pair_rels.setdefault((tg, s), set())  # ensure reverse exists
        by_relation.setdefault(rel, set()).add((s, tg))
        by_target.setdefault(tg, set()).add((s, rel))
        if rel in STATE_TERMINAL_RELS:
            dead_entities.add(tg)

    return {
        "entity_types": entity_types,
        "rels_set": rels_set,
        "pair_rels": pair_rels,
        "by_relation": by_relation,
        "by_target": by_target,
        "dead_entities": dead_entities,
    }


def _analyse_against_snapshot(snapshot: dict, triples: list[dict]) -> dict:
    """
    Compare incoming triples against the existing graph snapshot.
    Nothing is written to Neo4j — pure analysis only.
    """
    idx = _build_indexes(snapshot)
    existing_entity_types = idx["entity_types"]
    existing_rels = idx["rels_set"]
    pair_rels = idx["pair_rels"]
    by_relation = idx["by_relation"]
    by_target = idx["by_target"]
    dead_entities = idx["dead_entities"]

    new_entities = []
    type_conflicts = []
    new_relationships = []
    already_known = []
    semantic_conflicts = []
    state_conflicts = []
    exclusivity_conflicts = []
    causal_conflicts = []
    dominant_warnings = []

    seen_new_entities: set[tuple] = set()
    seen_conflicts: set[str] = set()  # dedup conflict messages

    for t in triples:
        src, rel, tgt = t["source"], t["relation"], t["target"]

        # ── Entity label checks ────────────────────────────────────────────
        for name, etype in [(src, t["source_type"]), (tgt, t["target_type"])]:
            if name in existing_entity_types:
                if existing_entity_types[name] != etype:
                    type_conflicts.append({
                        "entity": name,
                        "existing_type": existing_entity_types[name],
                        "incoming_type": etype,
                        "issue": f"Role shift: '{name}' was a {existing_entity_types[name]} but is now described as {etype}",
                    })
            else:
                key = (name, etype)
                if key not in seen_new_entities:
                    seen_new_entities.add(key)
                    new_entities.append({"name": name, "type": etype})

        # ── Relationship known/new ─────────────────────────────────────────
        rel_key = (src, rel, tgt)
        if rel_key in existing_rels:
            already_known.append({"source": src, "relation": rel, "target": tgt})
        else:
            new_relationships.append({"source": src, "relation": rel, "target": tgt})

        # ── Semantic contradiction (polarity + structural, both dirs) ─────
        for pair in [(src, tgt), (tgt, src)]:
            existing_on_pair = pair_rels.get(pair, set())
            for existing_rel in existing_on_pair:
                if _relations_conflict(rel, existing_rel):
                    msg = (f"'{pair[0]}' was previously '{existing_rel}' with '{pair[1]}', "
                        f"but the new chapter introduces '{rel}' — these contradict each other.")
                    if msg not in seen_conflicts:
                        seen_conflicts.add(msg)
                        semantic_conflicts.append({
                            "entity_a": pair[0], "entity_b": pair[1],
                            "existing_relation": existing_rel, "incoming_relation": rel,
                            "issue": msg,
                        })

        # ── State conflict: acting on dead/destroyed entity ────────────────
        if rel in STATE_REQUIRES_ALIVE:
            if tgt in dead_entities:
                msg = f"'{tgt}' was previously destroyed/killed, but the new chapter has '{src}' performing '{rel}' on it."
                if msg not in seen_conflicts:
                    seen_conflicts.add(msg)
                    state_conflicts.append({"entity": tgt, "incoming_relation": rel, "by": src, "issue": msg})
            if src in dead_entities:
                msg = f"'{src}' was previously destroyed/killed, but the new chapter has it performing '{rel}' on '{tgt}'."
                if msg not in seen_conflicts:
                    seen_conflicts.add(msg)
                    state_conflicts.append({"entity": src, "incoming_relation": rel, "target": tgt, "issue": msg})

        # ── Exclusive role conflict: two leaders/controllers of same target ─
        if rel in EXCLUSIVE_ROLES:
            existing_holders = by_relation.get(rel, set())
            for prev_src, prev_tgt in existing_holders:
                if prev_tgt == tgt and prev_src != src:
                    msg = f"'{prev_src}' already '{rel}' '{tgt}', but now '{src}' also claims to — only one should."
                    if msg not in seen_conflicts:
                        seen_conflicts.add(msg)
                        exclusivity_conflicts.append({
                            "existing_holder": prev_src, "incoming_holder": src,
                            "relation": rel, "target": tgt, "issue": msg,
                        })

        # ── Causal conflict: two different causes for same target event ────
        if rel in ("TRIGGERS", "CAUSES"):
            existing_causes = by_target.get(tgt, set())
            for prev_src, prev_rel in existing_causes:
                if prev_rel in ("TRIGGERS", "CAUSES") and prev_src != src:
                    msg = f"'{prev_src}' already '{prev_rel}' '{tgt}', but now '{src}' also claims to cause it."
                    if msg not in seen_conflicts:
                        seen_conflicts.add(msg)
                        causal_conflicts.append({
                            "existing_cause": prev_src, "incoming_cause": src,
                            "event": tgt, "issue": msg,
                        })

        # ── Dominant new entity warning ────────────────────────────────────
        if rel in DOMINANT_RELATIONS and src not in existing_entity_types:
            dominant_warnings.append({
                "entity": src, "relation": rel, "target": tgt,
                "issue": f"New entity '{src}' immediately '{rel}' '{tgt}' — high-impact introduction.",
            })

    # Merge all conflict issues into one flat list for the narrator
    all_conflicts = (
        [c["issue"] for c in semantic_conflicts]
        + [c["issue"] for c in state_conflicts]
        + [c["issue"] for c in exclusivity_conflicts]
        + [c["issue"] for c in causal_conflicts]
        + [c["issue"] for c in type_conflicts]
        + [w["issue"] for w in dominant_warnings]
    )

    return {
        "new_entities": sorted(new_entities, key=lambda e: e["name"]),
        "new_relationships": sorted(new_relationships, key=lambda r: (r["source"], r["relation"], r["target"])),
        "already_known_relationships": already_known,
        "type_conflicts": type_conflicts,
        "semantic_conflicts": semantic_conflicts,
        "state_conflicts": state_conflicts,
        "exclusivity_conflicts": exclusivity_conflicts,
        "causal_conflicts": causal_conflicts,
        "dominant_warnings": dominant_warnings,
        "all_conflict_issues": all_conflicts,
        "summary": {
            "new_entities_count": len(new_entities),
            "new_relationships_count": len(new_relationships),
            "already_known_count": len(already_known),
            "type_conflict_count": len(type_conflicts),
            "semantic_conflict_count": len(semantic_conflicts),
            "state_conflict_count": len(state_conflicts),
            "exclusivity_conflict_count": len(exclusivity_conflicts),
            "causal_conflict_count": len(causal_conflicts),
            "dominant_warning_count": len(dominant_warnings),
            "total_issues": len(all_conflicts),
        },
    }


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


@app.post("/hacks/what-changed")
def what_changed(req: ChapterRequest):
    """
    Read-only. Extracts triples from chapter text, diffs against Neo4j,
    and returns a single natural language continuity report.
    """
    global _cached_snapshot
    # Use cached snapshot if available, otherwise fetch once and cache
    if _cached_snapshot is None:
        _cached_snapshot = _fetch_snapshot()
    snapshot = _cached_snapshot

    triples = _extract_triples(req.text)
    diff = _analyse_against_snapshot(snapshot, triples)

    all_issues = diff["all_conflict_issues"]

    # Build user content focused on issues only — not the full graph
    user_content = f"""DETECTED ISSUES:
{chr(10).join(f'- {issue}' for issue in all_issues) or 'none'}
"""

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
    raw = response.json().get("response", "")
    return {"report": raw.strip()}
