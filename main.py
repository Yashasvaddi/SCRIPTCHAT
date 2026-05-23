from pydantic import BaseModel
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Header
from fastapi.middleware.cors import CORSMiddleware
import json

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        Analyze the script and return STRICT JSON with the following schema:

        {{
        "unresolved_promises": true | false,
        "entity_inconsistencies": true | false,
        "tone_instability": true | false,
        "logic_gaps": true | false,
        "missing_context": true | false,

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

        response = call_llm(
            prompt=prompt,
            model_name="gpt-oss:20b-cloud",  # small perception model
            stream=False
        )

        return json.loads(response)

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

        response = call_llm(
            prompt=prompt,
            model_name="gpt-oss:20b-cloud",  # same or smaller model
            stream=False
        )

        return json.loads(response)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Audience negative model returned invalid JSON"
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

        response = call_llm(
            prompt=prompt,
            model_name="gpt-oss:20b-cloud",  # very small synthesis model
            stream=False
        )

        return json.loads(response)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Audience synthesis model returned invalid JSON"
        )

