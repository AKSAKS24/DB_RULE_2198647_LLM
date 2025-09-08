from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, json

# --- LLM ENV ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Need OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2198647 Assessment - Pure Suggestion Variant")

# --- Data Models ---
class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    start_line: Optional[int] = 0
    end_line: Optional[int] = 0
    code: Optional[str] = ""  # unused now
    findings: Optional[List[Finding]] = Field(default_factory=list)

# --- LLM PROMPT ---
SYSTEM_MSG = """
You are a senior ABAP expert. Output ONLY JSON as response.
For every provided payload .findings[], write a bullet point that:
- Uses the "suggestion" field as action text for correction
- If a "snippet" field is present and not empty, include the text inline (ABAP code or text) before or after the action as appropriate.
- Do not refer to or require code context outside "snippet".
- If a finding does not have a suggestion, skip it.
- Each bullet must use the finding's own suggestion.
- Do NOT omit any finding with a suggestion; all must be covered.
Return only JSON in this structure:
{
  "assessment": "<summary>",
  "llm_prompt": "<action bullets as text, each on a new line (with or without snippet as explained above)>"
}
""".strip()

USER_TEMPLATE = """
Unit metadata:
Program: {pgm_name}
Include: {inc_name}
Unit type: {unit_type}
Unit name: {unit_name}
Class implementation: {class_implementation}
Start line: {start_line}
End line: {end_line}

findings (JSON list of findings, each with all fields above, for 2198647 errors):
{findings_json}

Instructions:
1. Write a 1-paragraph assessment summarizing 2198647 risks in human language.
2. Write a llm_prompt field: for every finding with a non-empty suggestion, add a bullet point with
   - The action from the suggestion field
   - If the snippet field is present and not empty, include it alongside the action (before or after, as fits)
   - Leave out any finding with no suggestion.
Return valid JSON with:
{
  "assessment": "...",
  "llm_prompt": "..."
}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE)
])
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    # filter findings: only those with non-empty suggestion
    relevant_findings = [f for f in (unit.findings or []) if f.suggestion and f.suggestion.strip()]
    if not relevant_findings:
        return None
    findings_json = json.dumps([f.model_dump() for f in relevant_findings], ensure_ascii=False, indent=2)
    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name or "",
            "class_implementation": unit.class_implementation or "",
            "start_line": unit.start_line or 0,
            "end_line": unit.end_line or 0,
            "findings_json": findings_json,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

@app.post("/assess-2198647")
async def assess_note_2198647(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        # do not inspect code or generate findings, just consume as input
        llm_out = llm_assess_and_prompt(u)
        if not llm_out:
            continue
        prompt_out = llm_out.get("llm_prompt", "")
        if isinstance(prompt_out, list):
            prompt_out = "\n".join(str(x) for x in prompt_out if x is not None)
        obj = {
            "pgm_name": u.pgm_name,
            "inc_name": u.inc_name,
            "type": u.type,
            "name": u.name,
            "class_implementation": u.class_implementation,
            "start_line": u.start_line,
            "end_line": u.end_line,
            "assessment": llm_out.get("assessment", ""),
            "llm_prompt": prompt_out
        }
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "note": "2198647", "model": OPENAI_MODEL}