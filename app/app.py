from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, json, re

# --- LLM required env ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Need OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2198647 Assessment - System-Style Strict (Findings)")

# --- Patterns and obsolete rules ---
OBSOLETE_TABLES = {"VBUK": "VBAK", "VBUP": "VBAP"}
DECL_LENGTHENED = {"VBTYP": "VBTYPL"}
DECL_OBSOLETE = {"VBTYP_EXT"}

SQL_SELECT_BLOCK_RE = re.compile(
    r"\bSELECT\b(?P<select>.+?)\bFROM\b\s+(?P<table>\w+)(?P<rest>.*?)(?=(\bSELECT\b|$))",
    re.IGNORECASE | re.DOTALL,
)
JOIN_RE = re.compile(r"\bJOIN\s+(?P<table>\w+)", re.IGNORECASE)
DECLARATION_RE = re.compile(
    r"\b(?:TYPE|LIKE)\b\s+(?P<field>[A-Z0-9_]+)", re.IGNORECASE
)

# --- Data Models (system-style full, using findings as field name) ---
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
    code: Optional[str] = ""
    findings: Optional[List[Finding]] = Field(default_factory=list)

# --- Detection Logic: fills all system model fields as possible ---
def build_findings(unit: Unit) -> List[Finding]:
    code = unit.code or ""
    findings = []
    for stmt in SQL_SELECT_BLOCK_RE.finditer(code):
        table = stmt.group("table").upper()
        rest_text = stmt.group("rest")
        full_stmt = stmt.group(0).strip()
        if table in OBSOLETE_TABLES:
            findings.append(Finding(
                pgm_name=unit.pgm_name,
                inc_name=unit.inc_name,
                type=unit.type,
                name=unit.name,
                class_implementation=unit.class_implementation,
                issue_type="Obsolete Table",
                severity="HIGH",
                message=f"Table {table} is obsolete in S/4HANA.",
                suggestion=f"Replace {table} with {OBSOLETE_TABLES[table]} as per SAP Note 2198647.",
                snippet=full_stmt
            ))
        for jm in JOIN_RE.finditer(rest_text):
            jtable = jm.group("table").upper()
            if jtable in OBSOLETE_TABLES:
                findings.append(Finding(
                    pgm_name=unit.pgm_name,
                    inc_name=unit.inc_name,
                    type=unit.type,
                    name=unit.name,
                    class_implementation=unit.class_implementation,
                    issue_type="Obsolete Table (JOIN)",
                    severity="HIGH",
                    message=f"Join on table {jtable} is obsolete in S/4HANA.",
                    suggestion=f"Replace {jtable} with {OBSOLETE_TABLES[jtable]} as per SAP Note 2198647.",
                    snippet=jm.group(0).strip()
                ))
        if "VBTYP_EXT" in full_stmt.upper():
            findings.append(Finding(
                pgm_name=unit.pgm_name,
                inc_name=unit.inc_name,
                type=unit.type,
                name=unit.name,
                class_implementation=unit.class_implementation,
                issue_type="Obsolete Field",
                severity="HIGH",
                message="Obsolete field VBTYP_EXT is used.",
                suggestion="Remove usage of field VBTYP_EXT as per SAP Note 2198647.",
                snippet=full_stmt
            ))
    for m in DECLARATION_RE.finditer(code):
        fld = m.group("field").upper()
        line_text = m.group(0).strip()
        if fld in DECL_OBSOLETE:
            findings.append(Finding(
                pgm_name=unit.pgm_name,
                inc_name=unit.inc_name,
                type=unit.type,
                name=unit.name,
                class_implementation=unit.class_implementation,
                issue_type="Obsolete Declaration",
                severity="MEDIUM",
                message=f"Obsolete data element: {fld}.",
                suggestion=f"Remove usage of obsolete field {fld} as per SAP Note 2198647.",
                snippet=line_text
            ))
        if fld in DECL_LENGTHENED:
            findings.append(Finding(
                pgm_name=unit.pgm_name,
                inc_name=unit.inc_name,
                type=unit.type,
                name=unit.name,
                class_implementation=unit.class_implementation,
                issue_type="Lengthened Declaration",
                severity="MEDIUM",
                message=f"Data element {fld} lengthened.",
                suggestion=f"Use {DECL_LENGTHENED[fld]} for compatibility as per SAP Note 2198647.",
                snippet=line_text
            ))
    return findings

# --- LLM Prompt ---
SYSTEM_MSG = """
You are a senior ABAP expert. Output ONLY JSON as response.
For every provided payload .findings[].snippet,
write a bullet point that:
- Displays the exact offending code
- Explains the necessary action to fix the error using the provided .suggestion text (if available).
- Bullet points should contain both offending code snippet and the fix (no numbering or referencing like "snippet[1]": display the code inline).
- Do NOT omit any snippet; all must be covered, no matter how many there are.
- Only show actual ABAP code for each snippet with its specific action.
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

ABAP code context (optional):
{code}

findings (JSON list of findings, each with all fields above, for 2198647 errors):
{findings_json}

Instructions:
1. Write a 1-paragraph assessment summarizing 2198647 risks, in human language.
2. Write a llm_prompt field: for every finding, add a bullet point with
   - The exact code (snippet field)
   - The action required for correction (taken from suggestion field).
   - Do not compress, omit, or refer to them by index; always display code inline.

Return valid JSON with:
{{
  "assessment": "<paragraph>",
  "llm_prompt": "<action bullets>"
}}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE)
])
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    findings_json = json.dumps([f.model_dump() for f in (unit.findings or [])], ensure_ascii=False, indent=2)
    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name or "",
            "class_implementation": unit.class_implementation or "",
            "start_line": unit.start_line or 0,
            "end_line": unit.end_line or 0,
            "code": unit.code or "",
            "findings_json": findings_json,
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

@app.post("/assess-2198647")
async def assess_note_2198647(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        u.findings = build_findings(u)
        if not u.findings:
            continue
        llm_out = llm_assess_and_prompt(u)
        # Always make llm_prompt a string
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