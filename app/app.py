# app_2198647_strict.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import os, re, json

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

# ---- LLM Setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2198647 Assessment - Strict Schema")

# ===== Tables & Fields impacted by Note 2198647 =====
OBSOLETE_TABLES = {"VBUK": "VBAK", "VBUP": "VBAP"}
SQL_OBSOLETE_FIELDS = {"VBTYP_EXT"}      # only checked in SQL
DECL_LENGTHENED = {"VBTYP": "VBTYPL"}    # checked in declarations
DECL_OBSOLETE = {"VBTYP_EXT"}            # obsolete declarations

# ===== Regex patterns =====
SQL_SELECT_BLOCK_RE = re.compile(
    r"\bSELECT\b(?P<select>.+?)\bFROM\b\s+(?P<table>\w+)(?P<rest>.*?)(?=(\bSELECT\b|$))",
    re.IGNORECASE | re.DOTALL,
)
JOIN_RE = re.compile(r"\bJOIN\s+(?P<table>\w+)", re.IGNORECASE)
DECLARATION_RE = re.compile(
    r"\b(?:TYPE|LIKE)\b\s+(?P<field>[A-Z0-9_]+)", re.IGNORECASE
)

# ===== Strict Models =====
class SelectItem(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: str

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def filter_none(cls, v: List[str]) -> List[str]:
        return [x for x in v if x is not None]

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: str
    code: Optional[str] = ""
    selects: List[SelectItem] = Field(default_factory=list)

# ===== Detection logic =====
def comment_table(tbl: str) -> str:
    return f"* TODO: Table {tbl} obsolete in S/4HANA (SAP Note 2198647). Replace with {OBSOLETE_TABLES[tbl]}."

def comment_decl_field(f: str) -> str:
    if f in DECL_OBSOLETE:
        return f"* TODO: Field {f} obsolete (2198647). Remove usage."
    if f in DECL_LENGTHENED:
        return f"* TODO: Data element {f} lengthened (2198647). Use {DECL_LENGTHENED[f]}."
    return ""

def comment_sql_field(f: str) -> str:
    return f"* TODO: Field {f} obsolete in S/4HANA (2198647). Remove usage."

def parse_and_build_selectitems(code: str) -> List[SelectItem]:
    results: List[SelectItem] = []

    for stmt in SQL_SELECT_BLOCK_RE.finditer(code):
        table = stmt.group("table").upper()
        rest_text = stmt.group("rest")

        # FROM
        if table in OBSOLETE_TABLES:
            results.append(
                SelectItem(
                    table=table,
                    target_type="TABLE",
                    target_name=table,
                    used_fields=[table],
                    suggested_fields=[OBSOLETE_TABLES[table]],
                    suggested_statement=comment_table(table)
                )
            )

        # JOIN
        for jm in JOIN_RE.finditer(rest_text):
            jtable = jm.group("table").upper()
            if jtable in OBSOLETE_TABLES:
                results.append(
                    SelectItem(
                        table=jtable,
                        target_type="TABLE",
                        target_name=jtable,
                        used_fields=[jtable],
                        suggested_fields=[OBSOLETE_TABLES[jtable]],
                        suggested_statement=comment_table(jtable)
                    )
                )

        # SQL fields
        if "VBTYP_EXT" in stmt.group(0).upper():
            results.append(
                SelectItem(
                    table="",
                    target_type="SQL_FIELD",
                    target_name="VBTYP_EXT",
                    used_fields=["VBTYP_EXT"],
                    suggested_fields=[],
                    suggested_statement=comment_sql_field("VBTYP_EXT")
                )
            )

    # Declaration scanning
    for m in DECLARATION_RE.finditer(code):
        fld = m.group("field").upper()
        if fld in DECL_OBSOLETE or fld in DECL_LENGTHENED:
            results.append(
                SelectItem(
                    table="",
                    target_type="DECLARATION",
                    target_name=fld,
                    used_fields=[fld],
                    suggested_fields=([DECL_LENGTHENED[fld]] if fld in DECL_LENGTHENED else []),
                    suggested_statement=comment_decl_field(fld)
                )
            )

    return results

# ===== Summariser =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    field_count: Dict[str, int] = {}
    flagged = []
    for s in unit.selects:
        for f in s.used_fields:
            field_count[f.upper()] = field_count.get(f.upper(), 0) + 1
            flagged.append({"field": f, "reason": s.suggested_statement})
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_occurrences": len(unit.selects),
            "fields_frequency": field_count,
            "note_2198647_flags": flagged
        }
    }

# ===== LLM prompt =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 2198647. Output strict JSON only."
USER_TEMPLATE = """
You are assessing ABAP code usage in light of SAP Note 2198647 (obsolete VBUK/VBUP tables, fields VBTYP and VBTYP_EXT).

We provide metadata and findings (under "selects"). 
Tasks:
1) Produce a concise assessment of impact.
2) Produce an actionable LLM remediation prompt to insert TODO comments.

Return ONLY strict JSON:
{{
  "assessment": "<concise 2198647 impact>",
  "llm_prompt": "<remediation prompt>"
}}

Unit metadata:
- Program: {pgm}
- Include: {inc}
- Unit type: {utype}
- Unit name: {uname}

Summary:
{plan_json}

Selects (JSON findings):
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE)
])
llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan_json = json.dumps(summarize_selects(unit), indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], indent=2)
    try:
        return chain.invoke({
            "pgm": unit.pgm_name,
            "inc": unit.inc_name,
            "utype": unit.type,
            "uname": unit.name,
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM failed: {e}")

# ===== API Endpoint =====
@app.post("/assess-2198647")
def assess_note_2198647(units: List[Unit]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for u in units:
        if u.code:
            u.selects = parse_and_build_selectitems(u.code)
        if not u.selects:
            # no impacted usage
            obj = u.model_dump()
            obj.pop("selects", None)
            obj["assessment"] = "No usage of obsolete tables/fields from SAP Note 2198647."
            obj["llm_prompt"] = ""
            out.append(obj)
            continue

        llm_out = llm_assess_and_prompt(u)
        obj = u.model_dump()
        obj.pop("selects", None)
        obj.pop("code", None)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "note": "2198647", "model": OPENAI_MODEL}