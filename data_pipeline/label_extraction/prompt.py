'''
Prompt construction for label extraction.

Builds the system + user prompt for the extraction call. The model must
answer with strict JSON only (see ``schema.py``).

Contents:
    - per-condition multilingual glossary (EN/ES/FR/NL; seed terms observed
      in the actual training data)
    - explicit negation and hedging policy
    - truncation / unparseable handling
    - a few-shot exemplar (a short real gold-labeled report)

Prompt versions:
    - v1: original prompt (default; used by all prior calibration runs).
    - v2: adds one narrow rule on top of v1 - an instruction to read the
      whole report body (not just the conclusion) specifically for
      Effusion, since it's the only condition where phi4's round-1 misses
      (run 20260909_phi4) showed a genuine, isolated coverage gap: real
      supporting text present but buried outside the summary line, fixable
      without side effects (a scoped test on this run dropped Effusion's
      FN 1->0 with no change in its FP count).
      A broader version of this rule (applied to all 12 conditions) and
      several other candidate rules (grade-1/2-sprain exclusions for
      ACL/MCL, mild-chondrosis thresholds for the OA conditions, an
      effusion-implies-synovitis override) were tried and rejected: the
      severity/grade-language rules all failed the same check - identical
      phrasing (e.g. "grade II sprain", "mild chondrosis") labeled both
      positive and negative for the same condition elsewhere in the 58-row
      gold set, i.e. genuine annotation noise, not a learnable pattern. A
      good chunk of the remaining false negatives (most of Synovitis,
      Fracture, Lateral OA, PF OA) have no supporting text in the report at
      all, so no prompt wording can recover them. The blanket coverage rule
      also measurably increased FPs on already over-calling conditions
      (ACL, MCL) without a matching benefit elsewhere - see notebook
      section 4 for the numbers.
    Selected via ``config.PROMPT_VERSION`` (env var ``PROMPT_VERSION``).
'''

from data_pipeline.label_extraction.config import CONDITIONS, CONDITION_DESCRIPTIONS, PROMPT_VERSION

# --- Per-condition multilingual glossary of finding terms (seed terms observed
#     in the multilingual training data). Deliberately terse. ---
GLOSSARY = {
    'ACL': (
        "anterior cruciate ligament; EN: ACL tear/injury/rupture, signal "
        "increase, disruption, discontinuity; ES: ligamento cruzado anterior, "
        "rotura, senal aumentada; FR: ligament croise anterieur, lesion, "
        "rupture; NL: voorste kruisband, scheur, ruptuur, verhoogd signaal"
    ),
    'MCL': (
        "medial collateral ligament; EN: MCL injury/tear/edema, signal "
        "increase; ES: ligamento colateral medial, rotura, edema; FR: "
        "ligament collaterale median, lesion, oedeme; NL: mediaal "
        "collaterale ligament, MCL, scheur, vocht"
    ),
    'Medial Meniscus': (
        "medial meniscus TEAR (or other acute parenchymal injury); EN: tear, "
        "rupture, intrameniscal signal communicating with the joint surface, "
        "abnormal morphology, flap, oedema; NOT an isolated degenerative/"
        "vacuolar appearance; ES: rotura de menisco medial, seal intra-"
        "meniscal, desgarro; FR: dechirure du menisque medial, lesion, signal "
        "intra-meniscal; NL: mediale meniscusscheur, intra-meniscaal signaal"
    ),
    'Lateral Meniscus': (
        "lateral meniscus TEAR (or other acute parenchymal injury); EN: tear, "
        "rupture, intrameniscal signal communicating with the joint surface, "
        "abnormal morphology, flap, oedema; NOT an isolated degenerative/"
        "vacuolar appearance; ES: rotura de menisco lateral; FR: dechirure du "
        "menisque lateral, lesion; NL: laterale meniscusscheur"
    ),
    'Medial OA': (
        "medial femorotibial osteoarthritis; EN: OA, chondral loss, "
        "subchondral sclerosis, chronic-pattern bone marrow lesion, "
        "osteophytes; ES: artrosis, condroapatia, esclerosis subcondral; FR: "
        "arthrose, chondropathie, sclerose sous-condylienne; NL: artrose, "
        "kraakbeenschade"
    ),
    'Lateral OA': (
        "lateral femorotibial osteoarthritis (same definitions as the medial "
        "compartiment, lateral side)"
    ),
    'PF OA': (
        "patellofemoral osteoarthritis / patellar cartilage abnormality; EN: "
        "chondropathy, patellar cartilage loss/damage; ES: condroapatia "
        "rotuliana, dano del cartilago; FR: chondropathie rotulienne, lesion "
        "du cartilage; NL: patellair cartilage, kraakbeenschade"
    ),
    'Effusion': (
        "joint effusion of any degree; EN: effusion, joint fluid, complex "
        "effusion, hemarthrosis; ES: derrame articular, liquido articular, "
        "hemartrosis; FR: epanchement articulaire, liquide articulaire, "
        "hemarthrose; NL: effusie, vocht, hemarthrose"
    ),
    'Synovitis': (
        "synovitis / synovial reaction; EN: synovitis, synovial "
        "thickening/hypertrophy, villous hypertrophy; ES: sinovitis, "
        "espesor sinovial; FR: synovite, epaississement synovial; NL: "
        "synovitis, vergroting van het slijmvlies"
    ),
    "Baker's": (
        "Baker's (popliteal) cyst; EN: Baker's cyst, popliteal cyst, cyst "
        "with flap or communication; ES: quiste popliteo, quiste de Baker; "
        "FR: kyste poplite, kyste de Baker; NL: Bakercyste, popliteale cyste"
    ),
    'Contusion': (
        "bone contusion / acute bone marrow edema (injury pattern); EN: bone "
        "contusion, acute bone marrow edema, focal marrow edema of the acute "
        "injury pattern; ES: contusion osea, edema de medula; FR: contusion "
        "osseuse, oedeme de la moelle; NL: botcontusie, beenmerg-oedeem"
    ),
    'Fracture': (
        "fracture anywhere in the knee; EN: fracture, non-displaced "
        "fracture, impacted fracture, hairline fracture; ES: fractura; FR: "
        "fracture, fracture de stress; NL: fractuur"
    ),
}

# A short, real, gold-labeled exemplar (kept in the codebase so the few-shot
# example stays stable across runs).
_EXEMPLAR_REPORT = (
    "Technique: MRI of the knee. ACL normal.  MCL normal. "
    "Medial meniscus tear. Lateral meniscus normal.  Cartilages normal. "
    "No baker's cyst. No effusion. Conclusion: Medial meniscus tear."
)
_EXEMPLAR_LABELS = {
    'ACL': 0, 'MCL': 0, 'Medial Meniscus': 1, 'Lateral Meniscus': 0,
    'Medial OA': 0, 'Lateral OA': 0, 'PF OA': 0, 'Effusion': 0,
    'Synovitis': 0, "Baker's": 0, 'Contusion': 0, 'Fracture': 0,
}


SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist. You read knee MRI
reports written in several languages (English, Spanish, French, Dutch, German,
Greek, and others) and extract binary findings for exactly 12 conditions.

Rules:
- For each of the 12 conditions, decide 1 (present) or 0 (absent).
- Positivity: label 1 when the report states the finding as a finding
  (definite, or mildly hedged like "suspicious for", "possible", "cannot
  rule out / exclude", "indizi", "verdacht").
- Negation: an explicit negation ("no tear", "sin rotura", "aucune
  dechirure", "geen scheur", "intact", "unauffaellig", "within normal
  limits") means label 0, even when the anatomical structure is mentioned.
- Mentioning a structure only to call it normal or unremarkable means 0.
- Use ONLY what the report says. Do not infer findings that are not stated.
- If the report is clearly truncated mid-sentence, is gibberish, or is not a
  readable knee report, answer with {{"unparseable": true}} and nothing else.

Multilingual glossary of condition terms (examples, not exhaustive):
{glossary}

Respond with STRICT JSON ONLY. No prose, no markdown, no code fences. The
schema is:
{{
  "unparseable": false,
  "labels": {{
    "<condition>": {{"value": 0 or 1, "evidence": "<short verbatim quote from the report supporting the value; empty string if none>"}},
    ... one entry for each of the 12 conditions ...
  }}
}}
"""

SYSTEM_PROMPT_V2 = SYSTEM_PROMPT.replace(
    "- If the report is clearly truncated mid-sentence, is gibberish, or is not a\n"
    "  readable knee report, answer with {{\"unparseable\": true}} and nothing else.\n",
    "- If the report is clearly truncated mid-sentence, is gibberish, or is not a\n"
    "  readable knee report, answer with {{\"unparseable\": true}} and nothing else.\n"
    "- Effusion is sometimes mentioned once in the body of the report and not\n"
    "  repeated in a summary/conclusion line - read the entire report body,\n"
    "  not just the conclusion, before deciding on Effusion specifically.\n"
)

_USER_TMPL = """Example (few-shot):
Report:
{exemplar_report}

Expected answer:
{exemplar_json}

---

Now do the same for this report:
Report:
{report}

Respond with STRICT JSON ONLY following the schema."""


def build_glossary_text() -> str:

    lines = []

    for c in CONDITIONS:
        desc = CONDITION_DESCRIPTIONS.get(c, c)
        lines.append(f'{c}: {desc}. Terms: {GLOSSARY[c]}')

    return '\n'.join(lines)


def build_system_prompt() -> str:
    template = SYSTEM_PROMPT_V2 if PROMPT_VERSION == 'v2' else SYSTEM_PROMPT
    return template.format(glossary=build_glossary_text())


def build_user_prompt(report: str) -> str:
    return _USER_TMPL.format(
        exemplar_report=_EXEMPLAR_REPORT,
        exemplar_json=_labels_json(_EXEMPLAR_LABELS),
        report=report,
    )


def build_messages(report: str):
    '''Return the list of (role, content) tuples for one extraction call.'''

    return [
        ('system', build_system_prompt()),
        ('user', build_user_prompt(report)),
    ]


def _labels_json(labels: dict) -> str:
    '''Render a {cond: 0|1} dict as exemplar label JSON.'''

    inner = ',\n    '.join(f'"{c}": {int(labels[c])}' for c in CONDITIONS)
    return ('{\n    "unparseable": false,\n    "labels": {\n    '+ inner + '\n    }\n}')
