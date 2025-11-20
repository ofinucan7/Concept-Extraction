import os
import re
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_ANNOTATIONS_DIR = Path(r".\annotations")  # where your sentence JSONs are
TEXTS_DIR = Path(r".\texts")                    # where your full transcripts are
OUTPUT_CONCEPTS_DIR = Path(r".\concepts")       # where to save concepts outputs
OUTPUT_CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)

# openai stuff
MODEL = "gpt-4.1-mini"
MAX_TOKENS = 1200
USE_RESPONSES_API = False

# Throughput safety
MAX_RETRIES = 8
THROTTLE_SECONDS = 1.0

# if json already exists --> skip making new one
SKIP_IF_JSON_EXISTS = False

SYSTEM_MSG = "You are a helpful assistant that follows the user instructions exactly."

# --------------------------------------------------------
# helpers

def find_deck_jsons(root: Path) -> List[Path]:
    return sorted([p for p in root.glob("*.json") if p.is_file()])


def gather_sentences(deck_obj: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for b in deck_obj.get("batches", []):
        if isinstance(b, dict) and "sentences" in b and isinstance(b["sentences"], list):
            out.extend([str(s) for s in b["sentences"]])
            continue
        if isinstance(b, dict) and "topics" in b and isinstance(b["topics"], list):
            for t in b["topics"]:
                if isinstance(t, dict) and "sentences" in t and isinstance(t["sentences"], list):
                    out.extend([str(s) for s in t["sentences"]])

    # de-duplicate + strip empties
    seen = set()
    uniq = []
    for s in out:
        s_norm = s.strip()
        if s_norm and s_norm not in seen:
            seen.add(s_norm)
            uniq.append(s_norm)
    return uniq

# --------------------------------------------------------
# backoff

def with_backoff(call_fn, *args, **kwargs) -> str:
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            return call_fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            is429 = ("rate limit" in msg.lower()) or ("429" in msg) or ("rate_limit_exceeded" in msg.lower())
            if not is429:
                raise
            m = re.search(r"try again in\s*([0-9.]+)s", msg, re.I)
            wait = float(m.group(1)) if m else delay
            wait *= (1.0 + random.random() * 0.4)  # jitter
            wait = min(wait, 30.0)
            print(f"[rate-limit] retry in {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("Gave up after repeated rate limit errors.")


def call_model_chat_api(client: OpenAI, model: str, prompt_text: str) -> str:
    """Simple chat call (no JSON mode) used by with_backoff."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": prompt_text},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content


def call_model_responses_api(client: OpenAI, model: str, prompt_text: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_MSG}]},
            {"role": "user",   "content": [{"type": "input_text", "text": prompt_text}]},
        ],
        temperature=0.1,
        max_output_tokens=MAX_TOKENS,
    )
    return getattr(resp, "output_text", None) or json.dumps(resp.to_dict())

# --------------------------------------------------------
# prompt

def read_in_helper_text(path):
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""

    return text

def build_concept_prompt(deck_name, sentences, slides_text):
    # example deck for in-context guidance
    cs0007_7_text_path = Path("prompt_helper_data") / "cs0007-7-transcript.txt"
    cs0007_7_sentence_path = Path("annotations") / "Lecture 7.json"
    cs0007_7_concepts_path = Path("prompt_helper_data") / "cs0007-7-concepts.txt"

    cs1502_10_text_path = Path("prompt_helper_data") / "cs1502-10-transcript.txt"
    cs1502_10_concepts_path = Path("prompt_helper_data") / "cs1502-10-concepts.txt"

    cs1550_23_text_path = Path("prompt_helper_data") / "cs1550-23-transcript.txt"
    cs1550_23_concepts_path = Path("prompt_helper_data") / "cs1550-23-concepts.txt"

    cs0007_7_text = read_in_helper_text(cs0007_7_text_path)
    cs0007_7_concepts = read_in_helper_text(cs0007_7_concepts_path)
    cs1502_10_text = read_in_helper_text(cs1502_10_text_path)
    cs1502_10_concepts = read_in_helper_text(cs1502_10_concepts_path)
    cs1550_23_text = read_in_helper_text(cs1550_23_text_path)
    cs1550_23_concepts = read_in_helper_text(cs1550_23_concepts_path)

    try:
        deck_obj = json.loads(cs0007_7_sentence_path.read_text(encoding="utf-8"))
        sent_list = gather_sentences(deck_obj)
        cs0007_7_sentences = "\n".join(f"- {s}" for s in sent_list)
    except FileNotFoundError:
        cs0007_7_sentences = ""

    # format the *current* sentences nicely too
    current_sentences = "\n".join(f"- {s}" for s in sentences)

    prompt = (
        "You are an expert academic annotator specializing in concept extraction from university lecture slides. "
        "Identify and list only the key concepts that are explicitly defined, emphasized, or used in examples within " 
        "the provided datasets. These datasets consist of a full transcript of the powerpoint along with sentences summarizing the powerpoint. Place a 90 percent weight on the text and 10 percent on the summaries. If unsure, exclude the term (prioritize precision over recall)."
        "Again, your goal is to produce the most concise and minimal list of explicitly defined concepts."
        "Do not include near-synonyms, plural variants, or implementation details unless the slide clearly defines "
        "them as separate terms. If you are uncertain, omit the term."
        "If multiple variants of a term appear (e.g., “override”, “overriding”), keep only the canonical or dictionary form. \n"
        
        "Slide-Deck Grounding Rule — Do not infer concepts not textually supported, but include any concept that the slide "
        "defines, titles, or repeatedly references. You should be able to point to a particular slide and say this is"
        "where this concept came from. \n"
        "Granularity Rule — Annotate at the most specific meaningful level. If a concept phrase (“superclass constructor”) " 
        "is only an instance or elaboration of a broader concept (“constructor”), include only the broader term unless "
        "the slide defines the specific variant. \n"
        "Definition & Emphasis Rule — Label words or phrases that are: explicitly defined (“X is a …”), visually emphasized (bold/italic/title-case), or used as key examples. \n"
        "Explicit evidence includes patterns such as “X is a Y”, “X means …”, bold headings, or definition bullets. \n"
        "Abbreviation Rule — Include both abbreviation and full form only if both appear. \n"
        "Normalization Rule — Output each concept once, in singular form. Use lowercase unless the concept is an acronym. Do not repeat plural or synonymous variants (e.g., “method/methods”, “class/classes”). \n"
        "Implementation Filter — Exclude code-implementation details (field, property, variable, getter, setter, overloaded constructor, superclass constructor) unless explicitly defined as conceptual terms. \n"
        "Relation Boundary Rule — Include relational concepts only if the term itself is introduced or defined (e.g., “Inheritance = relationship between superclass and subclass”). Do not list relational variants (“parent class”, “child class”, “subtype”, “inter-class relationship”) unless defined verbatim. \n"
        "Keyword Filter — Exclude language keywords or visibility specifiers (public, private, static, final) unless defined as concepts. \n"
        "Semantic Boundary Rule — Ignore meta-instructional phrases or general advice (e.g., “modeling a problem”, “find the nouns”). \n"
        "Core Concept Retention Rule — Always include foundational terms that appear in definitions, titles, or repeated headers (e.g., “program”, “memory”, “class”, “inheritance”). If a slide introduces a section or example with such a term, treat it as a core concept even if not fully defined. \n"
        "Precision Emphasis Rule — When uncertain, omit rather than guess.\n"
        "Self-Check Step — After extraction, remove any concept not clearly supported by definition, emphasis, or example. \n"

        "IMPORTANT OUTPUT FORMAT: Return ONLY the concept phrases, one per line, with NO bullets, "
        "NO hyphens, NO numbers, and NO extra commentary before or after the list. "
        "Concepts are to be no more than 4–5 words. Output the concepts as plain text, "
        "one concept per line.\n"

        "===================================== \n"

        "FILTERING STEP: \n"
        "After identifying possible concepts, remove any that are **not explicitly** defined, emphasized, or used in an example or explination. \n"
        "\n"
        "OUTPUT REQUIREMENTS \n"
        "- Extract all concepts from the sentences \n"
        "- Do **not** number them. \n "
        "- Do **not** include reasoning, commentary, or explanations. \n"
        "- Include abbreviations *and* their full forms as separate lines. \n"
        "- Keep consistent capitalization (use lowercase unless the concept is an acronym). \n"
        "- Do not include REPEATED COURSE NAME as a concept. \n"
        "\n"
        "Self-Check Step (Precision Pass): \n"
        "Before finalizing, review your list and remove any term that: \n"
        "is not directly defined, emphasized, or exemplified; "
        "duplicates or pluralizes another item; or "
        "is a code keyword, modifier, or descriptive relationship rather than a conceptual noun phrase. \n"

        f"You will be annotating {deck_name}. \n"
        "And here is the text from the powerpoint: \n "
        f"{slides_text}\n\n"
        "And here is the sentences summarizing it: \n"
        f"{sentences} \n \n"

        "After going pulling the concepts, go back through the concepts list and compare what you have to the sentences and powerpoint to ensure that it meets the criteria for a concept."
    )

    return prompt

# ------------------------------------------------------------------------
# main logic

def main():
    client = OpenAI()

    for deck_json_path in find_deck_jsons(INPUT_ANNOTATIONS_DIR):
        deck_name = deck_json_path.stem
        out_path = OUTPUT_CONCEPTS_DIR / f"{deck_name}.txt"

        if SKIP_IF_JSON_EXISTS and out_path.exists():
            print(f"[skip-existing] {out_path}")
            continue

        # 1) sentences from annotations JSON
        deck_obj = json.loads(deck_json_path.read_text(encoding="utf-8"))
        sentences = gather_sentences(deck_obj)

        # 2) full transcript from texts/<deck_name>.txt
        text_path = TEXTS_DIR / f"{deck_name}.txt"
        try:
            slides_text = text_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"[warn] Missing transcript for {deck_name} at {text_path}")
            slides_text = ""

        # 3) build prompt with both contexts
        prompt = build_concept_prompt(deck_name, sentences, slides_text)

        # 4) call model with backoff
        raw = with_backoff(
            call_model_chat_api if not USE_RESPONSES_API else call_model_responses_api,
            client=client,
            model=MODEL,
            prompt_text=prompt,
        )

        # 5) save raw concepts output
        out_path.write_text(raw, encoding="utf-8")
        print(f"[ok] {deck_name} → {out_path}")

if __name__ == "__main__":
    main()
