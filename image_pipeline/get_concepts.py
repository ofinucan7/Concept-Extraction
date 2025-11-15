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
MODEL = "gpt-4o-mini"
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
        "You are an expert slide deck annotator. Your goal is to output a list of the concepts. "
        "IMPORTANT OUTPUT FORMAT: Return ONLY the concept phrases, one per line, with NO bullets, "
        "NO hyphens, NO numbers, and NO extra commentary before or after the list. "
        "Concepts are to be no more than 4–5 words. Output the concepts as plain text, "
        "one concept per line.\n"
        "Do not output any headings, labels, or explanations — only the concepts themselves.\n\n"
        f"You will be annotating {deck_name}. A LLM has pulled these sentences from the "
        "slidedeck you will be annotating. Those sentences were based on both the text and the images "
        "on each slide. You should try to be as conservative as possible in concepts. Err on the side of "
        "caution (i.e., if you are unsure if something is a concept, don't label it as one).\n"
        "Here are the sentences for the new slidedeck:\n"
        f"{current_sentences}\n\n"
        "Additionally, here is the full text transcript of the slidedeck. "
        "You should rely primarily on this transcript when deciding what is a concept:\n"
        f"{slides_text}\n\n"
        "If the transcript and the sentences ever disagree, trust the transcript. "
        "A valid concept MUST appear verbatim in the transcript — the words of the concept must appear "
        "exactly as you output them.\n"
        "If a term or phrase appears only in the sentence list and NOT anywhere in the transcript text, "
        "you must assume it is noise or summarization and DO NOT output it as a concept.\n\n"
        "You must IGNORE the title slide. Do not mark any concept taken from the title slide, course code, "
        "lecture number, instructor name, date, or institution name. If a phrase appears only on the title slide "
        "(for example: 'CS 0007 – Programming in Java', 'Lecture 17', 'University of Pittsburgh'), "
        "do NOT output it as a concept.\n"
        "Similarly, do not output generic section titles such as 'Main Portions of OOP', "
        "'Inter-class relationships', 'Overview', or similar framing phrases as concepts. Instead, output "
        "the more specific ideas listed under them (for example: 'subclass(es)', 'superclass(es)', 'ownership').\n\n"
        "Now I am going to give you an example of a slidedeck that a human has annotated in this way. "
        "Use this example only to understand how to choose and format concepts. Just because a word is a "
        "concept in the EXAMPLE does not mean it will always be a concept in other slidedecks. "
        "You will be given slidedecks from a wide variety of courses across many different topics. "
        "Based on patterns you observe for why something is labeled as a concept in the example below, "
        "make the judgment calls for whether something is a concept or not for the CURRENT deck using the "
        "text given above.\n\n"
        "Within the main text or transcript of a slideshow, there should be a verbatim match for every concept. "
        "That is, the words must appear in the slideshow exactly as you write the concept for it to be considered "
        "a concept.\n\n"
        "Do not include examples as concepts. There may be concepts identified within an example, but you should "
        "not output whole examples as concepts. For example, if there is an example about cars, vehicles, "
        "or customers, do NOT output phrases like 'Vehicle class', 'Garage class', 'Automotive shop', "
        "'Customer details', 'Service quote', or similar example-specific phrases. Instead, only output the "
        "underlying generic CS concept, such as 'class', 'ownership', 'instance variables', or 'method'.\n\n"
        "Also, do NOT label very generic programming terms as concepts unless the deck devotes a slide to "
        "introducing or defining them as main ideas. In particular, terms like 'variable(s)', 'return type', "
        "'void', 'scope', 'pass-by-value', 'pass-by-reference', 'passing in', 'access modifier(s)', or 'class' "
        "by itself should usually NOT be output as concepts. Prefer the more specific OOP concepts such as "
        "'instance variable(s)', 'constructor(s)', 'method(s)', 'protected access modifier', 'extends', "
        "and 'inheritance' when those are clearly defined.\n"
        "For this deck, do NOT output the following phrases as concepts even if they appear in the transcript "
        "or sentences: 'variable(s)', 'return type', 'void', 'scope', 'pass-by-value', 'pass-by-reference', "
        "'passing in', 'access modifier(s)', 'private access modifier', 'inter-class relationships', and 'class' "
        "by itself. These are too generic here; instead, focus on the more specific OOP concepts they help explain.\n\n"
        "If you could remove an adjective or extra word and the remaining phrase would still clearly be the same "
        "concept, then REMOVE the extra word. For example, prefer 'method' over 'main method' unless 'main method' "
        "is explicitly defined as its own concept; prefer 'class component(s)' or 'instance variables' over phrases "
        "like 'simple properties', 'Vehicle class', or 'Garage class'.\n\n"
        "When there are multiple surface forms of the same concept:\n"
        "• If there are singular and plural variants, output a single canonical line using '(s)' for optional plural "
        "(for example: 'Class(es)', 'Subclass(es)', 'Superclass(es)', 'Constructor(s)').\n"
        "• If there is an abbreviation and its full form (for example 'OOP' and 'Object Oriented Programming'), "
        "combine them into a single line using '/', such as 'OOP/Object Oriented Programming'.\n"
        "Do NOT output both forms separately. Collapse such variants into a single canonical concept line.\n\n"
        "For a deck like this one, which explicitly introduces object-oriented programming with slides titled "
        "'Main Portions of OOP', 'Class Components', and 'Inter-class Relationships', you MUST include the "
        "following as concepts if they appear anywhere in the transcript: 'OOP/Object Oriented Programming', "
        "'class component(s)', 'class(es)', 'instance(s)', 'instance variable(s)', 'constructor(s)', 'method(s)', "
        "'extends', 'inheritance', 'subclass(es)', 'superclass(es)', 'override', 'override annotation', "
        "'method overriding', 'protected access modifier', and 'ownership'.\n\n"
        "Here is that example slidedeck transcript:\n"
        f"{cs0007_7_text}\n\n"
        "Here are the sentences for that example slideshow:\n"
        f"{cs0007_7_sentences}\n\n"
        "Finally, here are the concepts that should be pulled from that example slideshow and its sentences:\n"
        f"{cs0007_7_concepts}\n"
        "\n==============================================\n"
        "Here is another slidedeck text: \n"
        f"{cs1502_10_text}"
        "\n And for that slidedeck, here are the concepts \n"
        f"{cs1502_10_concepts}"
        "\n==============================================\n"
        "Here is one more slidedeck text: \n"
        f"{cs1550_23_text}"
        "\n And for that slidedeck, here are the concepts \n"
        f"{cs1550_23_concepts}"
        "\n==============================================\n"
        "Here is the annotation guide (for going from full text to concepts) that I want you to follow. "
        "Even though this is for the full text to slides, for the sentences, apply a similar logic.\n"
        "1. Purpose\n"
        "This codebook establishes consistent criteria for annotating concepts within lecture slides. A "
        "concept is defined as any explicitly introduced or emphasized idea that the instructor intends "
        "students to understand and retain.\n"
        "2. Primary Rules\n"
        "2.1 Slide-Deck Grounding Rule\n"
        "Only consider a term a concept if its justification exists within the slide deck itself. Do "
        "not infer concepts from outside knowledge, prior lectures, or general domain assumptions.\n"
        "2.2 Repetition Rule\n"
        "Frequency of mention does not imply conceptual status. Inclusion depends on definition, emphasis, "
        "or instructional focus, not on how many times a term appears.\n"
        "2.3 Granularity Rule\n"
        "Annotate concepts at the most specific level that adds meaningful information. If subdividing "
        "a phrase adds no additional meaning or context, treat it as a single concept. If the smaller "
        "components each carry distinct meaning, annotate them separately.\n"
        "2.4 Definition and Emphasis Rule\n"
        "A concept is any word or phrase that is:\n"
        "• Clearly defined or described in the slide deck,\n"
        "• Highlighted as an essential idea or takeaway,\n"
        "• Used as the subject of an example or explanation.\n"
        "2.5 Non-Conceptual Modifiers\n"
        "Modifiers, keywords, or syntactic elements are not labeled unless the slide deck explicitly "
        "defines or discusses them as concepts (e.g., 'static', 'visibility').\n"
        "3. Supporting Rules\n"
        "• Abbreviation Rule: Label abbreviations (e.g., 'ISR') when used to represent a defined concept.\n"
        "• Variant Form Rule: Include plural or adjectival variants, but collapse variants into a single "
        "canonical concept as described above.\n"
        "• Adjoinment Rule: Annotate multiword concepts when their meaning is unified (e.g., 'function header').\n"
        "• Generic Term Rule: Do not label generic or context-free terms (e.g., 'thing', 'object') unless defined "
        "specifically in this deck.\n"
        "• Out-of-Discipline Rule: Include external domain terms only if explicitly introduced "
        "(e.g., 'least squares regression').\n"
        "4. Annotation Strategy\n"
        "Annotators should:\n"
        "1. Read the slide deck in order.\n"
        "2. Identify explicitly defined or emphasized terms.\n"
        "3. Apply the granularity rule to decide labeling scope.\n"
        "4. Avoid inferring or overgeneralizing beyond the presented material.\n"
        "Remember: in your final answer, output ONLY the canonical concept phrases, one per line, "
        "with no bullets, no numbers, and no extra commentary."
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
