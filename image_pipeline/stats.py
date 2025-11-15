from pathlib import Path
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
NORMALIZE_MODEL = "gpt-4o-mini"

STATS_DIR = Path("stats")
GOLD_PATH = STATS_DIR / "3.txt"
PRED_PATH = STATS_DIR / "4.txt"

# ---------------

def load_concepts(path):
    if not path.exists():
        raise FileNotFoundError(f"bad file: {path}")

    concepts = set()
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        c = line.strip()
        if c:
            concepts.add(c.lower())
    return concepts

# ---------------

def do_scoring(gold_path, pred_path):

    gold = normalize_file_to_set(gold_path)
    pred = normalize_file_to_set(pred_path)

    print(gold)
    print(pred)

    tp = len(gold & pred) # true pos
    fp = len(pred - gold) # false pos
    fn = len(gold - pred) # false neg

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    
    return tp, fp, fn, precision, recall, f1

# ---------------

@lru_cache(maxsize=1024)
def normalize_concept_with_gpt(concept):
    concept = concept.strip()
    if not concept:
        return ""

    system_msg = (
        "You normalize CS concept labels for evaluation.\n"
        "Given ONE concept phrase from a lecture, return a short canonical label.\n"
        "If you see duplicate or near duplicate concepts, combined then together into 1 concept. \n"
        "For example, if you have 'hybird scheduling' and 'hybird scheduling approach', make that into "
        "a single 'hybird scheduling' concept. Minimize the number of concepts while keeping all unique "
        "concepts. "
        "If you see a 'concept' that looks to be really a sentence, get rid of it."
        "Follow the codebook as below: \n"
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

    user_msg = f"Normalize this concept label:\n{concept}"

    resp = client.chat.completions.create(
        model=NORMALIZE_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=32,
    )
    normalized = resp.choices[0].message.content.strip()
    return normalized

# ---------------

def normalize_file_to_set(path):
    text = path.read_text(encoding="utf-8")
    concepts = set()
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        norm = normalize_concept_with_gpt(raw)
        if norm:
            concepts.add(norm.lower())
    return concepts

# ---------------

def main():
    if not STATS_DIR.exists():
        raise FileNotFoundError("issue w sta")

    tp, fp, fn, precision, recall, f1 = do_scoring(GOLD_PATH, PRED_PATH)
    print(f"tp: {tp}")
    print(f"fp: {fp}")
    print(f"tn: {fn}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")


if __name__ == "__main__":
    main()