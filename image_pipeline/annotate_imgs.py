# annotate_imgs.py
import os
import re
import json
import base64
import time
import random
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Labels directory:
#   - "labels/<Deck Name>.labels.txt"   (per-slide hints)
#   - "labels/<Deck Name>.labels.json"  (per-slide hints)
#   - "labels/<Deck Name>.gold.txt"     (deck-level concepts; one per line)
#   - "labels/<Deck Name>.gold.json"    (deck-level concepts; list or {"concepts":[...]})

BATCH_SIZE = 3 # slides per request
USE_RESPONSES_API = False 

# vision token reducers
MAX_IMAGE_SIDE = 1280  # downscale longest side
JPEG_QUALITY   = 85 # decrease jpeg quality

# throughput safety
THROTTLE_SECONDS = 1.0
MAX_RETRIES = 8
SKIP_IF_JSON_EXISTS = False

LABELS_DIR = Path(r".\labels")

# JSON mode guardrail: include a system message that literally says "json"
SYSTEM_MSG_JSON_MODE = (
    "You are a precise annotator. Always respond in json only. "
    "Return a single, well-formed JSON object that matches the requested schema. "
    "Do not include any extra commentary or text."
)

# --------------------------------------------------------
# helpers w/ files
def find_slide_folder(root):
    slides_dirs: List[Path] = []
    for file in sorted(root.iterdir()):
        if file.is_dir() and (any(file.glob("*.jpg")) or any(file.glob("*.jpeg")) or any(file.glob("*.png"))):
            slides_dirs.append(file)
    return slides_dirs

def sanitize_name(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name).strip()

def sort_key_numeric(path):
    m = re.search(r'(\d+)', path.stem)
    return (int(m.group(1)) if m else 10**9, path.name.lower())

def gather_slide_imgs(slides_dir):
    imgs = list(slides_dir.glob("*.jpg")) + list(slides_dir.glob("*.jpeg")) + list(slides_dir.glob("*.png"))
    imgs.sort(key=sort_key_numeric)
    return imgs

def encode_b64_image(img_path):
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    longest = max(w, h)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(longest)
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# --------------------------------------------------------
# image loading
def parse_labels_txt(text):
    out: Dict[int, List[str]] = {}
    current: Optional[int] = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = re.match(r"^\s*Slide\s+(\d+)\s*:\s*(.*)$", raw, re.I)
        if m:
            current = int(m.group(1))
            first = m.group(2).strip()
            if first and first.lower() != "none":
                out[current] = [first]
            else:
                out[current] = []
            continue
        if current is not None and raw.startswith(("\t", " ", "•", "-", "—")):
            item = raw.strip(" \t-–—•").strip()
            if item and item.lower() != "none":
                out.setdefault(current, []).append(item)
    return out

def load_labels_for_deck(deck_name):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    cand_json = LABELS_DIR / f"{deck_name}.labels.json"
    cand_txt  = LABELS_DIR / f"{deck_name}.labels.txt"

    if cand_json.exists():
        obj = json.loads(cand_json.read_text(encoding="utf-8"))
        out: Dict[int, List[str]] = {}
        for k, v in obj.items():
            try:
                k_i = int(k)
            except Exception:
                continue
            if v is None:
                out[k_i] = []
            elif isinstance(v, list):
                out[k_i] = [str(x).strip() for x in v if str(x).strip().lower() != "none"]
            else:
                s = str(v).strip()
                out[k_i] = [] if s.lower() == "none" or not s else [s]
        return out

    if cand_txt.exists():
        return parse_labels_txt(cand_txt.read_text(encoding="utf-8"))

    return {}  # no labels found

def build_guidance_json(labels, slide_indices):
    subset = {str(i): labels.get(i, None) for i in slide_indices if i in labels}
    return json.dumps(subset, ensure_ascii=False)

# --------------------------------------------------------
# gold concepts
def parse_gold_txt(text):
    items: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if s and s.lower() != "none":
            items.append(s)
    return items

def load_gold_concepts_for_deck(deck_name):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    cand_json = LABELS_DIR / f"{deck_name}.gold.json"
    cand_txt  = LABELS_DIR / f"{deck_name}.gold.txt"

    if cand_json.exists():
        obj = json.loads(cand_json.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "concepts" in obj and isinstance(obj["concepts"], list):
            return [str(x).strip() for x in obj["concepts"] if str(x).strip()]
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]

    if cand_txt.exists():
        return parse_gold_txt(cand_txt.read_text(encoding="utf-8"))

    return []  # none provided

# --------------------------------------------------------
def prompt(deck_name, slide_indices, guidance_json, gold_concepts):
    prompt = ("You are an expert slide-deck annotator. Your goal is to look at screenshots of the images "
              "and create sentences to describe ALL of the key concepts in the slide. This includes looking"
              "at the text and processing meaning in addition to looking at any visuals on the slideshow."
              "Again, the goal of this is to get at least 1 sentence per key concept."
              "You should not write anything about the course number, ie CS-1234 or the professor"
              "Assume the reader of your sentences is taking the class for the first time and is "
              "just looking for brief summaries of the concepts from the lecture."
              "You should ensure that all key concepts from the slidedeck are included in at least one"
              "sentence. Multiple sentences can be about the same topic, just ensure that every concept"
              "is mentioned somewhere in that chunk of slide. Here is an example for the type of "
              "sentences that could be useful. These are by no means the only sentences that you "
              f"could derive from these slides, but these are a start. The slidedeck name is {deck_name}"
              f". The slides in this batch are {slide_indices}. After you create the sentences, we "
              "will run another model that pulls the concepts from the sentences. Here were some "
              f"examples from the gold label concepts from this slidedeck {gold_concepts}. For the "
              "output, output the result strictly as JSON (with no extra text) in the following "
              "schema: \n"
                "{\n"
                '  \"deck\": \"<deck_name>\",\n'
                '  \"batch_slides\": [<slide_numbers>],\n'
                '  \"per_slide\": { \"<slide_number>\": [\"sentence 1\", \"sentence 2\", ...], ... },\n'
                '  \"sentences\": [\"slide-tagged sentences flattened in order\"],\n'
                "}")
    return prompt

# --------------------------------------------------------
# openai helpers
def call_model_responses_api(client, model, prompt_text, img_b64_list):
    content = [{"type": "input_text", "text": prompt_text}]
    for b64 in img_b64_list:
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            temperature=0.2,
            max_output_tokens=1200,
            response_format={"type": "json_object"},
        )
    except TypeError:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            temperature=0.2,
            max_output_tokens=1200,
        )
    if getattr(resp, "output_text", None):
        return resp.output_text
    try:
        parts = resp.output[0].content
        return "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
    except Exception:
        return json.dumps(resp.to_dict())

def call_model_chat_api(client, model, prompt_text, img_b64_list):
    messages = [
        {"role": "system", "content": SYSTEM_MSG_JSON_MODE},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in img_b64_list
        ]},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=1200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        if "must contain the word 'json'" in str(e).lower():
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
            )
            return resp.choices[0].message.content
        raise

def with_backoff(call_fn, *args, **kwargs):
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

# --------------------------------------------------------
# main annotations
def annotate_deck(deck_dir, out_dir, client, model):
    deck_name = deck_dir.name
    deck_out = out_dir / (sanitize_name(deck_name) + ".json")

    if SKIP_IF_JSON_EXISTS and deck_out.exists():
        print(f"[skip-existing] {deck_out.name}")
        return

    slides = gather_slide_imgs(deck_dir)
    if not slides:
        print(f"[skip] No images in {deck_dir}")
        return

    labels_map    = load_labels_for_deck(deck_name)
    gold_concepts = load_gold_concepts_for_deck(deck_name)

    out_data: Dict[str, Any] = {"deck": deck_name, "batches": []}

    for i in range(0, len(slides), BATCH_SIZE):
        batch_paths = slides[i:i + BATCH_SIZE]
        slide_nums  = [i + 1 + j for j in range(len(batch_paths))]

        guidance_json = build_guidance_json(labels_map, slide_nums) if labels_map else None
        prompt_text   = prompt(deck_name, slide_nums, guidance_json, gold_concepts)

        img_b64s = [encode_b64_image(p) for p in batch_paths]

        try:
            raw = with_backoff(
                call_model_chat_api if not USE_RESPONSES_API else call_model_responses_api,
                client, model, prompt_text, img_b64s
            )
            parsed = json.loads(raw)

            flat_sentences: List[str] = []
            per_slide = parsed.get("per_slide")
            if isinstance(per_slide, dict):
                for sn in slide_nums:
                    entries = []
                    if str(sn) in per_slide and isinstance(per_slide[str(sn)], list):
                        entries += per_slide[str(sn)]
                    if sn in per_slide and isinstance(per_slide[sn], list):
                        entries += per_slide[sn]
                    for s in entries:
                        s = str(s).strip()
                        if s:
                            flat_sentences.append(s)
            if not flat_sentences and isinstance(parsed.get("sentences"), list):
                flat_sentences = [str(s).strip() for s in parsed["sentences"] if str(s).strip()]
            parsed["deck"] = deck_name
            parsed["batch_slides"] = slide_nums
            parsed["sentences"] = flat_sentences

            if "gold_coverage" not in parsed:
                parsed["gold_coverage"] = {}

        except Exception as e:
            parsed = {
                "deck": deck_name,
                "batch_slides": slide_nums,
                "sentences": [],
                "gold_coverage": {},
                "raw_text": f"{e} :: {raw if 'raw' in locals() else ''}"
            }

        out_data["batches"].append(parsed)
        time.sleep(THROTTLE_SECONDS)

    deck_out.parent.mkdir(parents=True, exist_ok=True)
    deck_out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"[ok] {deck_name} → {deck_out}")

# --------------------------------------------------------
def main():
    IMAGE_INPUT_FOLDER = Path(r".\imgs")
    ANNOTATIONS_OUTPUT_FOLDER = Path(r".\annotations")
    ANNOTATIONS_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    OPEN_AI_MODEL = "gpt-4.1-mini"
    ai_model = OpenAI()

    for slideshow in find_slide_folder(IMAGE_INPUT_FOLDER):
        annotate_deck(slideshow, ANNOTATIONS_OUTPUT_FOLDER, ai_model, OPEN_AI_MODEL)


if __name__ == "__main__":
    main()
