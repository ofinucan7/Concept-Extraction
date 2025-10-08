import json, re, argparse, requests
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import fitz
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

# note you need to install ollama, found here: https://ollama.com/download
# then after going through the setup, run: ollama pull qwen2.5:1.5b-instruct

# --------------------------------------------------
# models
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:1.5b-instruct" 

# how many words the max concept can be in length
LENGTH_OF_CONCEPT = (1, 3)

# candidates per page
TOP_N_PER_PAGE = 15

# aggression level of merging terms together into unique list
MERGE_AGGRESSION = 0.80

# stricter allignment with sylly
SYLLY_ALIGNMENT = 0.55

# --------------------------------------------------
# pdf text extractor - grab the text from pdf
# return [(page index, text)]
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r'\s+', ' ', text).strip()
        pages.append((i, text))
    return pages

# --------------------------------------------------
# clean a given phrase of bad tokens
# return the cleaned phrase
def clean_phrase(p):
    p = p.lower().strip()
    p = re.sub(r"[^a-z0-9\s\-/+]", "", p)
    p = re.sub(r"\s+", " ", p)
    return p

# --------------------------------------------------
# getting rid of similar terms (ie gradient descent, grad descent, gradient descent algorithm, grad desc --> into 1 singular term)
# unique list of terms
def dedup_semantic(phrases: List[str], embedder: SentenceTransformer, threshold: float) -> List[str]:
    phrases = [clean_phrase(p) for p in phrases if p and len(p) > 1]
    if not phrases:
        return []
    embs = embedder.encode(phrases, normalize_embeddings=True, convert_to_numpy=True)
    keep, used = [], np.zeros(len(phrases), dtype=bool)
    for i in range(len(phrases)):
        if used[i]: 
            continue
        keep.append(phrases[i])
        sims = embs @ embs[i]
        used[np.where(sims >= threshold)[0]] = True
    return keep

# --------------------------------------------------
# get the concepts from the text
# returns (phrase, score)
def get_concepts_from_text(text, kw, top_n):
    # KeyBERT returns (phrase, score)
    phrases = [p for p, s in kw.extract_keywords(
        text,
        keyphrase_ngram_range=LENGTH_OF_CONCEPT,
        stop_words="english",
        use_mmr=True,
        diversity=0.6,
        top_n=top_n
    )]
    return phrases

# --------------------------------------------------
# running model a
def run_model_a(pdf_path, embedder: SentenceTransformer):
    kw = KeyBERT(model=embedder)
    pages = extract_pdf_text(pdf_path)
    per_page = {}
    all_candidates = []
    for idx, txt in pages:
        if not txt:
            per_page[idx] = []
            continue
        cands = get_concepts_from_text(txt, kw, TOP_N_PER_PAGE)
        per_page[idx] = cands
        all_candidates.extend(cands)
    unique = dedup_semantic(all_candidates, embedder, MERGE_AGGRESSION)
    return {"per_page_candidates": per_page, "unique_candidates": unique}

# --------------------------------------------------
# do model b AKA compress the list
def ollama_generate(model, prompt, host: str = "http://localhost:11434"):
    resp = requests.post(f"{host}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]

# --------------------------------------------------

def compress_concepts_with_llm(candidates, context_hint: str = ""):
    prompt = f"""
            You are a teaching assistant on creating a list of key topics for an upcoming exam for your students. From this list
            of topics, we get what the most relevant and essential topics. Given the context from the syllabus, determine which topics
            are going to be on the exam and give a list of what students need to know for the test. Use knowledge about what is typically 
            taught in similar courses and pick terms that appear in the slideshow. If a question were to appear on an exam about a topic
            in this slideshow, you should directly be able to point to a concept and in turn, a slide. The list should be concise and.
            be of 'bite-size' topics, suitable as slide tags. Merge synonyms, keep domain terms. Only return a JSON array of strings.
            Do not add explanations.

            Context: {context_hint[:800]}

            Candidates:
            {json.dumps(candidates[:400], ensure_ascii=False, indent=2)}
        """
    # prompt the ollama model
    out = ollama_generate(OLLAMA_MODEL, prompt)
    match = re.search(r'\[.*\]', out, flags=re.S)
    if not match:
        raise RuntimeError(f"llm didn't output json:\n{out[:500]}")
    arr = json.loads(match.group(0))
    return [clean_phrase(x) for x in arr if isinstance(x, str) and 1 <= len(x) <= 60]

# --------------------------------------------------
# align the model based off the sylly
def align_to_syllabus(lecture_topics, syllabus_topics, embedder: SentenceTransformer, thr):
    if not syllabus_topics:
        return [], lecture_topics
    A = embedder.encode(lecture_topics, normalize_embeddings=True, convert_to_numpy=True)
    B = embedder.encode(syllabus_topics, normalize_embeddings=True, convert_to_numpy=True)
    sims = A @ B.T
    best = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(lecture_topics)), best]
    aligned, novel = [], []
    for i, topic in enumerate(lecture_topics):
        if best_sim[i] >= thr:
            aligned.append({
                "lecture_topic": topic,
                "syllabus_topic": syllabus_topics[best[i]],
                "similarity": float(best_sim[i])
            })
        else:
            novel.append(topic)
    return aligned, novel

# --------------------------------------------------
# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--syllabus", required=True)
    ap.add_argument("--out", default="concepts.json")
    args = ap.parse_args()

    embedder = SentenceTransformer(EMBED_MODEL)

    lecture_res = run_model_a(args.pdf, embedder)
    lecture_candidates = lecture_res["unique_candidates"]

    syllabus_topics = []
    if args.syllabus:
        syllabus_res = run_model_a(args.syllabus, embedder)
        syllabus_topics = compress_concepts_with_llm(syllabus_res["unique_candidates"])

    lecture_short = compress_concepts_with_llm(lecture_candidates, context_hint="; ".join(syllabus_topics) if syllabus_topics else "")

    aligned, novel = align_to_syllabus(lecture_short, syllabus_topics, embedder, SYLLY_ALIGNMENT)

    out = {
        "config": {
            "embed_model": EMBED_MODEL,
            "ngram_range": LENGTH_OF_CONCEPT,
            "dedup_sim": MERGE_AGGRESSION,
            "align_sim": SYLLY_ALIGNMENT,
            "compressor_model": OLLAMA_MODEL
        },
        "lecture_pdf": args.pdf,
        "syllabus_pdf": args.syllabus,
        "modelA": {
            "per_page_candidates": lecture_res["per_page_candidates"],
            "unique_candidates": lecture_candidates
        },
        "modelB": {
            "lecture_topics_short": lecture_short,
            "syllabus_topics_short": syllabus_topics
        },
        "alignment": {
            "aligned": aligned,
            "novel": novel
        }
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out}\n- lecture candidates: {len(lecture_candidates)}"
          f"\n- lecture short topics: {len(lecture_short)}"
          f"\n- aligned: {len(aligned)} | novel: {len(novel)}")

if __name__ == "__main__":
    main()
