#!/usr/bin/env python3
# Extract keyphrases with HF INSPEC (token-classification), then rank with your combined scorer.
import argparse, json
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ---- HF INSPEC extractor ----
def extract_hf_keyphrases(text: str, top_k: int = 20):
    nlp = pipeline("token-classification",
                   model="ml6team/keyphrase-extraction-distilbert-inspec",
                   aggregation_strategy="simple")
    spans = nlp(text)
    # dedupe by phrase (case-insensitive), keep max score
    best = {}
    for s in spans:
        phrase = s["word"].strip()
        if not phrase: continue
        key = phrase.lower()
        score = float(s.get("score", 0.0))
        if key not in best or score > best[key]["score"]:
            best[key] = {"phrase": phrase, "score": score}
    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    return [r["phrase"] for r in ranked[:top_k]] if top_k else [r["phrase"] for r in ranked]

# ---- Combined scorer (xenc + YAKE prior) ----
def predict_prob_xenc(model_dir: str, text: str, candidate: str) -> float:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    enc = tok(f"[CAND] {candidate} </s> [CTX] {text}",
              return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        logits = model(**enc).logits  # [1,2]
    exps = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    probs = exps / exps.sum(dim=-1, keepdim=True)
    return float(probs[0, 1].item())

def get_yake_score(text: str, candidate: str, n_min: int = 1, n_max: int = 3) -> Optional[float]:
    import yake
    cand = candidate.strip().lower()
    best = None
    for n in range(n_min, n_max + 1):
        kw = yake.KeywordExtractor(lan="en", n=n, top=100000)
        for term, score in kw.extract_keywords(text):
            if term.strip().lower() == cand:
                best = score if best is None else min(best, score)
    return best

def normalize_yake(score: float, alpha: float = 12.0) -> float:
    return 1.0 / (1.0 + alpha * float(score))

def fuzzy_presence(text: str, cand: str) -> float:
    try:
        from rapidfuzz.fuzz import token_set_ratio
        return token_set_ratio(text.lower(), cand.lower()) / 100.0
    except Exception:
        return 1.0 if cand.lower() in text.lower() else 0.0

def nounish_penalty(candidate: str) -> float:
    single = len(candidate.split()) == 1
    looks_nouny = candidate[:1].isupper() or candidate.isupper()
    return 1.0 if (not single or looks_nouny) else 0.9

def combined_prob(model_dir: str, text: str, candidate: str,
                  lambda_weight: float = 0.85, alpha: float = 12.0,
                  n_min: int = 1, n_max: int = 3) -> float:
    p_xenc = predict_prob_xenc(model_dir, text, candidate)
    score = get_yake_score(text, candidate, n_min=n_min, n_max=n_max)
    if score is None: score = 1.0
    p_yake = normalize_yake(score, alpha=alpha)
    presence = fuzzy_presence(text, candidate)
    p_yake = (0.7 * p_yake + 0.3 * presence) * nounish_penalty(candidate)
    return float(lambda_weight * p_xenc + (1.0 - lambda_weight) * p_yake)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="your trained xenc selector dir")
    ap.add_argument("--text_file", required=True)
    ap.add_argument("--top_k", type=int, default=20, help="HF phrases to consider")
    ap.add_argument("--lambda_weight", type=float, default=0.85)
    ap.add_argument("--alpha", type=float, default=12.0)
    ap.add_argument("--n_min", type=int, default=1)
    ap.add_argument("--n_max", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out_file", type=str, default=None, help="optional: write top-N phrases to file")
    args = ap.parse_args()

    with open(args.text_file, "r", encoding="utf-8") as f:
        paragraph = f.read()

    cands = extract_hf_keyphrases(paragraph, top_k=args.top_k)
    rows = []
    for c in cands:
        p = combined_prob(args.model_dir, paragraph, c,
                          lambda_weight=args.lambda_weight, alpha=args.alpha,
                          n_min=args.n_min, n_max=args.n_max)
        rows.append({"candidate": c, "p_final": p})

    rows.sort(key=lambda r: r["p_final"], reverse=True)

    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(f"{r['candidate']}\t{r['p_final']:.4f}\n")

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        width = max(12, max((len(r["candidate"]) for r in rows), default=12))
        print(f"{'candidate'.ljust(width)}  {'p_final':>8}")
        print("-"*(width+12))
        for r in rows:
            print(f"{r['candidate'].ljust(width)}  {r['p_final']:8.3f}")

if __name__ == "__main__":
    main()
