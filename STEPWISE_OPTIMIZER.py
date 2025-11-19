import os
import glob
import math
import re
from collections import Counter
import numpy as np
import pandas as pd
import statsmodels.api as sm

###############################################################
# LOAD CONCEPTS
###############################################################
def load_concepts(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

###############################################################
# F1 SCORE CALCULATION
###############################################################
def compute_f1(gold, pred):
    gold_set, pred_set = set(gold), set(pred)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return prec, rec, f1

###############################################################
# BASIC TEXT UTILITIES
###############################################################
def words_in(text):
    return re.findall(r"[A-Za-z0-9_]+", text.lower())

def compute_entropy(tokens):
    if len(tokens) == 0:
        return 0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c/total)*math.log((c/total)+1e-12) for c in counts.values())

###############################################################
# METRIC EXTRACTION (unchanged from earlier script)
###############################################################
def compute_text_metrics(text):
    lines = [l.strip() for l in text.split("\n")]
    nonempty = [l for l in lines if l]
    words = words_in(text)
    unique_words = set(words)

    line_ent = [compute_entropy(words_in(l)) for l in nonempty] if nonempty else [0]
    vocab_entropy = compute_entropy(words)

    concept_density = len(unique_words)/max(1,len(words))
    entropy_linelength_interaction = vocab_entropy * (sum(len(l) for l in nonempty)/max(1,len(nonempty)))

    local_entropy_slope = 0
    if len(line_ent) > 1:
        diffs=[line_ent[i]-line_ent[i-1] for i in range(1,len(line_ent))]
        local_entropy_slope=np.mean(diffs)

    line_compactness = np.mean([len(words_in(l)) for l in nonempty]) if nonempty else 0
    fragmentation_index = np.var([len(words_in(l)) for l in nonempty]) if nonempty else 0
    information_per_line = len(unique_words)/max(1,len(nonempty))
    verbosity_penalty = np.mean([len(l) for l in nonempty]) if nonempty else 0
    compositionality_score = np.mean([1/(1+len(words_in(l))) for l in nonempty]) if nonempty else 0

    unique_burst = np.var(line_ent)
    vocabulary_burstiness = np.var([len(set(words_in(l))) for l in nonempty]) if nonempty else 0

    headword_concentration = sum(1 for w in words if w in ["system","algorithm","model","data"]) / max(1,len(words))
    lexical_novelty = len([w for w in words if words.count(w)==1]) / max(1,len(words))
    definition_frequency = len(re.findall(r"\bis\b|\bdefined as\b", text.lower()))
    enumerative_structure_index = sum(1 for l in nonempty if re.match(r"^[-*•0-9]", l))
    linebreak_purity = np.mean([l.endswith((".", ";", ":")) for l in nonempty]) if nonempty else 0
    blankline_clarity = sum(1 for l in lines if l=="") / max(1,len(lines))

    discourse_mode = 1 if "example" in text.lower() else 2 if "definition" in text.lower() else 0
    example_count=text.lower().count("example")
    def_count=text.lower().count("definition")
    def_example_ratio = def_count/(example_count+1)

    referential_ambiguity = len(re.findall(r"\b(this|that|it)\b", text.lower()))
    abstraction_depth = len(re.findall(r"(theorem|proof|lemma|abstract|general)", text.lower()))
    symbol_density = len(re.findall(r"[=<>/*+]", text))/max(1,len(text))
    greek_letter_ratio = len(re.findall(r"(alpha|beta|gamma|delta|lambda)", text.lower()))
    formal_token_ratio = len(re.findall(r"[qQ][0-9]", text))

    punct=re.findall(r"[.,;:?!]", text)
    pos_entropy=compute_entropy(punct+words)

    syntactic_complexity = len(re.findall(r",|;| and | or ", text))
    parse_depth_variance = np.var([len(l.split(",")) for l in nonempty]) if nonempty else 0

    semantic_jumpiness=0
    if len(nonempty)>1:
        vals=[]
        for i in range(1,len(nonempty)):
            s1=set(words_in(nonempty[i-1])); s2=set(words_in(nonempty[i]))
            j=len(s1&s2)/max(1,len(s1|s2))
            vals.append(1-j)
        semantic_jumpiness=np.mean(vals)

    interline_drift=semantic_jumpiness
    topical_focus=len(unique_words)/max(1,len(words))
    concept_coherence = len(unique_words)/max(1, np.var([len(words_in(l)) for l in nonempty])+1)
    definition_boundary_clarity = definition_frequency/max(1,syntactic_complexity+1)

    redundancy = sum(1 for w in unique_words if words.count(w)>3)
    slide_compression_potential=1/(1+redundancy)
    redundancy_score = redundancy/max(1,len(unique_words))

    info_flow_sharpness=np.std([len(words_in(l)) for l in nonempty]) if nonempty else 0

    avg_line_length = np.mean([len(l) for l in nonempty]) if nonempty else 0
    bullet_count = sum(1 for l in nonempty if re.match(r"^[-•*0-9]+\.", l))
    short_line_ratio=len([l for l in nonempty if len(l)<25])/max(1,len(nonempty))
    noise_chars=re.findall(r"[^A-Za-z0-9\s.,;:?!()'\-/*+=<>]", text)
    noise_ratio=len(noise_chars)/max(1,len(text))
    cleanliness_score=1-noise_ratio
    burstiness=unique_burst

    return {
        "concept_density": concept_density,
        "entropy_linelength_interaction": entropy_linelength_interaction,
        "local_entropy_slope": local_entropy_slope,
        "line_compactness": line_compactness,
        "fragmentation_index": fragmentation_index,
        "information_per_line": information_per_line,
        "verbosity_penalty": verbosity_penalty,
        "compositionality_score": compositionality_score,
        "unique_term_burstiness": unique_burst,
        "vocabulary_burstiness": vocabulary_burstiness,
        "headword_concentration": headword_concentration,
        "lexical_novelty": lexical_novelty,
        "definition_frequency": definition_frequency,
        "enumerative_structure_index": enumerative_structure_index,
        "linebreak_purity": linebreak_purity,
        "blankline_clarity": blankline_clarity,
        "discourse_mode": discourse_mode,
        "def_example_ratio": def_example_ratio,
        "referential_ambiguity": referential_ambiguity,
        "abstraction_depth": abstraction_depth,
        "symbol_density": symbol_density,
        "greek_letter_ratio": greek_letter_ratio,
        "formal_token_ratio": formal_token_ratio,
        "pos_entropy": pos_entropy,
        "syntactic_complexity": syntactic_complexity,
        "parse_depth_variance": parse_depth_variance,
        "semantic_jumpiness": semantic_jumpiness,
        "interline_drift": interline_drift,
        "topical_focus": topical_focus,
        "concept_coherence": concept_coherence,
        "definition_boundary_clarity": definition_boundary_clarity,
        "slide_compression_potential": slide_compression_potential,
        "redundancy_score": redundancy_score,
        "info_flow_sharpness": info_flow_sharpness,
        "avg_line_length": avg_line_length,
        "bullet_count": bullet_count,
        "vocab_entropy": vocab_entropy,
        "short_line_ratio": short_line_ratio,
        "cleanliness_score": cleanliness_score,
        "burstiness": burstiness,
    }

###############################################################
# DATA IMPORT
###############################################################

MODEL_DIR = "model1-outputs"
SLIDES_DIR = "SLIDES"

def analyze_class(class_number):
    cs=f"{SLIDES_DIR}/CS-{class_number}"
    gold=f"gold-outputs/outputs-{class_number}"
    pred=f"{MODEL_DIR}/outputs-{class_number}"

    if not os.path.exists(cs):
        return []

    raw=sorted(glob.glob(os.path.join(cs,"*.txt")))
    out=[]

    for path in raw:
        name=os.path.splitext(os.path.basename(path))[0]
        gold_c=load_concepts(os.path.join(gold,f"{name}_annotations.txt"))
        pred_c=load_concepts(os.path.join(pred,f"{name}_annotations.txt"))
        _,_,f1=compute_f1(gold_c,pred_c)

        with open(path,"r",encoding="utf-8") as f:
            text=f.read()

        feats=compute_text_metrics(text)
        feats["class"]=class_number
        feats["slide"]=name
        feats["f1"]=f1
        out.append(feats)

    return out

def find_all_classes():
    out = set()
    gold_root = "gold-outputs"
    
    for d in os.listdir(gold_root):
        if not d.startswith("outputs-"):
            continue

        num = d.replace("outputs-", "")
        
        cs_dir = f"{SLIDES_DIR}/CS-{num}"
        gold_dir = f"gold-outputs/outputs-{num}"
        pred_dir = f"{MODEL_DIR}/outputs-{num}"

        # Require all three to exist AND contain matching files
        if not (os.path.exists(cs_dir) and os.path.exists(gold_dir) and os.path.exists(pred_dir)):
            continue

        # Ensure at least one prediction file exists
        pred_files = glob.glob(os.path.join(pred_dir, "*_annotations.txt"))
        gold_files = glob.glob(os.path.join(gold_dir, "*_annotations.txt"))
        cs_files = glob.glob(os.path.join(cs_dir, "*.txt"))

        if len(pred_files) == 0:
            continue
        if len(gold_files) == 0:
            continue
        if len(cs_files) == 0:
            continue

        # Require that the filenames match at least one slide
        pred_names = {os.path.basename(f).replace("_annotations.txt", "") for f in pred_files}
        gold_names = {os.path.basename(f).replace("_annotations.txt", "") for f in gold_files}
        cs_names   = {os.path.splitext(os.path.basename(f))[0] for f in cs_files}

        # intersection = slides with complete data
        common = pred_names & gold_names & cs_names
        if len(common) == 0:
            continue

        out.add(num)

    return sorted(out)


###############################################################
# STEPWISE AIC SELECTION
###############################################################
def stepwise_aic(df, response, candidates):
    selected=[]
    current_aic = sm.OLS(df[response], sm.add_constant(pd.DataFrame(index=df.index))).fit().aic

    improved=True
    while improved:
        improved=False

        # FORWARD
        remaining=[c for c in candidates if c not in selected]
        best_forward=None
        best_aic_f=current_aic

        for col in remaining:
            X=sm.add_constant(df[selected+[col]])
            model=sm.OLS(df[response],X).fit()
            if model.aic < best_aic_f:
                best_aic_f=model.aic
                best_forward=col

        if best_forward is not None and best_aic_f < current_aic:
            selected.append(best_forward)
            current_aic = best_aic_f
            improved=True
            continue

        # BACKWARD
        if len(selected)>1:
            best_backward=None
            best_aic_b=current_aic

            for col in selected:
                trial=[c for c in selected if c!=col]
                X=sm.add_constant(df[trial])
                model=sm.OLS(df[response],X).fit()
                if model.aic < best_aic_b:
                    best_aic_b=model.aic
                    best_backward=col

            if best_backward is not None and best_aic_b < current_aic:
                selected.remove(best_backward)
                current_aic = best_aic_b
                improved=True

    return selected, current_aic

###############################################################
# MAIN
###############################################################
def main():
    all_results=[]
    for c in find_all_classes():
        all_results.extend(analyze_class(c))

    df=pd.DataFrame(all_results)
    df.to_csv("all_metrics_raw_output.csv",index=False)

    metric_cols=[c for c in df.columns if c not in ["class","slide","f1"]]

    ###############################################################
    # STEP 1 — UNIVARIATE FILTERING (p < .05)
    ###############################################################
    univ=[]
    significant=[]
    for m in metric_cols:
        X=sm.add_constant(df[m])
        model=sm.OLS(df["f1"],X).fit()
        p=model.pvalues[m]

        univ.append({
            "metric":m,
            "slope":model.params[m],
            "std_error":model.bse[m],
            "t_value":model.tvalues[m],
            "p_value":p,
            "r_squared":model.rsquared,
            "significant_p<0.05":p<0.05
        })

        if p < 0.05:
            significant.append(m)

    pd.DataFrame(univ).to_csv("univariate_significance.csv",index=False)
    pd.DataFrame({"metric":significant}).to_csv("stepwise_candidates.csv",index=False)

    print("\nUnivariate filtering complete.")
    print("Significant candidates:", significant)

    ###############################################################
    # STEP 2 — RUN STEPWISE AIC ON SIGNIFICANT METRICS ONLY
    ###############################################################
    if len(significant)==0:
        print("\nNo significant metrics available for stepwise selection.")
        return

    selected, final_aic = stepwise_aic(df, "f1", significant)

    Xf=sm.add_constant(df[selected])
    final_model = sm.OLS(df["f1"],Xf).fit()

    out=pd.DataFrame({
        "variable":["intercept"]+selected,
        "coef":final_model.params.values,
        "std_err":final_model.bse.values,
        "t_value":final_model.tvalues.values,
        "p_value":final_model.pvalues.values
    })
    out.to_csv("stepwise_final_model_summary.csv",index=False)

    print("\n===== STEPWISE SELECTION COMPLETE =====")
    print("Final Selected Variables:", selected)
    print("Final AIC:", final_aic)
    print("Saved: stepwise_final_model_summary.csv\n")


if __name__=="__main__":
    main()
