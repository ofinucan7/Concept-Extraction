import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# USER SETTINGS 
# ------------------------------------------------------------
GOLD_DIR = "gold-outputs"
MODEL_DIR = "model1-outputs"


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def load_file_lines(path):
    """Load newline-separated items from a file."""
    if not os.path.exists(path):
        return []
    return [line.strip() for line in open(path, "r", encoding="utf-8")
            if line.strip()]


def compute_f1(gold, pred):
    """Set-based F1."""
    g, p = set(gold), set(pred)
    tp = len(g & p)
    fp = len(p - g)
    fn = len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0


# ------------------------------------------------------------
# CLASS DISCOVERY
# ------------------------------------------------------------
def find_matched_classes():
    gold = {d for d in os.listdir(GOLD_DIR) if d.startswith("outputs-")}
    model = {d for d in os.listdir(MODEL_DIR) if d.startswith("outputs-")}
    return sorted(gold & model)


# ------------------------------------------------------------
# PROCESS SINGLE CLASS
# ------------------------------------------------------------
def analyze_class(class_folder):
    """Return:
       - per-slide rows (list of dicts)
       - aggregate gold set
       - aggregate pred set
    """
    gold_path = os.path.join(GOLD_DIR, class_folder)
    model_path = os.path.join(MODEL_DIR, class_folder)

    # gold: slideID_annotations.txt -> slideID
    gold_map = {}
    for p in glob.glob(os.path.join(gold_path, "*.txt")):
        base = os.path.basename(p)
        if base.endswith("_annotations.txt"):
            slide_id = base.replace("_annotations.txt", "")
            gold_map[slide_id] = base

    # model: slideID.txt -> slideID
    model_map = {}
    for p in glob.glob(os.path.join(model_path, "*.txt")):
        base = os.path.basename(p)
        if base.endswith("_annotations.txt"):
            slide_id = base.replace("_annotations.txt", "")
        else:
            slide_id = base.replace(".txt", "")
        model_map[slide_id] = base

    matched_ids = sorted(gold_map.keys() & model_map.keys())

    rows = []
    class_num = class_folder.replace("outputs-", "")

    agg_gold = set()
    agg_pred = set()

    for slide_id in matched_ids:
        gold_file = gold_map[slide_id]
        model_file = model_map[slide_id]

        gold = load_file_lines(os.path.join(gold_path, gold_file))
        pred = load_file_lines(os.path.join(model_path, model_file))

        # accumulate
        agg_gold.update(gold)
        agg_pred.update(pred)

        f1 = compute_f1(gold, pred)
        rows.append({"class": class_num, "f1": f1})

    return rows, agg_gold, agg_pred


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    classes = find_matched_classes()
    all_rows = []
    agg_entries = []   # store aggregate f1 per class

    for folder in classes:
        per_slide_rows, agg_gold, agg_pred = analyze_class(folder)
        all_rows.extend(per_slide_rows)

        # compute aggregate F1 for the entire class
        agg_f1 = compute_f1(list(agg_gold), list(agg_pred))
        class_num = folder.replace("outputs-", "")

        agg_entries.append({
            "class": class_num,
            "aggregate_f1": agg_f1
        })

    if not all_rows:
        print("\nERROR: No valid classes or matched slide files found.")
        return

    df = pd.DataFrame(all_rows)
    df_agg = pd.DataFrame(agg_entries)

    summary = df.groupby("class").agg(
        avg_f1=("f1", "mean"),
        var_f1=("f1", "var")
    ).reset_index()

    # merge in aggregate F1 per class
    final_summary = summary.merge(df_agg, on="class", how="left")

    final_summary.to_csv("class_f1_summary.csv", index=False)
    print("\nSaved: class_f1_summary.csv\n")
    print(final_summary)

    plt.figure(figsize=(7, 5))
    plt.hist(final_summary["avg_f1"], bins=8, edgecolor="black")
    plt.title("Average F1 by Class")
    plt.xlabel("Avg F1")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.savefig("class_f1_histogram.png", dpi=150)
    plt.close()

    print("Saved: class_f1_histogram.png")


if __name__ == "__main__":
    main()
