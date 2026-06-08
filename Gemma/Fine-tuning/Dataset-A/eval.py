from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json, os

# ── CONFIG (same scheme as finetuning.py / infer_model.py) ───────────────
PROJECT_DIR = os.environ.get("WSD_PROJECT_DIR", "/content/drive/MyDrive/WSD_Project")
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODEL_TAG   = os.environ.get("WSD_MODEL_TAG", "gemma2_2b")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs", MODEL_TAG)

def evaluate_predictions(predictions_path, ground_truth_path, report_path):
    with open(predictions_path, "r", encoding="utf-8") as pred_file:
        predictions = json.load(pred_file)
    with open(ground_truth_path, "r", encoding="utf-8") as gt_file:
        ground_truth = json.load(gt_file)

    y_true = []
    y_pred = []

    for pred_sentence, gt_sentence in zip(predictions, ground_truth):
        pred_words = pred_sentence["words"]
        gt_words = gt_sentence["words"]

        for pred_word, gt_word in zip(pred_words, gt_words):
            y_true.append(str(gt_word["target_sense"]))
            y_pred.append(str(pred_word["target_sense"]))  # "none" becomes just another string label

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    report = {
        "Accuracy (%)": round(accuracy, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Total Evaluated Instances": len(y_true)
    }

    print(json.dumps(report, indent=4))

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

# Example run
evaluate_predictions(
    predictions_path = os.path.join(OUTPUT_DIR, f"predictions_{MODEL_TAG}.json"),
    ground_truth_path = os.path.join(DATA_DIR, "test_truth.json"),
    report_path = os.path.join(OUTPUT_DIR, f"report_{MODEL_TAG}.json"),
)
