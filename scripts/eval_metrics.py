"""
Calculate F1, Recall, Precision, Accuracy from filenames in a folder.
Expected filename format: img_{idx}_pred_{pred}_label_{label}.png
"""
import re
from pathlib import Path


def parse_folder(folder: str):
    preds, labels = [], []
    pattern = re.compile(r"pred_(\d+)_label_(\d+)")
    for f in Path(folder).iterdir():
        m = pattern.search(f.name)
        if m:
            preds.append(int(m.group(1)))
            labels.append(int(m.group(2)))
    return preds, labels


def compute_metrics(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(preds) if preds else 0.0

    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="/home/locht1/Documents/locht1/Isaac-GR00T/test_pick")
    args = parser.parse_args()

    preds, labels = parse_folder(args.folder)
    print(f"Total samples: {len(preds)}")

    m = compute_metrics(preds, labels)
    print(f"TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
    print(f"Precision : {m['precision']:.4f}")
    print(f"Recall    : {m['recall']:.4f}")
    print(f"F1        : {m['f1']:.4f}")
    print(f"Accuracy  : {m['accuracy']:.4f}")
