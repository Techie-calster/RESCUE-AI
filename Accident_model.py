"""
RESCUE-AI - Accident model (accident vs non-accident)
Tuned for MacBook Air M2 / 8 GB RAM.

Setup:
    pip install ultralytics

Folder layout (put the Roboflow download here, next to this script):
    accident_dataset/
        data.yaml
        train/images, train/labels
        valid/images, valid/labels
        test/images,  test/labels

Usage:
    python accident_model.py check              # 1) inspect labels, get a recommendation
    python accident_model.py train classify     # 2a) image-level model (recommended if check says so)
    python accident_model.py train detect       # 2b) bounding-box model
    python accident_model.py predict path/to/photo.jpg
"""

import os
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # let unsupported ops fall back to CPU

DATASET_DIR = Path("accident_dataset").resolve()
CLS_DIR = Path("accident_cls_dataset").resolve()
NAMES = ["accident", "non-accident"]  # must match data.yaml order
SPLITS = {"train": "train", "valid": "val", "test": "test"}  # roboflow name -> ultralytics name
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------------ 1) check
def read_label(path: Path):
    """Return list of (class_id, w, h) for a YOLO label file."""
    rows = []
    if path.exists():
        for ln in path.read_text().strip().splitlines():
            p = ln.split()
            if len(p) >= 5:
                rows.append((int(p[0]), float(p[3]), float(p[4])))
    return rows


def check():
    for split in SPLITS:
        img_dir, lbl_dir = DATASET_DIR / split / "images", DATASET_DIR / split / "labels"
        if not img_dir.exists():
            print(f"[{split}] missing"); continue
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        cls_count, empty, areas = Counter(), 0, []
        for img in imgs:
            rows = read_label(lbl_dir / (img.stem + ".txt"))
            if not rows:
                empty += 1
            for c, w, h in rows:
                cls_count[NAMES[c]] += 1
                areas.append(w * h)
        avg = sum(areas) / len(areas) if areas else 0
        print(f"[{split}] images={len(imgs)} empty_labels={empty} boxes={dict(cls_count)} "
              f"avg_box_area={avg:.2f} (fraction of image)")
        if split == "train":
            if avg > 0.75:
                print("  -> Boxes cover most of the image: use `train classify`.")
            else:
                print("  -> Boxes are localized: `train detect` is reasonable.")


# ------------------------------------------------------------------ 2a) classify
def build_classification_dataset():
    """Convert YOLO labels to class folders (majority class per image)."""
    if CLS_DIR.exists():
        shutil.rmtree(CLS_DIR)
    skipped = 0
    for split, out_split in SPLITS.items():
        img_dir, lbl_dir = DATASET_DIR / split / "images", DATASET_DIR / split / "labels"
        if not img_dir.exists():
            continue
        for name in NAMES:
            (CLS_DIR / out_split / name).mkdir(parents=True, exist_ok=True)
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            rows = read_label(lbl_dir / (img.stem + ".txt"))
            if not rows:
                skipped += 1
                continue
            label = NAMES[Counter(c for c, _, _ in rows).most_common(1)[0][0]]
            shutil.copy(img, CLS_DIR / out_split / label / img.name)
    print(f"Classification dataset ready at {CLS_DIR} (skipped {skipped} unlabeled images)")


def train_classify():
    from ultralytics import YOLO
    build_classification_dataset()
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=str(CLS_DIR), epochs=30, imgsz=224, batch=32,
        device="mps", workers=2, patience=8, project="runs", name="accident_cls",
    )


# ------------------------------------------------------------------ 2b) detect
def train_detect():
    from ultralytics import YOLO
    fixed = DATASET_DIR / "data_fixed.yaml"  # roboflow's relative ../ paths often break
    fixed.write_text(
        f"path: {DATASET_DIR}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n"
    )
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(fixed), epochs=40, imgsz=640, batch=8,
        device="mps", workers=2, patience=10, cache=False,
        project="runs", name="accident_det",
    )


# ------------------------------------------------------------------ 3) predict
# def predict(image_path: str):
#     from ultralytics import YOLO
#     for kind in ("accident_cls", "accident_det"):
#         w = Path("runs") / kind / "weights" / "best.pt"
#         if w.exists():
#             model = YOLO(str(w))
#             r = model.predict(image_path, device="mps", verbose=False)[0]
#             if kind == "accident_cls":
#                 top = r.probs.top1
#                 print({"accident_detected": r.names[top] == "accident",
#                        "confidence": round(float(r.probs.top1conf), 3)})
#             else:
#                 boxes = [(r.names[int(b.cls)], round(float(b.conf), 3)) for b in r.boxes]
#                 print(boxes or "no detections")
#             return
#     print("No trained weights found in runs/. Train first.")


# if __name__ == "__main__":
#     cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
#     if cmd == "check":
#         check()
#     elif cmd == "train":
#         (train_detect if (len(sys.argv) > 2 and sys.argv[2] == "detect") else train_classify)()
#     elif cmd == "predict":
#         predict(sys.argv[2])
#     else:
#         print(__doc__)

    
# ------------------------------------------------------------------ 3) predict
ACCIDENT_CONF = 0.35  # tune this: higher = fewer false alarms, lower = fewer misses
 
 
def find_weights():
    """Ultralytics may nest outputs (e.g. runs/detect/runs/accident_det), so search."""
    found = sorted(Path(".").glob("runs/**/accident_*/weights/best.pt"),
                   key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None
 
 
def predict(image_path: str):
    from ultralytics import YOLO
    w = find_weights()
    if w is None:
        print("No trained weights found under runs/. Train first."); return
    model = YOLO(str(w))
    r = model.predict(image_path, device="mps", conf=0.1, verbose=False)[0]
    if r.probs is not None:  # classification weights
        top = r.probs.top1
        print({"accident_detected": r.names[top] == "accident",
               "confidence": round(float(r.probs.top1conf), 3)})
        return
    acc = [float(b.conf) for b in r.boxes if r.names[int(b.cls)] == "accident"]
    best = max(acc, default=0.0)
    print({"accident_detected": best >= ACCIDENT_CONF, "confidence": round(best, 3)})
    r.save(filename="prediction.jpg")  # image with boxes drawn
    print("Saved annotated image: prediction.jpg  | weights used:", w)
 
 
def evaluate_test():
    from ultralytics import YOLO
    w = find_weights()
    YOLO(str(w)).val(data=str(DATASET_DIR / "data_fixed.yaml"), split="test", device="mps")
 
def image_level_report():
    """Image-level accuracy on the test split, at several confidence thresholds.
    Ground truth: image is 'accident' if its label file has any accident box."""
    from ultralytics import YOLO
    w = find_weights()
    model = YOLO(str(w))
    img_dir, lbl_dir = DATASET_DIR / "test" / "images", DATASET_DIR / "test" / "labels"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    truth, best = [], []
    for img in imgs:
        rows = read_label(lbl_dir / (img.stem + ".txt"))
        truth.append(any(NAMES[c] == "accident" for c, _, _ in rows))
        r = model.predict(str(img), device="mps", conf=0.05, verbose=False)[0]
        confs = [float(b.conf) for b in r.boxes if r.names[int(b.cls)] == "accident"]
        best.append(max(confs, default=0.0))
    print(f"{len(imgs)} test images | {sum(truth)} with accident, {len(truth) - sum(truth)} without")
    print("thr   TP   FP   FN   TN  recall precision")
    for thr in (0.2, 0.3, 0.35, 0.4, 0.5, 0.6):
        tp = sum(t and b >= thr for t, b in zip(truth, best))
        fp = sum((not t) and b >= thr for t, b in zip(truth, best))
        fn = sum(t and b < thr for t, b in zip(truth, best))
        tn = sum((not t) and b < thr for t, b in zip(truth, best))
        rec = tp / (tp + fn) if tp + fn else 0
        pre = tp / (tp + fp) if tp + fp else 0
        print(f"{thr:<5} {tp:<4} {fp:<4} {fn:<4} {tn:<4} {rec:.2f}   {pre:.2f}")
 
 
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        check()
    elif cmd == "train":
        (train_detect if (len(sys.argv) > 2 and sys.argv[2] == "detect") else train_classify)()
    elif cmd == "predict":
        predict(" ".join(sys.argv[2:]))  # tolerates unquoted paths with spaces
    elif cmd == "test":
        evaluate_test()
    elif cmd == "report":
        image_level_report()
    else:
        print(__doc__)
 