"""
RESCUE-AI - Accident model (one class: Accident). Works on Colab/Kaggle (CUDA), Apple MPS or CPU.

Environment variables:
    DATASET_DIR=/path/to/dataset   the folder containing data.yaml, train/, valid/, test/
    RUNS_DIR=/path/to/runs         where training output is saved (default: runs). On Colab point this at Google Drive
                                   so checkpoints survive a disconnect.
    WEIGHTS=/path/to/best.pt       use these weights for report/falsealarm/predict instead of the newest run

Commands:
    python accident_model.py check                     # inspect the dataset
    python accident_model.py negatives <folder> [N] [max]  # add accident-free photos to training; hold out N (default 60) for testing; use at most `max` for training
    python accident_model.py negatives_each <parent> [fraction] [max]  # same, for every sub-folder: hold out `fraction` (default 0.2), use at most `max` (default 600) per folder
    python accident_model.py train [epochs] [imgsz] [ram|disk]  # default 80 epochs, size 800. ram/disk caches decoded images (faster when data loading is the bottleneck)
    python accident_model.py resume                    # continue an interrupted run
    python accident_model.py report                    # test-set results + saved mistakes
    python accident_model.py falsealarm <folder>       # how often accident-free photos are flagged
    python accident_model.py predict <image or folder> # accident / no accident; folders are saved to predictions/
"""

import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DATASET_DIR = Path(os.environ.get("DATASET_DIR", "accident_dataset")).resolve()
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
ACCIDENT_CONF = 0.3  # chosen from report + falsealarm: ~98% of test accidents found, ~5% of normal photos flagged
RUNS_DIR = Path(os.environ.get("RUNS_DIR", "runs")).resolve()
NEG_TRAIN = Path("negatives_train").resolve()      # accident-free photos used for training
NEG_HOLDOUT = Path("negatives_holdout").resolve()  # accident-free photos kept for testing only
YAML_FIXED = Path("data_fixed.yaml").resolve()
MAX_SIDE = 800  # normal photos are shrunk to this longest side when added (big JPEGs slow training down a lot)


def dev():
    """CUDA GPU > Apple MPS > CPU."""
    import torch
    return 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ------------------------------------------------------------------ dataset helpers
def find_split(split):
    """Folder with images/ and labels/ for a split (handles train/ and the nested train/train/ layout)."""
    for cand in (DATASET_DIR / split, DATASET_DIR / split / split):
        if (cand / "images").exists():
            return cand
    for p in sorted(DATASET_DIR.rglob("images")):
        if p.parent.name == split:
            return p.parent
    raise SystemExit(f"Could not find '{split}/images' under {DATASET_DIR}. Check DATASET_DIR.")


def class_names():
    import yaml
    for p in [DATASET_DIR / "data.yaml", *DATASET_DIR.rglob("data.yaml")]:
        if p.exists():
            names = yaml.safe_load(p.read_text()).get("names")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=int)]
            if names:
                return list(names)
    return ["Accident"]


def label_rows(path):
    """[(class_id, w, h)] from one YOLO label file."""
    rows = []
    if path.exists():
        for ln in path.read_text().strip().splitlines():
            p = ln.split()
            if len(p) >= 5:
                rows.append((int(p[0]), float(p[3]), float(p[4])))
    return rows


def images_in(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)


# ------------------------------------------------------------------ check
def check():
    names = class_names()
    print("Dataset:", DATASET_DIR, "\nClasses:", names)
    for split in ("train", "valid", "test"):
        try:
            d = find_split(split)
        except SystemExit:
            print(f"[{split}] missing"); continue
        imgs = images_in(d / "images")
        boxes, empty, areas = Counter(), 0, []
        for p in imgs:
            rows = label_rows(d / "labels" / (p.stem + ".txt"))
            empty += not rows
            for c, w, h in rows:
                boxes[names[c]] += 1
                areas.append(w * h)
        avg = sum(areas) / len(areas) if areas else 0
        print(f"[{split}] images={len(imgs)} accident-free(empty labels)={empty} boxes={dict(boxes)} avg_box_area={avg:.2f}")


# ------------------------------------------------------------------ accident-free photos (negatives)
def save_small(src, dst_no_ext):
    """Save `src` as a JPEG whose longest side is at most MAX_SIDE. Returns the new path, or None if unreadable."""
    from PIL import Image, ImageOps
    dst = Path(str(dst_no_ext) + ".jpg")
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))   # keeps aspect ratio, never enlarges
            im.save(dst, "JPEG", quality=92)
        return dst
    except Exception as e:
        print(f"  skipped unreadable file {Path(src).name}: {e}")
        return None


def make_negatives(folder, holdout=60, max_train=None):
    """Add accident-free photos to training as negatives (empty labels) and keep `holdout` of them aside
    for testing. Can be run several times with different folders; the results accumulate."""
    import re
    imgs = images_in(folder)
    if not imgs:
        raise SystemExit(f"No images found in {folder}")
    random.Random(0).shuffle(imgs)
    hold, train = imgs[:holdout], imgs[holdout:]
    if max_train:
        train = train[:max_train]  # cap very large folders so negatives don't swamp the accident images
    tag = re.sub(r"\W+", "_", Path(folder).name)[:20] or "src"  # keeps files from different folders apart
    for d in (NEG_TRAIN / "images", NEG_TRAIN / "labels", NEG_HOLDOUT):
        d.mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(train):
        if save_small(p, NEG_TRAIN / "images" / f"neg_{tag}_{k:04d}"):
            (NEG_TRAIN / "labels" / f"neg_{tag}_{k:04d}.txt").write_text("")  # empty label = no accident
    for k, p in enumerate(hold):
        save_small(p, NEG_HOLDOUT / f"hold_{tag}_{k:04d}")
    n_train = len(images_in(NEG_TRAIN / "images"))
    n_hold = len(images_in(NEG_HOLDOUT))
    print(f"This folder: {len(train)} -> training negatives, {len(hold)} -> held out for testing")
    print(f"Totals now: {n_train} training negatives in {NEG_TRAIN}, {n_hold} held-out photos in {NEG_HOLDOUT}")


def make_negatives_each(parent, holdout_fraction=0.2, max_train=600):
    """Run make_negatives on every sub-folder of `parent` (e.g. car/, bus/, scooter/ ...).
    Each sub-folder keeps `holdout_fraction` of its photos for testing and contributes at most `max_train`
    photos to training, so one big folder (e.g. car) can't crowd out the others."""
    parent = Path(parent)
    subs = sorted(d for d in parent.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not subs:
        raise SystemExit(f"No sub-folders found in {parent}. For a single folder use: negatives <folder>")
    for d in subs:
        n = len(images_in(d))
        if n == 0:
            print(f"== {d.name}: no images, skipped"); continue
        print(f"== {d.name} ({n} photos)")
        make_negatives(d, max(1, round(n * holdout_fraction)), max_train)


# ------------------------------------------------------------------ training
def write_yaml():
    import json
    train_dirs = [str(find_split("train") / "images")]
    if (NEG_TRAIN / "images").exists():
        train_dirs.append(str(NEG_TRAIN / "images"))
        print("Including negatives from", NEG_TRAIN)
    names = class_names()
    YAML_FIXED.write_text(
        f"train: {json.dumps(train_dirs)}\n"
        f"val: {json.dumps(str(find_split('valid') / 'images'))}\n"
        f"test: {json.dumps(str(find_split('test') / 'images'))}\n"
        f"nc: {len(names)}\nnames: {json.dumps(names)}\n"
    )
    return YAML_FIXED


def train(epochs=80, imgsz=800, cache=False):
    from ultralytics import YOLO
    data = write_yaml()
    gpu = dev() == 0
    YOLO("yolo11n.pt").train(
        data=str(data), epochs=epochs, imgsz=imgsz,
        batch=(16 if imgsz <= 640 else 8) if gpu else (8 if imgsz <= 640 else 4),
        device=dev(), workers=4 if gpu else 2, patience=20, cache=cache,
        project=str(RUNS_DIR), name=f"accident_det_{imgsz}",
    )


def newest(pattern):
    found = sorted(RUNS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def find_weights():
    if os.environ.get("WEIGHTS"):
        return Path(os.environ["WEIGHTS"])
    w = newest("**/accident_*/weights/best.pt")
    if w is None:
        raise SystemExit(f"No trained weights found under {RUNS_DIR}. Train first (or set WEIGHTS=/path/to/best.pt).")
    return w


def resume():
    from ultralytics import YOLO
    last = newest("**/accident_*/weights/last.pt")
    if last is None:
        raise SystemExit(f"No last.pt found under {RUNS_DIR}.")
    write_yaml()  # recreates data_fixed.yaml, which a fresh Colab/Kaggle session no longer has
    print("Resuming from", last)
    YOLO(str(last)).train(resume=True)


# ------------------------------------------------------------------ inference / evaluation
def accident_conf(model, img, conf=0.05):
    r = model.predict(str(img), device=dev(), conf=conf, verbose=False)[0]
    return max((float(b.conf) for b in r.boxes), default=0.0)  # single class: every box is an accident


def predict(path):
    """Predict one image OR every image in a folder.
    Results go to predictions/:  accident/ (flagged, boxes drawn)  no_accident/ (copies)  results.csv
    File names start with the confidence, so they sort by score."""
    import csv
    from ultralytics import YOLO
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Not found: {path!r}")
    imgs = images_in(p) if p.is_dir() else [p]
    if not imgs:
        raise SystemExit(f"No images found in {path}")
    w = find_weights()
    model = YOLO(str(w))
    out = Path("predictions")
    shutil.rmtree(out, ignore_errors=True)  # start fresh each run
    (out / "accident").mkdir(parents=True)
    (out / "no_accident").mkdir()
    rows = []
    for img in imgs:
        try:
            r = model.predict(str(img), device=dev(), conf=0.05, verbose=False)[0]
        except Exception as e:
            print(f"  skipped {img.name}: {e}"); continue
        best = max((float(b.conf) for b in r.boxes), default=0.0)
        detected = best >= ACCIDENT_CONF
        label = "accident" if detected else "no accident"
        name = f"{best:.2f}_{img.name}"
        if detected:  # redraw with only the boxes that count
            model.predict(str(img), device=dev(), conf=ACCIDENT_CONF, verbose=False)[0].save(
                filename=str(out / "accident" / name))
        else:
            shutil.copy(img, out / "no_accident" / name)
        rows.append((img.name, label, round(best, 3)))
        print(f"{label:<12} {best:.2f}  {img.name}")
    with open(out / "results.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image", "label", "accident_confidence"), *rows])
    flagged = sum(r[1] == "accident" for r in rows)
    print(f"\n{len(rows)} images | {flagged} flagged as accident ({flagged / max(1, len(rows)):.1%}) "
          f"| threshold {ACCIDENT_CONF} | weights: {w}")
    print("Saved in predictions/: accident/ (boxes drawn), no_accident/, results.csv")


def report():
    """Image-level results on the test split, and annotated pictures of the mistakes."""
    from ultralytics import YOLO
    w = find_weights()
    model = YOLO(str(w))
    d = find_split("test")
    imgs = images_in(d / "images")
    truth = [bool(label_rows(d / "labels" / (p.stem + ".txt"))) for p in imgs]
    best = [accident_conf(model, p) for p in imgs]
    print(f"weights: {w}\n{len(imgs)} test images | {sum(truth)} with accident, {len(truth) - sum(truth)} without")
    print("thr   TP   FP   FN   TN  recall")
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        tp = sum(t and b >= thr for t, b in zip(truth, best))
        fp = sum((not t) and b >= thr for t, b in zip(truth, best))
        fn = sum(t and b < thr for t, b in zip(truth, best))
        tn = sum((not t) and b < thr for t, b in zip(truth, best))
        print(f"{thr:<5} {tp:<4} {fp:<4} {fn:<4} {tn:<4} {tp / (tp + fn) if tp + fn else 0:.2f}")
    print("Note: false-alarm numbers mean little unless the test set has many accident-free images.")
    out = Path("report_mistakes"); out.mkdir(exist_ok=True)
    for b, p, t in sorted(zip(best, imgs, truth)):
        if t and b < ACCIDENT_CONF or (not t) and b >= ACCIDENT_CONF:
            tag = "MISSED" if t else "FALSE_ALARM"
            print(f"  {tag} {b:.2f} {p.name}")
            model.predict(str(p), device=dev(), conf=0.05, verbose=False)[0].save(filename=str(out / f"{tag}_{p.name}"))
    print("Annotated mistakes saved in", out)


def falsealarm(folder):
    """Run on photos with NO accident; report how many are wrongly flagged."""
    if not folder or "<" in folder or not Path(folder).exists():
        raise SystemExit(f"Folder not found: {folder!r}. Use the real folder path, not a placeholder.")
    from ultralytics import YOLO
    model = YOLO(str(find_weights()))
    imgs = images_in(folder)
    best = [(accident_conf(model, p), p) for p in imgs]
    print(f"{len(imgs)} accident-free images\nthr   false_alarms  rate")
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        k = sum(b >= thr for b, _ in best)
        print(f"{thr:<5} {k:<13} {k / len(imgs):.1%}")
    out = Path("false_alarms"); out.mkdir(exist_ok=True)
    print("Highest-scoring images (annotated copies in false_alarms/):")
    for b, p in sorted(best, reverse=True)[:8]:
        print(f"  {b:.2f}  {p.name}")
        model.predict(str(p), device=dev(), conf=0.05, verbose=False)[0].save(filename=str(out / p.name))


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        check()
    elif cmd == "negatives":
        make_negatives(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 60,
                       int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif cmd == "negatives_each":
        make_negatives_each(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.2,
                            int(sys.argv[4]) if len(sys.argv) > 4 else 600)
    elif cmd == "train":
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 80, int(sys.argv[3]) if len(sys.argv) > 3 else 800,
              sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] in ("ram", "disk") else False)
    elif cmd == "resume":
        resume()
    elif cmd == "report":
        report()
    elif cmd == "falsealarm":
        falsealarm(" ".join(sys.argv[2:]))
    elif cmd == "predict":
        predict(" ".join(sys.argv[2:]))
    else:
        print(__doc__)
