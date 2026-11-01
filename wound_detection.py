"""
Rescue AI - count injured vs uninjured people in an image. Single-file version.

  python wound_detection .py inspect-fallen [path]   check the fallen dataset
  python wound_detection.py inspect-wound  [path]   check the wound dataset(s)
  python wound_detection.py train-state             YOLO detector: normal vs fallen persons
  python wound_detection.py train-wound             YOLO classifier: wound vs no_wound (2 classes)
  python wound_detection.py infer crash.jpg [--save out.jpg]

How "injured" is decided, per person:
  FALLEN (state detector) OR WOUND (wound classifier finds a wound in any body patch).
  Everyone else is uninjured.

Install:  pip install ultralytics pyyaml pillow opencv-python
"""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageFilter

# ----------------------------- CONFIG ---------------------------------------
WOUND_DS = Path("/kaggle/input/datasets/gauravkumardu01/wound-dataset/segmentation.v1i.yolov11")      # ~1000 images with boxes; every box becomes class "wound"
FALLEN_DS = Path("datasets/fallen")
# Folder of healthy-skin images (no labels). Each image becomes ONE "no_wound" patch, used as it is (no cropping).
NORMAL_SKIN_DS = Path("/kaggle/input/datasets/gauravkumardu01/normal-skin-dataset/Normal_skin")
WORK = Path("datasets/_prepared")

# Class names in the fallen dataset's yaml that mean "person is fallen".
# Everything else there is treated as a normal, uninjured person. EDIT THIS.
FALLEN_CLASS_NAMES = {"fallen", "fall", "lying", "laying", "fallen_person"}

IMG_SIZE = 640          # detector input size
PATCH = 224             # classifier input size
BATCH = 16
NEG_PER_BOX = 3         # no_wound patches cut from each normal person (only if the skin set is too small)
VAL_FRACTION = 0.10
SEED = 0

# wound training patches: square crop where the wound's longer side fills 40-83% of the patch,
# so the model sees wounds at the same "zoom" as the inference tiles (a wound inside a tile)
WOUND_FILL = (0.40, 0.83)
NEG_VAL_FRACTION = 0.20 # share of no_wound source images held out for validation (copies of one photo stay together)

# inference
WOUND_THRESH = 0.85     # raise to reduce false alarms, lower to catch more wounds
MIN_PATCH_PX = 24       # tiles smaller than this are skipped (person too far away to check)
TILE_FRACS = (0.15, 0.30)   # two tile scales, as a fraction of the person's longer box side
MAX_PATCHES = 160       # per person; if more, tiles are subsampled evenly (never cut off the legs)
# -----------------------------------------------------------------------------

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(SEED)


# ------------------------------ shared helpers -------------------------------
def load_names(ds: Path):
    cfg = ds / "data.yaml" if (ds / "data.yaml").exists() else ds / "dataset.yaml"
    names = yaml.safe_load(open(cfg))["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def find_split(ds: Path, aliases):
    for s in aliases:
        for img_dir, lbl_dir in ((ds / s / "images", ds / s / "labels"),
                                 (ds / "images" / s, ds / "labels" / s)):
            if img_dir.exists():
                return img_dir, lbl_dir
    # flat layout: ds/images + ds/labels with no split folders -> treat as train
    flat = ds / "images"
    if "train" in aliases and flat.exists() and any(p.suffix.lower() in IMG_EXT for p in flat.iterdir()):
        return flat, ds / "labels"
    return None


def collect(ds: Path, aliases):
    sp = find_split(ds, aliases)
    if sp is None:
        return []
    img_dir, lbl_dir = sp
    return [(p, lbl_dir / f"{p.stem}.txt") for p in sorted(img_dir.iterdir())
            if p.suffix.lower() in IMG_EXT]


def parse_label(line: str):
    """-> (cls, cx, cy, w, h) normalized. Accepts box, YOLO-pose or polygon labels."""
    p = line.split()
    cls = int(float(p[0]))
    v = list(map(float, p[1:]))
    if len(v) == 4:
        return cls, *v
    if len(v) == 4 + 17 * 3:        # YOLO pose: box + 17 keypoints (x y v) -> keep box only
        return cls, *v[:4]
    xs, ys = v[0::2], v[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def source_key(p: Path):
    """Roboflow names augmented copies '<name>.rf.<hash>.jpg'. Copies of one photo must stay in one split."""
    return p.name.split(".rf.")[0]


def split_train_val(ds: Path):
    train = collect(ds, ["train"])
    val = collect(ds, ["val", "valid", "validation"])
    if not val:
        groups = defaultdict(list)
        for it in train:
            groups[source_key(it[0])].append(it)
        keys = sorted(groups)
        random.shuffle(keys)
        k = max(1, int(len(keys) * VAL_FRACTION))
        val = [it for key in keys[:k] for it in groups[key]]
        train = [it for key in keys[k:] for it in groups[key]]
    return train, val


# ------------------------------ inspect --------------------------------------
SPLITS = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}


def inspect_fallen(ds: Path):
    names = load_names(ds)
    print("Classes:", dict(enumerate(names)), "\n")
    for split, aliases in SPLITS.items():
        found = find_split(ds, aliases)
        if not found:
            print(f"[{split}] not found\n")
            continue
        img_dir, lbl_dir = found
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        boxes, per_img, rel_h, missing, empty = Counter(), [], [], 0, 0
        for p in imgs:
            lbl = lbl_dir / f"{p.stem}.txt"
            if not lbl.exists():
                missing += 1
                continue
            rows = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
            if not rows:
                empty += 1
            per_img.append(len(rows))
            for r in rows:
                boxes[names[int(float(r[0]))]] += 1
                if len(r) in (5, 56):
                    rel_h.append(float(r[4]))
        sizes = [Image.open(p).size for p in imgs[:50]]

        total = sum(boxes.values()) or 1
        print(f"[{split}] {len(imgs)} images | {missing} missing label files | {empty} empty labels")
        for n, c in boxes.most_common():
            print(f"    {n:<15} {c:>6} boxes ({100 * c / total:.1f}%)")
        if per_img:
            print(f"    people per image: avg {sum(per_img) / len(per_img):.1f}, max {max(per_img)}")
        if rel_h:
            small = sum(h < 0.1 for h in rel_h)
            print(f"    boxes shorter than 10% of image height: {100 * small / len(rel_h):.0f}%")
        if sizes:
            print(f"    sample image sizes: {Counter(sizes).most_common(3)}")
        print()


def _summarize_wound(split, counts, sizes, cover):
    total = sum(counts.values())
    print(f"[{split}] {total} wound samples")
    for c, n in counts.most_common():
        print(f"    {c:<20} {n:>6} ({100 * n / max(total, 1):.1f}%)")
    if sizes:
        ws, hs = zip(*sizes)
        print(f"    image size: min {min(ws)}x{min(hs)}, max {max(ws)}x{max(hs)}")
        small = sum(min(w, h) < 64 for w, h in sizes)
        print(f"    images with short side < 64px: {100 * small / len(sizes):.0f}%")
    if cover:
        print(f"    avg wound box covers {100 * sum(cover) / len(cover):.0f}% of image "
              f"(near 100% = whole image is the patch)")
    print()


def inspect_wound(ds: Path):
    seen = defaultdict(set)  # image hash -> splits it appears in (detects train/val leakage)
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()

    if (ds / "data.yaml").exists():
        print("Format: DETECTION (data.yaml found)")
        names = load_names(ds)
        print("Classes:", dict(enumerate(names)), "\n")
        for split, aliases in SPLITS.items():
            found = find_split(ds, aliases)
            if not found:
                print(f"[{split}] not found\n")
                continue
            img_dir, lbl_dir = found
            counts, sizes, cover, no_label = Counter(), [], [], 0
            for p in img_dir.iterdir():
                if p.suffix.lower() not in IMG_EXT:
                    continue
                seen[md5(p)].add(split)
                lbl = lbl_dir / f"{p.stem}.txt"
                rows = [r.split() for r in lbl.read_text().splitlines() if r.strip()] if lbl.exists() else []
                if not rows:
                    no_label += 1
                    continue
                sizes.append(Image.open(p).size)
                for r in rows:
                    ci = int(float(r[0]))
                    counts[names[ci] if ci < len(names) else f"UNKNOWN_class_{ci}"] += 1   # flags bad ids, no crash
                    if len(r) >= 5:
                        cover.append(min(1.0, float(r[3]) * float(r[4])))
            if no_label:
                print(f"    ({split}: {no_label} images have no labels and will be skipped)")
            _summarize_wound(split, counts, sizes, cover)
    else:
        print("Format: FOLDER (no data.yaml; expecting <split>/<class>/*.jpg)\n")
        for split, aliases in SPLITS.items():
            counts, sizes = Counter(), []
            for a in aliases:
                for cdir in (ds / a).glob("*") if (ds / a).exists() else []:
                    if cdir.is_dir():
                        for p in cdir.iterdir():
                            if p.suffix.lower() in IMG_EXT:
                                counts[cdir.name] += 1
                                seen[md5(p)].add(split)
                                if len(sizes) < 300:
                                    sizes.append(Image.open(p).size)
            if counts:
                _summarize_wound(split, counts, sizes, [])
            else:
                print(f"[{split}] not found\n")

    leaks = sum(len(s) > 1 for s in seen.values())
    print("Leakage check:", f"{leaks} identical images appear in more than one split (train/val leakage!)"
          if leaks else "no identical images across splits")


# ------------------------------ training -------------------------------------
def degrade(im: Image.Image) -> Image.Image:
    """Wound patches are sharp close-ups; in a crash scene they are tiny and blurry.
    Randomly degrade ALL classes so resolution isn't a shortcut for the model."""
    if random.random() < 0.5:
        s = random.choice([48, 64, 96, 128])
        im = im.resize((s, s)).resize((PATCH, PATCH))
    if random.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
    return im


def two_stage(model_path, data, task_args, tag, epochs1, epochs2, freeze, lr1, lr2):
    from ultralytics import YOLO
    common = dict(data=data, batch=BATCH, seed=SEED, project="runs", exist_ok=True,
                  optimizer="AdamW", **task_args)   # explicit, otherwise Ultralytics ignores lr0
    m = YOLO(model_path)
    m.train(epochs=epochs1, freeze=freeze, lr0=lr1, patience=15, name=f"{tag}_stage1", **common)
    m = YOLO(m.trainer.best)
    m.train(epochs=epochs2, freeze=0, lr0=lr2, lrf=0.01, cos_lr=True, patience=25,
            name=f"{tag}_stage2", **common)
    print(f"\n[{tag}] best weights: {m.trainer.best}")
    dest = Path("weights") / f"{tag}.pt"       # fixed location that `infer` uses by default
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(m.trainer.best, dest)
    print(f"[{tag}] copied to {dest}")
    return m.trainer.best


def prepare_state():
    out = WORK / "state"
    if out.exists():
        shutil.rmtree(out)
    names = load_names(FALLEN_DS)
    id_map = {i: (1 if n.lower() in FALLEN_CLASS_NAMES else 0) for i, n in enumerate(names)}
    print("fallen dataset class mapping:", {names[i]: ["normal", "fallen"][v] for i, v in id_map.items()})

    train, val = split_train_val(FALLEN_DS)
    for split, items in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for img, lbl in items:
            shutil.copy2(img, out / "images" / split / img.name)
            lines = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    if line.split():
                        c, *box = parse_label(line)
                        lines.append(" ".join([str(id_map[c])] + [f"{x:.6f}" for x in box]))
            (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
    yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                    "names": {0: "normal", 1: "fallen"}}, open(out / "data.yaml", "w"))
    return out / "data.yaml"


def train_state():
    data = prepare_state()
    aug = dict(imgsz=IMG_SIZE, hsv_h=0.015, fliplr=0.5, mosaic=1.0, close_mosaic=10)
    # freeze=10 -> backbone frozen in stage 1
    two_stage("yolov8m.pt", str(data), aug, "state", 30, 80, 10, 0.001, 0.0002)


def load_wound_items():
    """WOUND_DS -> {"train": [(img, lbl|None)], "val": [...]}. Any class counts as "wound".
    Byte-identical images are kept once (in val if they are in both)."""
    ds = WOUND_DS
    if not ds.exists():
        raise SystemExit(f"{ds} not found. Check WOUND_DS.")
    if (ds / "data.yaml").exists() or (ds / "dataset.yaml").exists():      # detection format
        train, val = split_train_val(ds)
        items = {"train": train, "val": val}
    else:                                              # folder format: <split>/<class>/*.jpg, whole image = patch
        items = {"train": [], "val": []}
        for split, aliases in (("train", ["train"]), ("val", ["val", "valid", "validation"])):
            for a in aliases:
                for cdir in ((ds / a).glob("*") if (ds / a).exists() else []):
                    if cdir.is_dir():
                        items[split] += [(p, None) for p in cdir.iterdir() if p.suffix.lower() in IMG_EXT]
    print(f"{ds}: {len(items['train']) + len(items['val'])} images")

    owner, dropped = set(), 0
    for split in ("val", "train"):                     # val first -> a duplicate never stays in train
        keep = []
        for img, lbl in items[split]:
            h = hashlib.md5(img.read_bytes()).hexdigest()
            if h in owner:
                dropped += 1
                continue
            owner.add(h)
            keep.append((img, lbl))
        items[split] = keep
    if dropped:
        print(f"dropped {dropped} byte-identical duplicate images (across splits/datasets)")
    return items


def context_crop(im: Image.Image, box, fill):
    """Square crop that contains `box` (pixels x1,y1,x2,y2) with box's longer side = `fill` x crop side.
    Random position, but the wound stays inside the crop. Square -> no squashing at resize time."""
    W, H = im.size
    bx1, by1, bx2, by2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    side = min(max(bx2 - bx1, by2 - by1) / fill, W, H)

    def start(lo_box, hi_box, limit):
        lo, hi = max(0.0, hi_box - side), min(lo_box, limit - side)
        if lo <= hi:
            return random.uniform(lo, hi)
        return min(max(0.0, (lo_box + hi_box) / 2 - side / 2), limit - side)   # wound bigger than crop: centre it

    x, y, side = int(start(bx1, bx2, W)), int(start(by1, by2, H)), int(side)
    return im.crop((x, y, x + side, y + side))      # integer edges -> exactly square


def prepare_wound():
    out = WORK / "wound_cls"
    if out.exists():
        shutil.rmtree(out)
    counts = {}

    def save(im, split, cls):
        d = out / split / cls
        d.mkdir(parents=True, exist_ok=True)
        counts[(split, cls)] = counts.get((split, cls), 0) + 1
        degrade(im.convert("RGB").resize((PATCH, PATCH))).save(d / f"{counts[(split, cls)]:06d}.jpg", quality=92)

    # ---- wound patches: every box of every dataset -> class "wound" ----
    items = load_wound_items()
    for split in ("train", "val"):
        for img, lbl in items[split]:
            try:
                im = Image.open(img).convert("RGB")
            except Exception as e:
                print(f"skipped unreadable image {img.name}: {e}")
                continue
            W, H = im.size
            if lbl is None:                                    # folder format: the whole image is the wound patch
                boxes = [(0, 0, W, H)]
            elif lbl.exists():
                boxes = []
                for line in lbl.read_text().splitlines():
                    if line.split():
                        _, cx, cy, w, h = parse_label(line)
                        boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
            else:
                boxes = []
            for box in boxes:
                if max(box[2] - box[0], box[3] - box[1]) > 8:
                    save(context_crop(im, box, random.uniform(*WOUND_FILL)), split, "wound")

    n_w = {sp: counts.get((sp, "wound"), 0) for sp in ("train", "val")}
    print("wound patches:", n_w)
    if not n_w["train"]:
        raise SystemExit("No wound patches found. Check WOUND_DS and that the dataset has label files.")

    # ---- no_wound patches: every healthy-skin image once, as it is (centred square cut so it isn't squashed) ----
    files = sorted(p for p in NORMAL_SKIN_DS.rglob("*") if p.suffix.lower() in IMG_EXT) if NORMAL_SKIN_DS.exists() else []
    if not files:
        print(f"WARNING: no images in {NORMAL_SKIN_DS} -> no healthy-skin negatives")
    seen, groups = set(), defaultdict(list)
    for p in files:                                        # drop byte-identical copies ("... Copy.jpg" etc.)
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            groups[source_key(p)].append(p)                # Roboflow-style copies of one photo share a group
    if len(seen) < len(files):
        print(f"dropped {len(files) - len(seen)} byte-identical duplicate skin images")
    keys = sorted(groups)
    random.shuffle(keys)
    n_val = max(1, int(len(keys) * NEG_VAL_FRACTION)) if len(keys) > 1 else 0
    for split, ks in (("val", keys[:n_val]), ("train", keys[n_val:])):
        for k in ks:
            for p in groups[k]:
                try:
                    im = Image.open(p).convert("RGB")
                except Exception as e:
                    print(f"skipped unreadable image {p.name}: {e}")
                    continue
                side = min(im.size)
                if side >= 8:
                    x, y = (im.size[0] - side) // 2, (im.size[1] - side) // 2
                    save(im.crop((x, y, x + side, y + side)), split, "no_wound")
        n = counts.get((split, "no_wound"), 0)
        print(f"no_wound [{split}]: {n} images (wound: {n_w[split]})")
        if n < 0.9 * n_w[split]:
            print(f"  note: fewer no_wound than wound patches ({n} vs {n_w[split]}), expect more false alarms; "
                  "raise WOUND_THRESH if so, or add more healthy images.")

    cap = max(0, n_w["train"] - counts.get(("train", "no_wound"), 0))  # gap still to fill from person crops
    if cap == 0 or not FALLEN_DS.exists():
        return out

    # ---- top up no_wound with patches cut from NORMAL persons in the fallen dataset ----
    names = load_names(FALLEN_DS)
    normal_ids = {i for i, n in enumerate(names) if n.lower() not in FALLEN_CLASS_NAMES}
    train, val = split_train_val(FALLEN_DS)
    cands = []
    for split, its in (("train", train), ("val", val)):
        for img, lbl in its:
            if not lbl.exists():
                continue
            for line in lbl.read_text().splitlines():
                if line.split():
                    c, cx, cy, w, h = parse_label(line)
                    if c in normal_ids:
                        cands.append((split, img, cx, cy, w, h))
    random.shuffle(cands)
    made = 0
    for split, img, cx, cy, w, h in cands:
        if made >= cap:
            break
        im = Image.open(img).convert("RGB")
        W, H = im.size
        bx1, by1, bw, bh = (cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H
        for _ in range(NEG_PER_BOX):
            side = min(random.uniform(0.25, 0.5) * max(bw, bh), bw, bh)
            if side < 24:
                continue
            x = bx1 + random.uniform(0, bw - side)
            y = by1 + random.uniform(0, bh - side)
            save(im.crop((int(x), int(y), int(x + side), int(y + side))), split, "no_wound")
            made += 1
    print(f"no_wound patches added from normal persons: {made}")
    return out


def train_wound():
    data = prepare_wound()
    # scale=0.2: Ultralytics' random-resized-crop keeps 80-100% of the patch (default 50%),
    # so augmentation can't crop the wound out of a patch that is still labelled 'wound'
    aug = dict(imgsz=PATCH, hsv_h=0.01, scale=0.2)
    # classification model: layers 0-8 = backbone, layer 9 = classifier head -> freeze=9
    two_stage("yolov8m-cls.pt", str(data), aug, "wound", 15, 40, 9, 0.001, 0.0002)


# ------------------------------ inference ------------------------------------
def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    i = max(0.0, w) * max(0.0, h)
    return i / (area(a) + area(b) - i + 1e-9)


def _starts(n, side, stride):
    """Window start positions along one axis, always including the last edge."""
    st = list(range(0, max(1, n - side + 1), stride))
    if st[-1] + side < n:
        st.append(n - side)
    return st


def tiles(img, box):
    """Square windows (50% overlap) over the person's box at each scale in TILE_FRACS.
    -> [(patch, (x1, y1, x2, y2) in image pixels)]"""
    H, W = img.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(W, int(box[2])), min(H, int(box[3]))
    h, w = y2 - y1, x2 - x1
    out, sides = [], set()
    for f in TILE_FRACS:
        side = int(min(f * max(h, w), min(h, w)))
        if side < MIN_PATCH_PX or side in sides:
            continue
        sides.add(side)
        stride = max(1, side // 2)
        for y in _starts(h, side, stride):
            for x in _starts(w, side, stride):
                out.append((img[y1 + y:y1 + y + side, x1 + x:x1 + x + side],
                            (x1 + x, y1 + y, x1 + x + side, y1 + y + side)))
    if len(out) > MAX_PATCHES:      # subsample evenly so the whole body stays covered
        out = [out[i * len(out) // MAX_PATCHES] for i in range(MAX_PATCHES)]
    return out


def wound_score(wound_model, img, box):
    """-> (highest P(wound) over the person's body tiles, the tile it came from).
    (0.0, None) if the person is too small to check."""
    items = tiles(img, box)
    if not items:
        return 0.0, None
    no_idx = [i for i, n in wound_model.names.items() if n == "no_wound"][0]
    best, best_box = 0.0, None
    for i in range(0, len(items), 64):                 # chunks keep memory small
        chunk = items[i:i + 64]
        for r, (_, tb) in zip(wound_model([p for p, _ in chunk], imgsz=PATCH, verbose=False), chunk):
            sc = 1.0 - float(r.probs.data[no_idx])
            if sc > best:
                best, best_box = sc, tb
    return best, best_box


def analyze(image_path, state_weights, wound_weights, save_path=None,
            person_weights="yolov8m.pt", person_conf=0.35, state_conf=0.30):
    import cv2
    from ultralytics import YOLO

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    person_model = YOLO(person_weights)       # general COCO person finder
    state_model = YOLO(state_weights) if state_weights else None   # normal vs fallen (optional)
    wound_model = YOLO(wound_weights)         # wound patch classifier

    persons = [{"box": b, "fallen": False}
               for b in person_model(img, classes=[0], conf=person_conf, verbose=False)[0].boxes.xyxy.tolist()]

    if state_model:
        det = state_model(img, conf=state_conf, verbose=False)[0]
        for box, cls in zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist()):
            is_fallen = state_model.names[int(cls)] == "fallen"
            match = next((p for p in persons if iou(p["box"], box) >= 0.4), None)
            if match:
                match["fallen"] |= is_fallen
            else:                              # person the COCO model missed
                persons.append({"box": box, "fallen": is_fallen})

    for p in persons:
        p["score"], p["where"] = wound_score(wound_model, img, p["box"])
        p["wound"] = p["score"] >= WOUND_THRESH
        p["injured"] = p["fallen"] or p["wound"]

    if save_path:
        for p in persons:
            color = (0, 0, 255) if p["injured"] else (0, 200, 0)   # red / green
            x1, y1, x2, y2 = map(int, p["box"])
            tag = "OK" if not p["injured"] else (("FALLEN" if p["fallen"] else "") +
                                                 ("+WOUND" if p["wound"] and p["fallen"] else
                                                  "WOUND" if p["wound"] else ""))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if p["wound"] and p["where"]:                   # thin box on the tile that triggered the alarm
                wx1, wy1, wx2, wy2 = p["where"]
                cv2.rectangle(img, (wx1, wy1), (wx2, wy2), (0, 255, 255), 1)
            cv2.putText(img, f"{tag} {p['score']:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(save_path, img)

    injured = sum(p["injured"] for p in persons)
    return {
        "people_detected": len(persons),
        "injured_people_detected": injured,
        "uninjured_people_detected": len(persons) - injured,
        "people": [{"box": [round(v) for v in p["box"]],
                    "status": "injured" if p["injured"] else "uninjured",
                    "score": round(p["score"], 3),      # P(wound), 0-1
                    "fallen": p["fallen"],
                    "wound_tile": list(p["where"]) if p["wound"] and p["where"] else None}
                   for p in persons],
    }


# ------------------------------ CLI ------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Rescue AI: injured vs uninjured people counter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect-fallen").add_argument("path", nargs="?", type=Path, default=FALLEN_DS)
    sub.add_parser("inspect-wound").add_argument("path", nargs="?", type=Path, default=WOUND_DS,
                                                 help="default: WOUND_DS")
    sub.add_parser("train-state")
    sub.add_parser("train-wound")
    p = sub.add_parser("infer")
    p.add_argument("image")
    p.add_argument("--state-weights", default=None, help="optional fallen-person model (not trained yet)")
    p.add_argument("--wound-weights", default="weights/wound.pt")
    p.add_argument("--save", default=None, help="path to save annotated image")
    a = ap.parse_args()

    if a.cmd == "inspect-fallen":
        inspect_fallen(a.path)
    elif a.cmd == "inspect-wound":
        inspect_wound(a.path)
    elif a.cmd == "train-state":
        train_state()
    elif a.cmd == "train-wound":
        train_wound()
    else:
        print(json.dumps(analyze(a.image, a.state_weights, a.wound_weights, a.save), indent=2))


if __name__ == "__main__":
    main()