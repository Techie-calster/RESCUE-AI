"""
RESCUE-AI - Wound Detection Model
=================================

Original baseline model only.
No normal-skin / negative images are included.

Environment variables:

    DATASET_DIR=/path/to/dataset
        Dataset root containing train/, valid/, test/

    RUNS_DIR=/path/to/runs
        Where training output and checkpoints are saved.

    WEIGHTS=/path/to/best.pt
        Use a specific model checkpoint for report/predict.

Commands:

    python wound_detection.py check
        Inspect dataset structure, image counts and labels.

    python wound_detection.py train [epochs] [imgsz]
        Train the original baseline model.
        Defaults: 60 epochs, image size 640.

    python wound_detection.py resume
        Resume an interrupted training run using last.pt.

    python wound_detection.py report
        Evaluate the model on the held-out test dataset.

    python wound_detection.py predict <image or folder>
        Run wound detection on an image or folder.

Examples:

    python wound_detection.py check

    python wound_detection.py train

    python wound_detection.py train 60 640

    python wound_detection.py resume

    WEIGHTS=/path/to/best.pt python wound_detection.py report

    WEIGHTS=/path/to/best.pt python wound_detection.py predict ./Testing

    WEIGHTS=/path/to/best.pt python wound_detection.py predict ./Testing/image.jpg

    WEIGHTS=/path/to/best.pt python wound_detection.py predict ./Testing --conf 0.15

Model:
    Architecture : YOLO11n
    Classes      : 1 (wound)

Baseline training configuration:
    Epochs       : 60
    Image size   : 640
    Batch size   : 16
    Patience     : 15
    Seed         : 42

Prediction:
    Image size   : 640
    Confidence   : 0.25 (default notebook prediction threshold)

"""

import os
import sys
import yaml
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DATASET_DIR = Path(
    os.environ.get(
        "DATASET_DIR",
        "./dataset/segmentation.v1i.yolov11"
    )
).resolve()

RUNS_DIR = Path(
    os.environ.get("RUNS_DIR", "./runs")
).resolve()

RUN_NAME = "wound_detection_baseline"

YAML_FIXED = Path("./wound_data_fixed.yaml").resolve()

DEFAULT_WEIGHTS = (
    RUNS_DIR
    / RUN_NAME
    / "weights"
    / "best.pt"
)

IMG_EXT = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}

BASE_MODEL = "yolo11n.pt"

DEFAULT_EPOCHS = 60
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_PATIENCE = 15

PREDICT_CONF = 0.15
PREDICT_IMGSZ = 640

PREDICTION_DIR = RUNS_DIR / "wound_predictions"


# ============================================================
# DEVICE SELECTION
# ============================================================

def dev():
    """
    CUDA GPU > Apple MPS > CPU.
    """

    if torch.cuda.is_available():
        return 0

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


# ============================================================
# DATASET UTILITIES
# ============================================================

def images_in(folder):

    folder = Path(folder)

    if not folder.exists():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )


def find_split(split):

    path = DATASET_DIR / split

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}"
        )

    return path


def find_weights():

    """
    WEIGHTS environment variable takes priority.
    Otherwise use the baseline best.pt checkpoint.
    """

    if os.environ.get("WEIGHTS"):

        weights = Path(os.environ["WEIGHTS"]).expanduser()

        if not weights.is_file():
            raise FileNotFoundError(
                f"WEIGHTS file not found: {weights}"
            )

        return weights.resolve()

    if DEFAULT_WEIGHTS.exists():
        return DEFAULT_WEIGHTS

    raise FileNotFoundError(
        "Baseline weights not found.\n"
        f"Expected: {DEFAULT_WEIGHTS}\n"
        "Train the model first or set WEIGHTS=/path/to/best.pt"
    )


# ============================================================
# 1. CHECK DATASET
# ============================================================

def check():

    print("\n========== WOUND DATASET CHECK ==========\n")

    print("Dataset:", DATASET_DIR)
    print("Runs directory:", RUNS_DIR)
    print("Device:", dev())

    if not DATASET_DIR.exists():

        print("\nERROR: Dataset directory not found.")
        return

    total_images = 0
    total_labels = 0
    total_boxes = 0

    for split in ["train", "valid", "test"]:

        split_dir = DATASET_DIR / split

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        print(f"\n--- {split.upper()} ---")

        if not images_dir.exists():
            print("Missing images directory:", images_dir)
            continue

        if not labels_dir.exists():
            print("Missing labels directory:", labels_dir)
            continue

        images = images_in(images_dir)

        labels = list(labels_dir.glob("*.txt"))

        print("Images:", len(images))
        print("Label files:", len(labels))

        total_images += len(images)
        total_labels += len(labels)

        valid_boxes = 0
        invalid_lines = 0

        for label_file in labels:

            for line_no, line in enumerate(
                label_file.read_text().splitlines(),
                start=1
            ):

                parts = line.split()

                if not parts:
                    continue

                try:
                    values = [float(x) for x in parts]

                except ValueError:
                    invalid_lines += 1
                    print(
                        "Invalid number:",
                        label_file,
                        "line:",
                        line_no
                    )
                    continue

                if len(values) != 5:

                    invalid_lines += 1
                    print(
                        "Invalid YOLO format:",
                        label_file,
                        "line:",
                        line_no
                    )
                    continue

                cls, x, y, w, h = values

                if (
                    cls != 0
                    or not all(0 <= v <= 1 for v in [x, y, w, h])
                    or w <= 0
                    or h <= 0
                ):

                    invalid_lines += 1
                    print(
                        "Invalid YOLO values:",
                        label_file,
                        "line:",
                        line_no
                    )
                    continue

                valid_boxes += 1

        total_boxes += valid_boxes
        total_labels += 0

        print("Valid wound boxes:", valid_boxes)
        print("Invalid annotations:", invalid_lines)

    print("\n========== SUMMARY ==========")

    print("Total images:", total_images)
    print("Total label files:", total_labels)
    print("Total valid wound boxes:", total_boxes)

    print("\nDataset YAML:", YAML_FIXED)

    print("\n========================================")


# ============================================================
# 2. CREATE DATASET YAML
# ============================================================

def write_yaml():

    required = [
        DATASET_DIR / "train/images",
        DATASET_DIR / "train/labels",
        DATASET_DIR / "valid/images",
        DATASET_DIR / "valid/labels",
        DATASET_DIR / "test/images",
        DATASET_DIR / "test/labels",
    ]

    for folder in required:

        if not folder.exists():

            raise FileNotFoundError(
                f"Required dataset folder missing: {folder}"
            )

    data = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["wound"],
    }

    with open(YAML_FIXED, "w") as f:

        yaml.safe_dump(
            data,
            f,
            sort_keys=False
        )

    print("\nDataset YAML created:", YAML_FIXED)

    return YAML_FIXED


# ============================================================
# 3. TRAIN BASELINE MODEL
# ============================================================

def train(epochs=60, imgsz=640):

    print("\n========== BASELINE TRAINING ==========\n")

    print("Model:", BASE_MODEL)
    print("Dataset:", DATASET_DIR)
    print("Epochs:", epochs)
    print("Image size:", imgsz)
    print("Device:", dev())

    data = write_yaml()

    gpu = dev() == 0

    batch = 16 if gpu else 8

    model = YOLO(BASE_MODEL)

    model.train(

        data=str(data),

        epochs=epochs,

        imgsz=imgsz,

        batch=batch,

        patience=DEFAULT_PATIENCE,

        device=dev(),

        workers=4 if gpu else 2,

        project=str(RUNS_DIR),

        name=RUN_NAME,

        exist_ok=True,

        plots=True,

        seed=42

    )

    print("\n========== TRAINING COMPLETED ==========")

    print(
        "Best weights:",
        RUNS_DIR / RUN_NAME / "weights/best.pt"
    )

    print(
        "Last weights:",
        RUNS_DIR / RUN_NAME / "weights/last.pt"
    )


# ============================================================
# 4. RESUME INTERRUPTED TRAINING
# ============================================================

def resume():

    last_weights = (
        RUNS_DIR
        / RUN_NAME
        / "weights"
        / "last.pt"
    )

    if not last_weights.exists():

        raise FileNotFoundError(
            f"last.pt not found: {last_weights}"
        )

    print("\nResuming training from:", last_weights)

    model = YOLO(str(last_weights))

    model.train(resume=True)

    print("\nTraining resumed/completed.")


# ============================================================
# 5. TEST DATASET REPORT
# ============================================================

def report():

    weights = find_weights()

    data = write_yaml()

    print("\n========== WOUND MODEL REPORT ==========")

    print("Weights:", weights)

    model = YOLO(str(weights))

    metrics = model.val(

        data=str(data),

        split="test",

        imgsz=640,

        batch=16,

        plots=True,

        project=str(RUNS_DIR),

        name="wound_test_evaluation",

        exist_ok=True

    )

    print("\n========== TEST RESULTS ==========")

    print("Precision:", metrics.box.mp)

    print("Recall:", metrics.box.mr)

    print("mAP50:", metrics.box.map50)

    print("mAP50-95:", metrics.box.map)

    print("\nReport saved under:", RUNS_DIR)


# ============================================================
# 6. WOUND PREDICTION
# ============================================================

def predict(source, conf=PREDICT_CONF):

    weights = find_weights()

    source = Path(source).expanduser().resolve()

    if not source.exists():

        raise FileNotFoundError(
            f"Prediction source not found: {source}"
        )

    if source.is_file() and source.suffix.lower() not in IMG_EXT:

        raise ValueError(
            f"Unsupported image format: {source.suffix}"
        )

    if source.is_dir() and not images_in(source):

        raise FileNotFoundError(
            f"No images found in: {source}"
        )

    print("\n========== WOUND PREDICTION ==========")

    print("Weights:", weights)
    print("Source:", source)
    print("Confidence:", conf)
    print("Image size:", PREDICT_IMGSZ)

    model = YOLO(str(weights))

    results = model.predict(

        source=str(source),

        imgsz=PREDICT_IMGSZ,

        conf=conf,

        save=True,

        project=str(PREDICTION_DIR),

        name="predictions",

        exist_ok=True,

        verbose=True

    )

    print("\n========== PREDICTION COMPLETED ==========")

    print("Images processed:", len(results))

    print(
        "Predictions saved:",
        PREDICTION_DIR / "predictions"
    )


# ============================================================
# 7. MAIN COMMAND DISPATCHER
# ============================================================

if __name__ == "__main__":

    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"

    if cmd == "check":

        check()

    elif cmd == "train":

        train(
            int(sys.argv[2]) if len(sys.argv) > 2 else 60,
            int(sys.argv[3]) if len(sys.argv) > 3 else 640
        )

    elif cmd == "resume":

        resume()

    elif cmd == "report":

        report()

    elif cmd == "predict":

        if len(sys.argv) < 3:

            print(
                "Usage: python wound_detection.py predict <image or folder>"
            )

        else:

            predict(" ".join(sys.argv[2:]))

    else:

        print(__doc__)