"""
RESCUE-AI — Final Vision Pipeline

Commands:
    python vision.py check
    python vision.py predict ./Testing/accident.jpg

Project structure:
    RESCUE-AI/
    ├── vision.py
    ├── models/
    │   ├── accident_best.pt
    │   ├── wound_best.pt
    │   └── yolo26n.pt
    ├── Testing/
    └── output/

Install:
    pip install ultralytics opencv-python torch
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Model weights
ACCIDENT_WEIGHTS = BASE_DIR / "models" / "accident_best.pt"
PERSON_WEIGHTS = BASE_DIR / "models" / "yolo26n.pt"
WOUND_WEIGHTS = BASE_DIR / "models" / "wound_best.pt"

# Output directory
OUTPUT_DIR = BASE_DIR / "output"

# Accident detection
ACCIDENT_CONF = 0.30

# Person detection
PERSON_CONF = 0.20
PERSON_IMGSZ = 960

# Wound detection
WOUND_CONF = 0.15
WOUND_IMGSZ = 640

# Person crop settings
CROP_PADDING = 0.08
BLUR_THRESHOLD = 80.0

# Display-only wound score adjustment
UPLIFT_FACTOR = 1.25
MIN_DISPLAY_SCORE = 0.50
MAX_DISPLAY_SCORE = 1.00

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ============================================================
# DEVICE SELECTION
# ============================================================

def select_device():

    if torch.cuda.is_available():
        return 0

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


DEVICE = select_device()


# ============================================================
# MODEL CHECK
# ============================================================

def check_weight_files():

    weights = {
        "Accident model": ACCIDENT_WEIGHTS,
        "Person model": PERSON_WEIGHTS,
        "Wound model": WOUND_WEIGHTS
    }

    print("\nRESCUE-AI MODEL CHECK")
    print("-" * 55)

    all_found = True

    for name, path in weights.items():

        found = path.is_file()

        print(
            f"{name:18s}: "
            f"{'FOUND' if found else 'MISSING'}"
        )

        print(f"  {path}")

        if not found:
            all_found = False

    print(f"\nDevice: {DEVICE}")

    return all_found


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    if not check_weight_files():

        raise FileNotFoundError(
            "\nOne or more model files are missing. "
            "Place the weights inside the models/ folder."
        )

    print("\nLoading models...")

    models = {

        "accident": YOLO(
            str(ACCIDENT_WEIGHTS)
        ),

        "person": YOLO(
            str(PERSON_WEIGHTS)
        ),

        "wound": YOLO(
            str(WOUND_WEIGHTS)
        )
    }

    print("All models loaded successfully.")
    print("Device:", DEVICE)

    return models


# ============================================================
# GENERAL HELPERS
# ============================================================

def adjust_display_score(original_confidence):

    if original_confidence < MIN_DISPLAY_SCORE:

        adjusted = max(
            original_confidence * UPLIFT_FACTOR,
            MIN_DISPLAY_SCORE
        )

    else:

        adjusted = original_confidence

    return min(adjusted, MAX_DISPLAY_SCORE)


def get_image_files(source):

    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {source}"
        )

    if source.is_file():

        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {source.suffix}"
            )

        return [source]

    images = sorted(
        p for p in source.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    if not images:

        raise ValueError(
            f"No supported images found in: {source}"
        )

    return images


def create_output_folders(image_path):

    image_output_dir = OUTPUT_DIR / image_path.stem

    json_dir = image_output_dir / "json"
    crop_dir = image_output_dir / "person_crops"
    annotated_dir = image_output_dir / "annotated"

    for folder in (
        json_dir,
        crop_dir,
        annotated_dir
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    return (
        image_output_dir,
        json_dir,
        crop_dir,
        annotated_dir
    )


def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def get_class_name(model, class_id):

    names = model.names

    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


# ============================================================
# STAGE 1 — ACCIDENT DETECTION
# ============================================================

def detect_accident(image_path, accident_model):

    result = accident_model.predict(

        source=str(image_path),

        conf=ACCIDENT_CONF,

        device=DEVICE,

        verbose=False

    )[0]

    accident_boxes = []
    accident_scores = []

    if result.boxes is not None:

        for index, box in enumerate(result.boxes):

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            confidence = float(
                box.conf[0].item()
            )

            class_id = int(
                box.cls[0].item()
            )

            accident_scores.append(confidence)

            accident_boxes.append({

                "box_id": index + 1,

                "class_id": class_id,

                "class_name": get_class_name(
                    accident_model,
                    class_id
                ),

                "confidence": round(
                    confidence,
                    4
                ),

                "bbox_xyxy": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]

            })

    return {

        "detected": len(accident_boxes) > 0,

        "confidence": round(
            max(accident_scores),
            4
        ) if accident_scores else 0.0,

        "threshold": ACCIDENT_CONF,

        "bounding_boxes": accident_boxes
    }


# ============================================================
# STAGE 2 — PERSON DETECTION AND CROPPING
# ============================================================

def detect_people(
    original,
    image_path,
    person_model,
    crop_dir
):

    height, width = original.shape[:2]

    result = person_model.predict(

        source=str(image_path),

        classes=[0],

        conf=PERSON_CONF,

        imgsz=PERSON_IMGSZ,

        device=DEVICE,

        verbose=False

    )[0]

    people = []

    if result.boxes is None:
        return people

    for box in result.boxes:

        raw_x1, raw_y1, raw_x2, raw_y2 = (
            box.xyxy[0].tolist()
        )

        confidence = float(
            box.conf[0].item()
        )

        x1 = max(0, int(raw_x1))
        y1 = max(0, int(raw_y1))

        x2 = min(width, int(raw_x2))
        y2 = min(height, int(raw_y2))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            continue

        pad_x = int(
            box_width * CROP_PADDING
        )

        pad_y = int(
            box_height * CROP_PADDING
        )

        crop_x1 = max(
            0,
            x1 - pad_x
        )

        crop_y1 = max(
            0,
            y1 - pad_y
        )

        crop_x2 = min(
            width,
            x2 + pad_x
        )

        crop_y2 = min(
            height,
            y2 + pad_y
        )

        person_crop = original[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        if person_crop.size == 0:
            continue

        person_id = len(people) + 1

        crop_path = (
            crop_dir
            / f"person_{person_id:03d}.jpg"
        )

        if not cv2.imwrite(
            str(crop_path),
            person_crop
        ):

            print(
                f"Warning: could not save crop "
                f"for person {person_id}"
            )

            continue

        gray = cv2.cvtColor(
            person_crop,
            cv2.COLOR_BGR2GRAY
        )

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        is_blurry = (
            sharpness < BLUR_THRESHOLD
        )

        people.append({

            "person_id": person_id,

            "confidence": round(
                confidence,
                4
            ),

            "bbox_original_xyxy": [
                x1,
                y1,
                x2,
                y2
            ],

            "crop_region_original_xyxy": [
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2
            ],

            "crop_path": str(crop_path),

            "image_quality": {

                "sharpness_score": round(
                    sharpness,
                    2
                ),

                "is_blurry": is_blurry,

                "wound_eligible": not is_blurry
            }

        })

    return people


# ============================================================
# STAGE 3 — WOUND DETECTION
# ============================================================

def detect_wounds(people, wound_model):

    wound_person_results = []

    total_wounds = 0
    injured_people = 0
    people_evaluated = 0
    people_skipped = 0

    all_adjusted_scores = []

    for person in people:

        person_id = person["person_id"]

        print(
            f"\nProcessing Person {person_id}"
        )

        if not person["image_quality"]["wound_eligible"]:

            print("Skipped: blurry crop")

            wound_person_results.append({

                "person_id": person_id,

                "evaluation_status": "skipped_blurry",

                "wound_detected": None,

                "wound_count": 0,

                "wounds": []

            })

            people_skipped += 1

            continue

        wound_result = wound_model.predict(

            source=person["crop_path"],

            conf=WOUND_CONF,

            imgsz=WOUND_IMGSZ,

            device=DEVICE,

            verbose=False

        )[0]

        people_evaluated += 1

        offset_x, offset_y = (
            person["crop_region_original_xyxy"][:2]
        )

        wounds = []

        if wound_result.boxes is not None:

            for index, box in enumerate(
                wound_result.boxes
            ):

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                original_confidence = float(
                    box.conf[0].item()
                )

                adjusted_confidence = (
                    adjust_display_score(
                        original_confidence
                    )
                )

                class_id = int(
                    box.cls[0].item()
                )

                # Convert crop coordinates
                # back to original image coordinates.

                ox1 = x1 + offset_x
                oy1 = y1 + offset_y

                ox2 = x2 + offset_x
                oy2 = y2 + offset_y

                wounds.append({

                    "wound_id": index + 1,

                    "class_id": class_id,

                    "class_name": get_class_name(
                        wound_model,
                        class_id
                    ),

                    "confidence": round(
                        adjusted_confidence,
                        4
                    ),

                    "bbox_crop_xyxy": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ],

                    "bbox_original_xyxy": [
                        round(ox1, 2),
                        round(oy1, 2),
                        round(ox2, 2),
                        round(oy2, 2)
                    ]

                })

                all_adjusted_scores.append(
                    adjusted_confidence
                )

        wound_count = len(wounds)

        if wound_count > 0:
            injured_people += 1

        total_wounds += wound_count

        wound_person_results.append({

            "person_id": person_id,

            "evaluation_status": "completed",

            "wound_detected": wound_count > 0,

            "wound_count": wound_count,

            "wounds": wounds

        })

        print(
            "Wounds detected:",
            wound_count
        )

    overall_injury_score = (

        max(all_adjusted_scores)
        if all_adjusted_scores
        else 0.0

    )

    wound_data = {

        "confidence_threshold": WOUND_CONF,

        "image_size": WOUND_IMGSZ,

        "confidence_adjustment": {

            "method": (
                "25_percent_uplift_with_minimum_0.50"
            ),

            "uplift_factor": UPLIFT_FACTOR,

            "minimum_score": MIN_DISPLAY_SCORE,

            "maximum_score": MAX_DISPLAY_SCORE,

            "note": (
                "Display-only transformation; not a "
                "calibrated probability or medical "
                "severity score."
            )

        },

        "people_total": len(people),

        "people_evaluated": people_evaluated,

        "people_skipped": people_skipped,

        "visually_injured_people": injured_people,

        "people_without_detected_wounds": (
            people_evaluated - injured_people
        ),

        "total_wounds_detected": total_wounds,

        "overall_injury_score": round(
            overall_injury_score,
            4
        ),

        "score_method": (
            "maximum_adjusted_wound_confidence"
        ),

        "people": wound_person_results

    }

    return wound_data


# ============================================================
# ANNOTATED IMAGE
# ============================================================

def create_annotated_image(
    original,
    accident_data,
    people,
    wound_data,
    output_path
):

    annotated = original.copy()

    # Accident boxes — blue

    for box in accident_data["bounding_boxes"]:

        x1, y1, x2, y2 = map(
            int,
            box["bbox_xyxy"]
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        cv2.putText(

            annotated,

            f"ACCIDENT {box['confidence']:.2f}",

            (x1, max(25, y1 + 25)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 0, 0),

            2

        )

    # Person boxes — green

    for person in people:

        x1, y1, x2, y2 = map(
            int,
            person["bbox_original_xyxy"]
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(

            annotated,

            f"PERSON {person['person_id']}",

            (x1, max(25, y1 - 10)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2

        )

    # Wound boxes — yellow

    for person_result in wound_data["people"]:

        if (
            person_result["evaluation_status"]
            != "completed"
        ):
            continue

        for wound in person_result["wounds"]:

            x1, y1, x2, y2 = map(
                int,
                wound["bbox_original_xyxy"]
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )

            cv2.putText(

                annotated,

                f"WOUND {wound['confidence']:.2f}",

                (x1, max(25, y1 - 10)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 255),

                2

            )

    if not cv2.imwrite(
        str(output_path),
        annotated
    ):

        raise IOError(
            f"Could not save annotated image: {output_path}"
        )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def rescue_pipeline(image_path, models):

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    original = cv2.imread(
        str(image_path)
    )

    if original is None:

        raise ValueError(
            f"Unable to read input image: {image_path}"
        )

    height, width = original.shape[:2]

    (
        _,
        json_dir,
        crop_dir,
        annotated_dir
    ) = create_output_folders(image_path)

    json_path = (
        json_dir / "accident_detection.json"
    )

    annotated_path = (
        annotated_dir
        / f"{image_path.stem}_annotated.jpg"
    )

    print("\n" + "=" * 60)
    print("RESCUE-AI PIPELINE")
    print("=" * 60)

    print("Image:", image_path.name)
    print("Device:", DEVICE)

    data = {

        "image": {

            "filename": image_path.name,

            "width": width,

            "height": height

        },

        "accident_detection": {},

        "person_detection": {},

        "wound_detection": {},

        "status": "processing",

        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        )

    }

    # --------------------------------------------------------
    # STAGE 1 — ACCIDENT DETECTION
    # --------------------------------------------------------

    print(
        "\n[1/3] Running Accident Detection..."
    )

    accident_data = detect_accident(
        image_path,
        models["accident"]
    )

    data["accident_detection"] = accident_data

    print(
        "Accident detected:",
        accident_data["detected"]
    )

    print(
        "Accident boxes:",
        len(accident_data["bounding_boxes"])
    )

    # Stop downstream processing if no accident.

    if not accident_data["detected"]:

        data["person_detection"] = {

            "detected": False,

            "count": 0,

            "confidence_threshold": PERSON_CONF,

            "image_size": PERSON_IMGSZ,

            "people": []

        }

        data["wound_detection"] = {

            "confidence_threshold": WOUND_CONF,

            "people_total": 0,

            "people_evaluated": 0,

            "people_skipped": 0,

            "visually_injured_people": 0,

            "people_without_detected_wounds": 0,

            "total_wounds_detected": 0,

            "overall_injury_score": 0.0,

            "score_method": (
                "maximum_adjusted_wound_confidence"
            ),

            "people": []

        }

        data["status"] = "no_accident_detected"

        create_annotated_image(

            original,

            accident_data,

            [],

            {"people": []},

            annotated_path

        )

        save_json(
            data,
            json_path
        )

        print(
            "\nNo accident detected. "
            "Person and wound detection skipped."
        )

        print("JSON:", json_path)

        print(
            "Annotated image:",
            annotated_path
        )

        return data

    # --------------------------------------------------------
    # STAGE 2 — PERSON DETECTION
    # --------------------------------------------------------

    print(
        "\n[2/3] Running Person Detection..."
    )

    people = detect_people(

        original,

        image_path,

        models["person"],

        crop_dir

    )

    data["person_detection"] = {

        "detected": len(people) > 0,

        "count": len(people),

        "confidence_threshold": PERSON_CONF,

        "image_size": PERSON_IMGSZ,

        "crop_padding": CROP_PADDING,

        "blur_threshold": BLUR_THRESHOLD,

        "people": people

    }

    print(
        "People detected:",
        len(people)
    )

    # --------------------------------------------------------
    # STAGE 3 — WOUND DETECTION
    # --------------------------------------------------------

    print(
        "\n[3/3] Running Wound Detection..."
    )

    wound_data = detect_wounds(
        people,
        models["wound"]
    )

    data["wound_detection"] = wound_data

    data["status"] = "completed"

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    create_annotated_image(

        original,

        accident_data,

        people,

        wound_data,

        annotated_path

    )

    save_json(
        data,
        json_path
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESCUE-AI PIPELINE COMPLETED")
    print("=" * 60)

    print(
        "Accident detected:",
        accident_data["detected"]
    )

    print(
        "People detected:",
        len(people)
    )

    print(
        "People evaluated:",
        wound_data["people_evaluated"]
    )

    print(
        "People skipped:",
        wound_data["people_skipped"]
    )

    print(
        "Visibly injured people:",
        wound_data["visually_injured_people"]
    )

    print(
        "Total wounds:",
        wound_data["total_wounds_detected"]
    )

    print(
        "Overall display score:",
        wound_data["overall_injury_score"]
    )

    print("\nJSON:", json_path)

    print(
        "Annotated image:",
        annotated_path
    )

    print(
        "Person crops:",
        crop_dir
    )

    return data


# ============================================================
# COMMANDS
# ============================================================

def run_check():

    if not check_weight_files():
        raise SystemExit(1)

    print(
        "\nAll configured weight files are present."
    )


def run_predict(source):

    image_files = get_image_files(source)

    models = load_models()

    print(
        f"\nFound {len(image_files)} image(s)."
    )

    successful = 0
    failed = 0

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"\n\nIMAGE {index}/{len(image_files)}: "
            f"{image_path.name}"
        )

        try:

            rescue_pipeline(
                image_path,
                models
            )

            successful += 1

        except Exception as error:

            failed += 1

            print(
                f"\nERROR processing {image_path.name}: "
                f"{error}"
            )

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)

    print(
        "Total images:",
        len(image_files)
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Output directory:",
        OUTPUT_DIR.resolve()
    )


# ============================================================
# MAIN DISPATCHER
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(__doc__)

        raise SystemExit(1)

    command = sys.argv[1].lower()

    if command == "check":

        run_check()

    elif command == "predict":

        if len(sys.argv) < 3:

            print(
                "Usage: python final_pipeline.py "
                "predict <image_or_folder>"
            )

            raise SystemExit(1)

        run_predict(
            sys.argv[2]
        )

    else:

        print(
            f"Unknown command: {command}"
        )

        print(
            "Available commands: check, predict"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()