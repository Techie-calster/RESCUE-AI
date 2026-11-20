
from pathlib import Path
from ultralytics import YOLO

# ==============================
# Project paths
# ==============================
BASE_DIR = Path(__file__).resolve().parent

# INPUT: Provide either a single image or a folder
INPUT_PATH = BASE_DIR / "test_image/crash.png"  # Or test image folder /test image path

OUTPUT_DIR = BASE_DIR / "outputs"

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ==============================
# Load pretrained YOLO model
# ==============================
model = YOLO("yolo26n.pt")

# ==============================
# Find images to test
# ==============================
if INPUT_PATH.is_file():
    if INPUT_PATH.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {INPUT_PATH}")

    images = [INPUT_PATH]

elif INPUT_PATH.is_dir():
    images = sorted(
        path for path in INPUT_PATH.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

else:
    raise FileNotFoundError(
        f"Input path does not exist: {INPUT_PATH}"
    )

if not images:
    raise FileNotFoundError(
        f"No supported images found in: {INPUT_PATH}"
    )

print(f"Total images to process: {len(images)}")

# ==============================
# Run person detection
# ==============================
for image_path in images:

    result = model.predict(
        source=str(image_path),
        classes=[0],
        conf=0.20,
        iou=0.50,
        imgsz=960,
        max_det=100,
        save=True,
        project=str(OUTPUT_DIR),
        name="person_detection_tuned",
        exist_ok=True,
        verbose=False,
    )[0]

    boxes = result.boxes
    person_count = len(boxes) if boxes is not None else 0

    print(f"\nImage: {image_path.name}")
    print(f"People detected: {person_count}")

    if boxes is not None:
        for index, box in enumerate(boxes, start=1):

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            print(
                f"Person {index}: "
                f"confidence={confidence:.3f}, "
                f"bbox=({x1}, {y1}, {x2}, {y2})"
            )

# ==============================
# Output location
# ==============================
print(
    f"\nAnnotated images saved in: "
    f"{OUTPUT_DIR / 'person_detection_tuned'}"
)
