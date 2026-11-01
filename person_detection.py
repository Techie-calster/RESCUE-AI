
from pathlib import Path
from ultralytics import YOLO

# Project paths
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "test_image/crash.png"
OUTPUT_DIR = BASE_DIR / "outputs"

# Pretrained YOLO model
model = YOLO("yolo26n.pt")

# Find images to test
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
images = [
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in image_extensions
]

if not images:
    raise FileNotFoundError(
        f"No images found in {IMAGE_DIR}. "
        "Add an accident image and run again."
    )

# Run person detection
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

print(f"\nAnnotated images saved in: "
      f"{OUTPUT_DIR / 'person_detection'}")
