from pathlib import Path
import cv2
import numpy as np

def load_face_dataset(img_size=(128, 128)):
    base_path = Path(__file__).resolve().parent.parent / "data" / "Full_data"
    training = []
    testing  = []

    for subfolder in sorted(Path(base_path).iterdir()):
        if not subfolder.is_dir():
            continue

        img_files = sorted([
            p for p in subfolder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])

        for idx, img_path in enumerate(img_files):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[SKIP] Failed to read: {img_path}")
                continue

            gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized    = cv2.resize(gray, img_size, interpolation=cv2.INTER_LANCZOS4)
            equalized  = cv2.equalizeHist(resized)
            normalized = equalized.astype(np.float32) / 255.0
            vector     = normalized.flatten()

            if idx < 12:
              training.append((subfolder.name, vector))
            else:
                testing.append((subfolder.name, vector))

    return np.array(training), np.array(testing)