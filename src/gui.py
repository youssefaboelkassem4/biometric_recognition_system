
import sys
import os
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Import the biometric pipeline modules ────────────────────────────────
from preprocessing import load_face_dataset
from feature_extraction import EigenfaceExtractor, LBPExtractor, HOGExtractor
from build_features import build_gallery_features
from matching import identify_subject, compute_all_scores
from fusion import FusedExtractor
from evaluation import compute_eer


# ── Main application class ───────────────────────────────────────────────
class FaceRecognitionApp:
    """Tkinter GUI that streams webcam video, detects faces, and identifies them."""

    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    CANVAS_W = 640
    CANVAS_H = 480

    def __init__(self, root):
        self.root = root
        self.root.title("Biometric Face Recognition")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        # Camera state
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detected_faces = []

        # Load Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

        # Build the GUI first so the user sees the window immediately
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load the biometric pipeline in the background
        self.status_var.set("Status: Loading biometric models…")
        self.root.update()
        self._load_pipeline()

    # ── Load biometric models & gallery ──────────────────────────────────
    def _load_pipeline(self):
        """Load the fused model, build gallery features, and compute the EER threshold."""

        # Load gallery data
        gallery, probes = load_face_dataset()

        # Load the pre-trained fused extractor
        fused_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "fused_model.pkl"
        )
        self.fused_ext = FusedExtractor.load(fused_path)

        # Build gallery feature vectors (one averaged template per subject)
        self.gallery_features = build_gallery_features(gallery, self.fused_ext.extract)

        # Build probe features and compute genuine/impostor scores for EER
        from build_features import build_probe_features
        probe_features = build_probe_features(probes, self.fused_ext.extract)
        genuine, impostor = compute_all_scores(
            self.gallery_features, probe_features, metric="euclidean"
        )
        _, self.threshold = compute_eer(genuine, impostor)

        # Load any previously enrolled faces from data/enrolled/
        self._load_enrolled_faces()

        self.status_var.set(
            f"Status: Ready — {len(self.gallery_features)} subjects loaded, "
            f"threshold={self.threshold:.4f}"
        )

    def _load_enrolled_faces(self):
        """Scan data/enrolled/ and add each person's face images to the gallery."""
        enroll_root = os.path.join(
            os.path.dirname(__file__), "..", "data", "enrolled"
        )
        if not os.path.isdir(enroll_root):
            return

        for name in sorted(os.listdir(enroll_root)):
            person_dir = os.path.join(enroll_root, name)
            if not os.path.isdir(person_dir):
                continue

            vectors = []
            for img_file in sorted(os.listdir(person_dir)):
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Preprocess exactly like preprocessing.py
                resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LANCZOS4)
                equalized = cv2.equalizeHist(resized)
                normalized = equalized.astype(np.float32) / 255.0
                vector = normalized.flatten()

                feat = self.fused_ext.extract(vector)
                vectors.append(feat)

            if vectors:
                self.gallery_features[name] = np.mean(vectors, axis=0)
                print(f"  Loaded enrolled face: {name} ({len(vectors)} images)")

    # ── Identification using the real pipeline ───────────────────────────
    def _identify_face(self, face_gray_128):
        """
        Takes a 128×128 grayscale uint8 face image,
        preprocesses it the same way preprocessing.py does,
        and runs it through the fused extractor + matching with threshold.
        """
        # Match preprocessing.py: equalize histogram → normalize to [0,1] → flatten
        equalized = cv2.equalizeHist(face_gray_128)
        normalized = equalized.astype(np.float32) / 255.0
        vector = normalized.flatten()

        # Extract fused feature vector
        probe_vec = self.fused_ext.extract(vector)

        # Match against gallery with EER threshold
        pred_id, distance = identify_subject(
            probe_vec, self.gallery_features,
            metric="euclidean", threshold=self.threshold,
        )
        return pred_id, distance

    # ── UI Construction ──────────────────────────────────────────────────
    def _build_ui(self):
        """Create all tkinter widgets."""

        # Title
        title = tk.Label(
            self.root,
            text="🔐 Biometric Face Recognition",
            font=("Segoe UI", 18, "bold"),
            fg="#cdd6f4",
            bg="#1e1e2e",
        )
        title.pack(pady=(16, 8))

        # Video canvas
        self.canvas = tk.Canvas(
            self.root,
            width=self.CANVAS_W,
            height=self.CANVAS_H,
            bg="#11111b",
            highlightthickness=2,
            highlightbackground="#45475a",
        )
        self.canvas.pack(padx=20, pady=(0, 10))

        self.canvas.create_text(
            self.CANVAS_W // 2, self.CANVAS_H // 2,
            text="Camera is off.\nPress 'Start Camera' to begin.",
            fill="#6c7086",
            font=("Segoe UI", 14),
            justify="center",
            tags="placeholder",
        )

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(pady=(0, 8))

        self.btn_camera = tk.Button(
            btn_frame,
            text="Start Camera",
            font=("Segoe UI", 12, "bold"),
            fg="#1e1e2e",
            bg="#a6e3a1",
            activebackground="#94e2d5",
            width=16,
            relief="flat",
            cursor="hand2",
            command=self._toggle_camera,
        )
        self.btn_camera.grid(row=0, column=0, padx=8)

        self.btn_identify = tk.Button(
            btn_frame,
            text="Capture & Identify",
            font=("Segoe UI", 12, "bold"),
            fg="#1e1e2e",
            bg="#89b4fa",
            activebackground="#74c7ec",
            width=16,
            relief="flat",
            cursor="hand2",
            command=self._capture_and_identify,
        )
        self.btn_identify.grid(row=0, column=1, padx=8)

        # Enroll new face button
        self.btn_enroll = tk.Button(
            btn_frame,
            text="Enroll Face",
            font=("Segoe UI", 12, "bold"),
            fg="#1e1e2e",
            bg="#f9e2af",
            activebackground="#f5c2e7",
            width=16,
            relief="flat",
            cursor="hand2",
            command=self._enroll_face,
        )
        self.btn_enroll.grid(row=0, column=2, padx=8)

        # Status label
        self.status_var = tk.StringVar(value="Status: Camera off")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 12),
            fg="#f5c2e7",
            bg="#1e1e2e",
        )
        self.status_label.pack(pady=(0, 16))

    # ── Camera Controls ──────────────────────────────────────────────────
    def _toggle_camera(self):
        if self.is_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Status: Could not open camera")
            return

        self.is_running = True
        self.btn_camera.configure(text="Stop Camera", bg="#f38ba8")
        self.canvas.delete("placeholder")
        self.status_var.set("Status: Camera running")
        self._update_frame()

    def _stop_camera(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.current_frame = None
        self.detected_faces = []

        self.btn_camera.configure(text="Start Camera", bg="#a6e3a1")
        self.status_var.set("Status: Camera off")

        self.canvas.delete("all")
        self.canvas.create_text(
            self.CANVAS_W // 2, self.CANVAS_H // 2,
            text="Camera is off.\nPress 'Start Camera' to begin.",
            fill="#6c7086",
            font=("Segoe UI", 14),
            justify="center",
            tags="placeholder",
        )

    # ── Frame Loop ───────────────────────────────────────────────────────
    def _update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Status:Failed to read frame")
            self._stop_camera()
            return

        frame = cv2.flip(frame, 1)  # mirror horizontally
        self.current_frame = frame.copy()

        # Resize frame to fit the canvas so there's no black bar
        frame = cv2.resize(frame, (self.CANVAS_W, self.CANVAS_H))

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.detected_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60),
        )

        for (x, y, w, h) in self.detected_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert BGR → RGB → PIL → ImageTk and draw on canvas
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas._imgtk = imgtk  # prevent garbage collection

        self.root.after(33, self._update_frame)

    # ── Crop face helper ──────────────────────────────────────────────────
    def _crop_current_face(self):
        """Crop the first detected face from the current frame.
        Returns a 128×128 grayscale numpy array, or None on failure."""
        if not self.is_running or self.current_frame is None:
            self.status_var.set("Status: Start the camera first!")
            return None

        if len(self.detected_faces) == 0:
            self.status_var.set("Status: No face detected — try again")
            return None

        x, y, w, h = self.detected_faces[0]

        # Scale coordinates back to the original (un-resized) frame
        frame_h, frame_w = self.current_frame.shape[:2]
        scale_x = frame_w / self.CANVAS_W
        scale_y = frame_h / self.CANVAS_H
        ox = int(x * scale_x)
        oy = int(y * scale_y)
        ow = int(w * scale_x)
        oh = int(h * scale_y)
        face_bgr = self.current_frame[oy:oy + oh, ox:ox + ow]

        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        return face_resized

    # ── Capture & Identify ───────────────────────────────────────────────
    def _capture_and_identify(self):
        face = self._crop_current_face()
        if face is None:
            return

        pred_id, distance = self._identify_face(face)

        if pred_id == "Unknown":
            self.status_var.set(
                f"Status: Unknown person (distance={distance:.4f} > threshold={self.threshold:.4f})"
            )
        else:
            self.status_var.set(
                f"Status: Identified as → {pred_id}  (distance={distance:.4f})"
            )

    # ── Enroll a new face ─────────────────────────────────────────────────
    def _enroll_face(self):
        """Capture 5 face frames, average their features, and add to the gallery."""

        # Ask the user for a name / ID
        name = simpledialog.askstring(
            "Enroll New Face",
            "Enter name or ID for this person:",
            parent=self.root,
        )
        if not name or not name.strip():
            return
        name = name.strip()

        # Check if the name already exists in the gallery
        if name in self.gallery_features:
            overwrite = messagebox.askyesno(
                "Already Enrolled",
                f"'{name}' is already in the gallery.\nOverwrite?",
            )
            if not overwrite:
                return

        NUM_CAPTURES = 5
        vectors = []
        saved_images = []

        # Create a folder to save the enrolled face images
        enroll_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "enrolled", name
        )
        os.makedirs(enroll_dir, exist_ok=True)

        self.status_var.set(f"Status: Enrolling '{name}' — capturing 0/{NUM_CAPTURES}")
        self.root.update()

        import time
        for i in range(NUM_CAPTURES):
            # Wait a moment between captures so we get slightly different angles
            time.sleep(0.4)

            # Grab a fresh frame
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    # Re-run face detection on the fresh frame
                    resized = cv2.resize(frame, (self.CANVAS_W, self.CANVAS_H))
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    self.detected_faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60),
                    )

            face = self._crop_current_face()
            if face is None:
                self.status_var.set(
                    f"Status:Lost face during capture {i+1}/{NUM_CAPTURES} — try again"
                )
                return

            # Preprocess exactly like preprocessing.py
            equalized = cv2.equalizeHist(face)
            normalized = equalized.astype(np.float32) / 255.0
            vector = normalized.flatten()

            # Extract fused feature vector
            feat = self.fused_ext.extract(vector)
            vectors.append(feat)

            # Save the face image to disk
            img_path = os.path.join(enroll_dir, f"{i+1}.jpg")
            cv2.imwrite(img_path, face)
            saved_images.append(img_path)

            self.status_var.set(
                f"Status:Enrolling '{name}' — captured {i+1}/{NUM_CAPTURES}"
            )
            self.root.update()

        # Average the feature vectors into one gallery template
        template = np.mean(vectors, axis=0)
        self.gallery_features[name] = template

        self.status_var.set(
            f"Status: Enrolled '{name}' — {NUM_CAPTURES} images saved, "
            f"{len(self.gallery_features)} subjects in gallery"
        )

    # ── Cleanup ──────────────────────────────────────────────────────────
    def _on_close(self):
        self._stop_camera()
        self.root.destroy()


# ── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
