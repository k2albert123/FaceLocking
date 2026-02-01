import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import pickle
import os

# -----------------------------
# Models
# -----------------------------
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

session = ort.InferenceSession("../models/embedder_arcface.onnx")
input_name = session.get_inputs()[0].name

# -----------------------------
# Alignment reference points
# -----------------------------
REF_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

INDICES = [33, 263, 1, 61, 291]  # eye, eye, nose, mouth, mouth


# -----------------------------
# Preprocess for ArcFace
# -----------------------------
def preprocess(aligned):
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


# -----------------------------
# Load DB
# -----------------------------
DB_PATH = "../data/db/face_db.pkl"

if os.path.exists(DB_PATH):
    with open(DB_PATH, 'rb') as f:
        db = pickle.load(f)
else:
    db = {}

# -----------------------------
# Enrollment Setup
# -----------------------------
name = input("Enter identity name: ").strip()
os.makedirs(f"../data/enroll/{name}", exist_ok=True)

embeddings = []
count = 0

cap = cv2.VideoCapture(0)

print("Look at camera. Auto-capture on good face. Aim for 15+ samples. Press Q to finish.")

# -----------------------------
# Capture Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    if len(faces) > 0:
        x, y, w, h = faces[0]

        # safety crop bounds
        x, y = max(0, x), max(0, y)
        crop = frame[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]

            ch, cw = crop.shape[:2]
            pts = np.array(
                [[lm.landmark[i].x * cw, lm.landmark[i].y * ch] for i in INDICES],
                dtype=np.float32
            )

            M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)

            if M is None:
                continue

            aligned = cv2.warpAffine(
                crop, M, (112, 112),
                flags=cv2.INTER_LINEAR,
                borderValue=0
            )

            blob = preprocess(aligned)

            # ✅ correct ONNX call
            emb = session.run(None, {input_name: blob})[0][0]

            # normalize embedding safely
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            emb = emb / norm

            embeddings.append(emb)
            count += 1

            cv2.imwrite(f"../data/enroll/{name}/{count:04d}.jpg", aligned)

            cv2.putText(
                frame,
                f"Captured {count}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.imshow("Saved Aligned", aligned)

    cv2.imshow("Enroll", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q OR ESC
        print("Exiting...")
        break

    if cv2.getWindowProperty('Enroll', cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed")
        break

# -----------------------------
# Save DB
# -----------------------------
if embeddings:
    db.setdefault(name, []).extend(embeddings)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)

    print(f"Enrolled {name} with {len(embeddings)} new samples (total: {len(db[name])})")
else:
    print("No samples captured.")

cap.release()
cv2.destroyAllWindows()
