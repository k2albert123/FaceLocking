import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import pickle
import os
from datetime import datetime

# Config
THRESHOLD = 0.62
TARGET_NAME = input("Enter the identity to lock onto (e.g., your name): ").strip().lower()
MISS_TOLERANCE = 20  # Frames to tolerate no face before unlock
MOVEMENT_THRESHOLD = 30  # Pixels for left/right movement
BLINK_EAR_THRESHOLD = 0.25
SMILE_RATIO_THRESHOLD = 1.8

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
session = ort.InferenceSession("../models/embedder_arcface.onnx")

REF_POINTS = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)

INDICES_5PT = [33, 263, 1, 61, 291]

# Eye landmarks for EAR (left and right)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def preprocess(aligned):
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(aligned):
    blob = preprocess(aligned)
    emb = session.run(None, {'input.1': blob})[0][0]
    return emb / np.linalg.norm(emb)

def compute_ear(landmarks, eye_indices, h, w):
    points = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in eye_indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

# Load database
with open('../data/db/face_db.pkl', 'rb') as f:
    db = pickle.load(f)

reference = {}
for name, embs in db.items():
    mean_emb = np.mean(np.array(embs), axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    reference[name.lower()] = mean_emb

if TARGET_NAME not in reference:
    print(f"Warning: {TARGET_NAME} not found in database!")
    exit()

target_emb = reference[TARGET_NAME]

# State variables
locked = False
miss_count = 0
prev_nose_x = None
prev_mouth_width = None
history_file = None
locked_timestamp = None
prev_bbox = None

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]

    # Region of interest (full frame if not locked, expanded previous bbox if locked)
    if locked and prev_bbox:
        x, y, w, h = prev_bbox
        margin = 100
        roi_x = max(0, x - margin)
        roi_y = max(0, y - margin)
        roi_w = min(w_frame - roi_x, w + 2 * margin)
        roi_h = min(h_frame - roi_y, h + 2 * margin)
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        offset_x, offset_y = roi_x, roi_y
    else:
        roi = frame
        offset_x, offset_y = 0, 0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    face_found = len(faces) > 0

    if face_found:
        x, y, w, h = faces[0]
        x += offset_x
        y += offset_y
        crop = frame[y:y+h, x:x+w]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_crop)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            # 5-point alignment
            pts = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in INDICES_5PT], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)
            aligned = cv2.warpAffine(crop, M, (112, 112), flags=cv2.INTER_LINEAR)

            # Embedding and recognition
            query_emb = get_embedding(aligned)
            sim = np.dot(query_emb, target_emb)

            nose_x = int(lm.landmark[1].x * w) + x  # Nose tip
            eye_dist = np.linalg.norm(np.array([lm.landmark[33].x * w, lm.landmark[33].y * h]) -
                                      np.array([lm.landmark[263].x * w, lm.landmark[263].y * h]))
            mouth_width = np.linalg.norm(np.array([lm.landmark[61].x * w, lm.landmark[61].y * h]) -
                                         np.array([lm.landmark[291].x * w, lm.landmark[291].y * h]))

            # Action detection
            action = None
            if locked:
                # Movement
                if prev_nose_x is not None:
                    delta_x = nose_x - prev_nose_x
                    if delta_x > MOVEMENT_THRESHOLD:
                        action = f"moved right ({delta_x:.0f} pixels)"
                    elif delta_x < -MOVEMENT_THRESHOLD:
                        action = f"moved left ({abs(delta_x):.0f} pixels)"

                # Blink
                ear_left = compute_ear(lm, LEFT_EYE, h, w)
                ear_right = compute_ear(lm, RIGHT_EYE, h, w)
                ear = (ear_left + ear_right) / 2
                if ear < BLINK_EAR_THRESHOLD:
                    action = f"blink detected (EAR: {ear:.2f})"

                # Smile
                smile_ratio = mouth_width / eye_dist if eye_dist > 0 else 0
                if smile_ratio > SMILE_RATIO_THRESHOLD:
                    action = f"smile/laugh detected (ratio: {smile_ratio:.2f})"

                if action and history_file:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    with open(history_file, 'a') as f:
                        f.write(f"{timestamp} | {action}\n")

            # Locking logic
            if not locked and sim >= THRESHOLD:
                locked = True
                miss_count = 0
                locked_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                history_file = f"../data/{TARGET_NAME}_history_{locked_timestamp}.txt"
                with open(history_file, 'w') as f:
                    f.write(f"Face locking started for {TARGET_NAME} at {datetime.now()}\n")
                print(f"LOCKED onto {TARGET_NAME}")

            if locked:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame, f"LOCKED: {TARGET_NAME} ({sim:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Searching for {TARGET_NAME} ({sim:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.imshow('Aligned', aligned)
            prev_nose_x = nose_x
            prev_bbox = (x, y, w, h)
            miss_count = 0
        else:
            face_found = False
    else:
        if locked:
            miss_count += 1
            if miss_count > MISS_TOLERANCE:
                locked = False
                print("Lock released - face disappeared")
                if history_file:
                    with open(history_file, 'a') as f:
                        f.write(f"Lock released at {datetime.now()}\n")
                history_file = None

    cv2.imshow('Face Locking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()