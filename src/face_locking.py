import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import pickle
import os
import argparse
import sys
from datetime import datetime

# Parse arguments
parser = argparse.ArgumentParser(description="Face Locking System")
parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
parser.add_argument("--target", type=str, help="Identity to lock onto")
args = parser.parse_args()

# Config
THRESHOLD = 0.62
if args.target:
    TARGET_NAME = args.target.strip().lower()
else:
    TARGET_NAME = input("Enter the identity to lock onto (e.g., your name): ").strip().lower()

MISS_TOLERANCE = 20  # Frames to tolerate no face before unlock
MOVEMENT_THRESHOLD = 30  # Pixels for left/right movement
BLINK_EAR_THRESHOLD = 0.25
SMILE_RATIO_THRESHOLD = 1.8

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    try:
        import mediapipe.solutions.face_mesh as mp_face_mesh
        import mediapipe.solutions.drawing_utils as mp_drawing
    except AttributeError:
        # Fallback for some installs
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Resolve paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

model_path = os.path.join(ROOT_DIR, "models", "embedder_arcface.onnx")
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

try:
    session = ort.InferenceSession(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

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
db_path = os.path.join(ROOT_DIR, "data", "db", "face_db.pkl")
if not os.path.exists(db_path):
    print(f"Error: Database not found at {db_path}")
    sys.exit(1)

with open(db_path, 'rb') as f:
    db = pickle.load(f)

reference = {}
for name, embs in db.items():
    mean_emb = np.mean(np.array(embs), axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    reference[name.lower()] = mean_emb

if TARGET_NAME not in reference:
    print(f"Warning: '{TARGET_NAME}' not found in database! Available: {list(reference.keys())}")
    # We allow it to run, but it won't lock
    # exit() # assignment implies we should select a VALID identity, but helpful for debugging to see available

if TARGET_NAME in reference:
    target_emb = reference[TARGET_NAME]
else:
    target_emb = None

def compute_pose(landmarks, h, w):
    # Simplified head pose estimation using key landmarks
    # Nose: 1, Left Eye: 33, Right Eye: 263, Mouth Left: 61, Mouth Right: 291, Chin: 152
    nose = np.array([landmarks.landmark[1].x * w, landmarks.landmark[1].y * h])
    left_eye = np.array([landmarks.landmark[33].x * w, landmarks.landmark[33].y * h])
    right_eye = np.array([landmarks.landmark[263].x * w, landmarks.landmark[263].y * h])
    chin = np.array([landmarks.landmark[152].x * w, landmarks.landmark[152].y * h])
    
    # Roll: Angle between eyes
    eye_delta = right_eye - left_eye
    roll = np.degrees(np.arctan2(eye_delta[1], eye_delta[0]))
    
    # Yaw: Horizontal symmetry (dist between nose and eyes)
    dist_l = np.linalg.norm(nose - left_eye)
    dist_r = np.linalg.norm(nose - right_eye)
    yaw = (dist_r - dist_l) / (dist_l + dist_r) * 100 # percentage-like
    
    # Pitch: Vertical symmetry
    eye_center = (left_eye + right_eye) / 2
    dist_nose_eyes = np.linalg.norm(nose - eye_center)
    dist_nose_chin = np.linalg.norm(nose - chin)
    pitch = (dist_nose_chin - dist_nose_eyes) / (dist_nose_eyes + dist_nose_chin) * 100
    
    return roll, yaw, pitch

# State variables
locked = False
miss_count = 0
person_prev_nose_x = {} # Tracking nose x for all enrolled people
history_file = None
locked_timestamp = None
prev_bbox = None
current_people_in_frame = set()

print(f"\n" + "="*50)
print(f"Starting Face Locking & Scanning System")
print(f"Target: {TARGET_NAME}")
print(f"Detection Range: ENHANCED (Anywhere)")
print(f"Scanning Mode: Granular Trajectory + Pose")
print(f"Press 'q' to quit")
print("="*50 + "\n")

cap = cv2.VideoCapture(args.camera)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]

    # Region of interest (wider for better recovery/range)
    if locked and prev_bbox:
        x, y, w, h = prev_bbox
        margin = 300 # Even wider margin for "anywhere" feel
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
    # Reduced minSize to (30, 30) for much better distance detection
    faces_detected = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    target_recognized_this_frame = False
    detected_this_frame = set()

    for (fx, fy, fw, fh) in faces_detected:
        x, y, w, h = fx + offset_x, fy + offset_y, fw, fh
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0: continue
        
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_crop)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            # Pose Estimation
            roll, yaw, pitch = compute_pose(lm, h, w)
            
            # 5-point alignment
            pts = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in INDICES_5PT], dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)
            aligned = cv2.warpAffine(crop, M, (112, 112), flags=cv2.INTER_LINEAR)

            # Embedding and recognition
            query_emb = get_embedding(aligned)
            
            max_sim = -1
            identity = "Unknown"
            for name, ref_emb in reference.items():
                sim = np.dot(query_emb, ref_emb)
                if sim > max_sim:
                    max_sim = sim
                    identity = name
            
            label = identity if max_sim >= THRESHOLD else "Unknown"
            detected_this_frame.add(label)
            
            nose_x = int(lm.landmark[1].x * w) + x
            
            if label != "Unknown":
                # Descriptive Movement Scanning for ALL enrolled people
                if label in person_prev_nose_x:
                    delta_x = nose_x - person_prev_nose_x[label]
                    if delta_x > MOVEMENT_THRESHOLD:
                        print(f"[ACTION] {label.upper()} moved to the right")
                    elif delta_x < -MOVEMENT_THRESHOLD:
                        print(f"[ACTION] {label.upper()} moved to the left")
                person_prev_nose_x[label] = nose_x

                if label not in current_people_in_frame:
                    print(f"[SCAN] {label.upper()} initialized (sim: {max_sim:.2f})")
            
            is_target = (label.lower() == TARGET_NAME.lower())
            
            if is_target:
                target_recognized_this_frame = True
                sim_target = max_sim
                
                # Locking logic
                if not locked:
                    locked = True
                    miss_count = 0
                    locked_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    history_file = os.path.join(ROOT_DIR, "data", f"{TARGET_NAME}_history_{locked_timestamp}.txt")
                    with open(history_file, 'w') as f:
                        f.write(f"Comprehensive scanning started for {TARGET_NAME} at {datetime.now()}\n")
                        f.write("Timestamp | Pos(X,Y) | Size(W,H) | Pose(R,Y,P) | Actions\n")
                    print(f"[LOCKED] Wide-tracking active for {TARGET_NAME.upper()}")
                
                # Granular Scanning Data
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                pos_str = f"({x},{y})"
                size_str = f"({w}x{h})"
                pose_str = f"R:{roll:+.1f} Y:{yaw:+.1f} P:{pitch:+.1f}"
                
                # Terminal output for "Every Movement" (Throttled or concise)
                if cv2.getTickCount() % 10 == 0: # Increased throttle to make deskcrptive actions clearer
                    print(f"[SCANNING] {TARGET_NAME.upper()} | Pos: {pos_str} | Pose: {pose_str}")

                # Detect discrete actions too
                action = ""
                eye_dist = np.linalg.norm(np.array([lm.landmark[33].x * w, lm.landmark[33].y * h]) -
                                          np.array([lm.landmark[263].x * w, lm.landmark[263].y * h]))
                mouth_width = np.linalg.norm(np.array([lm.landmark[61].x * w, lm.landmark[61].y * h]) -
                                             np.array([lm.landmark[291].x * w, lm.landmark[291].y * h]))
                
                ear = (compute_ear(lm, LEFT_EYE, h, w) + compute_ear(lm, RIGHT_EYE, h, w)) / 2
                if ear < BLINK_EAR_THRESHOLD: action += "BLINK "
                if (mouth_width / eye_dist if eye_dist > 0 else 0) > SMILE_RATIO_THRESHOLD: action += "SMILE "
                
                # Detailed Log Entry
                if history_file:
                    with open(history_file, 'a') as f:
                        f.write(f"{timestamp} | {pos_str:10} | {size_str:10} | {pose_str:25} | {action}\n")

                prev_bbox = (x, y, w, h)
                miss_count = 0
                
                # UI for target (crosshair style)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.line(frame, (x+w//2, y), (x+w//2, y+h), (0, 255, 0), 1) # Vertical center
                cv2.line(frame, (x, y+h//2), (x+w, y+h//2), (0, 255, 0), 1) # Horizontal center
                cv2.putText(frame, f"LOCKED: {label.upper()}", (x, y-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"POSE: {pose_str}", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                color = (255, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
                cv2.putText(frame, f"{label}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Release logic
    if not target_recognized_this_frame:
        if locked:
            miss_count += 1
            if miss_count > MISS_TOLERANCE:
                locked = False
                print(f"[LOST] Lost signal on {TARGET_NAME.upper()}")
                history_file = None
                prev_bbox = None

    # Handle disappearances for terminal
    for person in current_people_in_frame - detected_this_frame:
        if person != "Unknown":
            print(f"[LOST] {person.upper()} signal lost")
            if person in person_prev_nose_x:
                del person_prev_nose_x[person]
    current_people_in_frame = detected_this_frame

    cv2.imshow('Wide-Range Face Locking & Scanning', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

