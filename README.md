# Face Recognition with ArcFace ONNX and 5-Point Alignment

Reproduction of the book by Gabriel Baziramwabo.

## Setup
1. `pip install -r requirements.txt`
2. Download ArcFace model: Place `w600k_r50.onnx` from InsightFace buffalo_l as `models/embedder_arcface.onnx`
3. Run `python src/init_project.py` to create folders

## Validation Stages (Run in order)
- `python src/camera.py`
- `python src/detect.py`
- `python src/landmarks.py`
- `python src/haar_5pt.py`
- `python src/align.py`
- `python src/embed.py`

## Enrollment
`python src/enroll.py` — Enroll ≥10 identities, saves aligned crops in data/enroll/<name>/ and embeddings in data/db/face_db.pkl

## Threshold Evaluation
`python src/evaluate.py` — Shows plot, update THRESHOLD variable in recognize.py

## Live Recognition
`python src/recognize.py` — Reliable recognition + unknown rejection

## Key Details
- 5-point alignment using similarity transform (better than affine)
- Embeddings L2-normalized, mean per identity
- Threshold: 0.62 (from evaluation — genuine ~0.92 mean, impostor ~0.65)
- Unknown faces rejected properly

# Face Locking

Extension of the ArcFace face recognition project with face locking and action detection.

## How to Run
- Enroll identities as before.
- Run `python src/face_locking.py`
- Enter the target identity name when prompted.
- When the person appears, it locks (green box), tracks stably, detects actions, and logs to `data/<name>_history_*.txt`

## Face Locking Works
- Normal recognition until high-confidence match for target → lock.
- Once locked, focuses on the same face region (ignores others, tolerates brief misses up to 20 frames).

## Actions Detected
- Left/right movement: Nose x-position change >30 pixels.
- Blink: Average Eye Aspect Ratio (EAR) <0.25 using eye landmarks.
- Smile/laugh: Mouth width / inter-eye distance >1.8.

## History Files
Stored in `data/` as `<name>_history_YYYYMMDDHHMMSS.txt` with timestamped actions.