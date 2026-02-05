import cv2
import sys

# Get camera index from command line args, default to 0
cam_index = 0
if len(sys.argv) > 1:
    try:
        cam_index = int(sys.argv[1])
    except ValueError:
        print(f"Invalid camera index: {sys.argv[1]}. Using default 0.")

print(f"Opening camera {cam_index}...")
cap = cv2.VideoCapture(cam_index)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)
    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()