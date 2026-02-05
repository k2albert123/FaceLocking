import sys
import os

output_file = "debug_mp.txt"

with open(output_file, "w") as f:
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Python executable: {sys.executable}\n")
    try:
        import mediapipe as mp
        f.write(f"MediaPipe version: {getattr(mp, '__version__', 'unknown')}\n")
        f.write(f"MediaPipe file: {mp.__file__}\n")
        f.write(f"MediaPipe dir: {dir(mp)}\n")
        
        try:
            from mediapipe.python.solutions import face_mesh
            f.write("Successfully imported mediapipe.python.solutions.face_mesh\n")
        except ImportError as e:
            f.write(f"Failed to import face_mesh from mediapipe.python.solutions: {e}\n")
            
        try:
            import mediapipe.solutions as solutions
            f.write("Successfully imported mediapipe.solutions\n")
        except ImportError as e:
            f.write(f"Failed to import mediapipe.solutions: {e}\n")

    except ImportError:
        f.write("MediaPipe is not installed in this environment.\n")
    except Exception as e:
        f.write(f"An unexpected error occurred: {e}\n")

