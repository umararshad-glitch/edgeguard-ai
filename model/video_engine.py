import os
import cv2

# ---------------- SETUP ----------------
TEMP_FRAMES_DIR = "temp_frames"

# Load Haar cascade (comes with OpenCV)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- FRAME EXTRACTION ----------------
def extract_frames(video_path, every_n=10):
    os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            frame_path = os.path.join(
                TEMP_FRAMES_DIR, f"frame_{saved}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


# ---------------- FACE DETECTION ----------------
def count_faces(frame_path):
    img = cv2.imread(frame_path)
    if img is None:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return len(faces)


# ---------------- VIDEO ANALYSIS ----------------
def analyze_video(video_path):
    frame_count = extract_frames(video_path)

    face_frames = 0
    for fname in os.listdir(TEMP_FRAMES_DIR):
        if fname.endswith(".jpg"):
            if count_faces(os.path.join(TEMP_FRAMES_DIR, fname)) > 0:
                face_frames += 1

    # --------- SIMPLE & EXPLAINABLE HEURISTIC ---------
    if face_frames < max(3, frame_count * 0.3):
        prediction = "FAKE"
        confidence = 75
    else:
        prediction = "REAL"
        confidence = 65

    return {
        "prediction": prediction,
        "confidence": confidence,
        "engine": "Frame + Face Consistency (OpenCV)"
    }
