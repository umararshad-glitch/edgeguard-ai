import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    face_scores = []

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.6
    ) as face_detection:

        while cap.isOpened() and frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    score = detection.score[0]
                    face_scores.append(score)

            frame_count += 1

    cap.release()

    # No face detected → uncertain
    if len(face_scores) == 0:
        return {
            "engine": "EdgeGuard-VIDEO",
            "mode": "OFFLINE",
            "prediction": "UNCERTAIN",
            "confidence": 0.0
        }

    avg_score = np.mean(face_scores)

    # Simple heuristic logic
    if avg_score < 0.65:
        prediction = "FAKE"
    elif avg_score < 0.75:
        prediction = "UNCERTAIN"
    else:
        prediction = "REAL"

    return {
        "engine": "EdgeGuard-VIDEO",
        "mode": "OFFLINE",
        "prediction": prediction,
        "confidence": round(avg_score * 100, 2)
    }
