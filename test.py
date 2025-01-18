import cv2
import mediapipe as mp

# MediaPipe Face Detection 초기화
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 감지
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # 감지 결과를 시각화
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        cv2.imshow('MediaPipe Face Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
            break

cap.release()
cv2.destroyAllWindows()
