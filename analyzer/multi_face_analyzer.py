# analyzer/multi_face_analyzer.py

import cv2
import mediapipe as mp
import numpy as np

class MultiFaceAnalyzer:
    """
    여러 얼굴을 FaceMesh로 검출하고, 선택(포커스)할 수 있게 도와주는 클래스.

    - detect_faces():
      프레임에서 여러 얼굴 랜드마크를 추출하고,
      각 얼굴의 (xmin, ymin, xmax, ymax) 바운딩박스를 계산하여 self.face_info 리스트에 저장.

    - draw_faces():
      각 얼굴 바운딩박스와 (옵션) 랜드마크를 그려줌.
      선택된 얼굴은 다른 색/두께로 강조.

    - select_face():
      마우스 클릭 좌표(x,y)가 어느 얼굴 바운딩박스 내부인지 판단, 해당 인덱스를 선택.
    """

    def __init__(self, max_num_faces=5, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 255),
            thickness=1,
            circle_radius=1
        )
        # 여러 얼굴 정보를 저장할 구조
        # 예: self.face_info = [
        #   {
        #       "bbox": (xmin, ymin, xmax, ymax),
        #       "landmarks": [(x1, y1, z1), (x2, y2, z2), ...],
        #   },
        #   ...
        # ]
        self.face_info = []
        self.selected_face_index = None

    def detect_faces(self, frame):
        """
        Mediapipe FaceMesh로 여러 얼굴 랜드마크 탐지 후, 바운딩박스 계산.
        -> self.face_info 에 저장
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        self.face_info = []
        if not results.multi_face_landmarks:
            return

        h, w = frame.shape[:2]
        # 여러 얼굴 반복
        for face_landmarks in results.multi_face_landmarks:
            x_vals = []
            y_vals = []
            # 468개 랜드마크
            for lm in face_landmarks.landmark:
                x_vals.append(lm.x)
                y_vals.append(lm.y)
            # min/max를 통해 바운딩박스 계산
            xmin = int(min(x_vals) * w)
            xmax = int(max(x_vals) * w)
            ymin = int(min(y_vals) * h)
            ymax = int(max(y_vals) * h)

            # landmarks (픽셀좌표) 리스트
            landmark_points = []
            for lm in face_landmarks.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pz = lm.z  # 깊이값(상대적), 보통 시각화는 x,y만
                landmark_points.append((px, py, pz))

            info = {
                "bbox": (xmin, ymin, xmax, ymax),
                "landmarks": landmark_points
            }
            self.face_info.append(info)

    def draw_faces(self, frame, draw_landmarks=True):
        """
        - 각 얼굴의 바운딩박스를 그려줌
        - 선택된 얼굴은 빨간색/두껍게 강조
        - (옵션) draw_landmarks=True일 때, 랜드마크 점도 표시
        """
        for i, face_data in enumerate(self.face_info):
            (xmin, ymin, xmax, ymax) = face_data["bbox"]
            if i == self.selected_face_index:
                color = (0, 0, 255)  # 빨강
                thickness = 3
            else:
                color = (0, 255, 0)  # 초록
                thickness = 2

            # bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)

            # (옵션) 랜드마크 그리기
            if draw_landmarks:
                # 선택된 얼굴은 다른 색으로 그려도 됨
                landmark_color = (0, 255, 255) if i != self.selected_face_index else (0, 100, 255)
                for (px, py, pz) in face_data["landmarks"]:
                    cv2.circle(frame, (px, py), 1, landmark_color, -1)

    def select_face(self, x, y):
        """
        (x, y)를 클릭했을 때, face_info 중 어느 bbox 내부인지 확인.
        내부라면 selected_face_index 업데이트,
        아니면 선택 해제(None).
        """
        for i, face_data in enumerate(self.face_info):
            (xmin, ymin, xmax, ymax) = face_data["bbox"]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                self.selected_face_index = i
                print(f"[INFO] Face #{i} selected!")
                return
        self.selected_face_index = None
