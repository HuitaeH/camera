import cv2
import mediapipe as mp
import numpy as np

class BodyAnalyzer:
    """
    Mediapipe Pose를 이용해 단일 인물의 몸(전신 스켈레톤) 랜드마크 검출 및 시각화를 담당.
    - detect_body(frame): Pose Landmarks 계산
    - draw_body(frame): 스켈레톤(관절 선) 및 바운딩박스 시각화
    - get_body_bbox(): 전체 몸을 감싸는 바운딩박스(xmin, ymin, xmax, ymax)를 반환
    - get_landmarks(): 33개 Pose Landmark의 (px, py, pz) 리스트 반환

    추가:
    - get_body_center(normalized=False): 몸 중심(평균 좌표) 반환
    - get_pose_balance(): 어깨선·골반선이 얼마나 수평인지 0~1 범위로 환산
    """

    def __init__(self, static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmarks = []  # [(px, py, pz), ... 33개]
        self.bbox = None     # (xmin, ymin, xmax, ymax)
        self.image_width = None
        self.image_height = None

    def detect_body(self, frame: np.ndarray):
        """
        Mediapipe Pose로 몸(전신) 랜드마크를 찾고, self.landmarks & self.bbox 를 업데이트
        """
        h, w = frame.shape[:2]
        self.image_width = w
        self.image_height = h

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks is None:
            self.landmarks = []
            self.bbox = None
            return

        # 33개 랜드마크
        landmark_list = []
        for lm in results.pose_landmarks.landmark:
            px, py = int(lm.x * w), int(lm.y * h)
            pz = lm.z  # 상대적 깊이
            landmark_list.append((px, py, pz))

        self.landmarks = landmark_list

        # 바운딩박스 계산
        x_coords = [lm[0] for lm in landmark_list]
        y_coords = [lm[1] for lm in landmark_list]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        self.bbox = (xmin, ymin, xmax, ymax)

    def draw_body(self, frame: np.ndarray, draw_bbox=True):
        """
        스켈레톤(관절 선)과 (옵션) 바운딩박스를 그려서 frame에 반영
        """
        POSE_CONNECTIONS = [
            (11, 13), (13, 15),  # 왼팔
            (12, 14), (14, 16),  # 오른팔
            (11, 12),            # 어깨
            (23, 24),            # 골반
            (11, 23), (12, 24),  # 상체-하체 연결
            (23, 25), (25, 27),  # 왼다리
            (24, 26), (26, 28)   # 오른다리
        ]

        if len(self.landmarks) == 0:
            return

        # 스켈레톤 그리기
        for idx1, idx2 in POSE_CONNECTIONS:
            if idx1 < len(self.landmarks) and idx2 < len(self.landmarks):
                x1, y1, _ = self.landmarks[idx1]
                x2, y2, _ = self.landmarks[idx2]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 각 점도 표시
        for (px, py, pz) in self.landmarks:
            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        # 바운딩박스 표시 (옵션)
        if draw_bbox and self.bbox is not None:
            (xmin, ymin, xmax, ymax) = self.bbox
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                          (0, 0, 255), 2)

    def get_body_bbox(self):
        """현재 프레임에서 추출한 몸 바운딩박스 반환"""
        return self.bbox

    def get_landmarks(self):
        """현재 프레임에서 추출한 33개 Pose 랜드마크 리스트 반환"""
        return self.landmarks

    def get_body_center(self, normalized=False):
        """
        33개 랜드마크의 평균 좌표를 '몸 중심'으로 간주하여 반환.
        normalized=True 이면 0~1 스케일 (width, height 기준),
        False 면 픽셀 좌표 그대로 반환.
        """
        if not self.landmarks:
            return None

        xs = [lm[0] for lm in self.landmarks]
        ys = [lm[1] for lm in self.landmarks]
        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)

        if normalized and self.image_width and self.image_height:
            return (avg_x / self.image_width, avg_y / self.image_height)
        else:
            return (avg_x, avg_y)

    def get_pose_balance(self):
        """
        간단한 포즈 균형 예시:
        - 어깨선: left_shoulder(11) vs right_shoulder(12)
        - 골반선: left_hip(23) vs right_hip(24)
        두 선의 높이 차를 측정해서, 0~1 범위 점수로 변환.
        1에 가까울수록 좌우가 수평(균형), 0이면 크게 기울었음.
        
        (응용) 어깨선 기울기와 골반선 기울기를 각각 계산해서 평균 or 가중합.
        """
        if len(self.landmarks) < 25:
            # 랜드마크가 25개도 안 되면 포즈 인식 실패로 본다
            return 0.0

        # 어깨
        left_shoulder = self.landmarks[11]  # (px, py, pz)
        right_shoulder = self.landmarks[12]
        # 골반
        left_hip = self.landmarks[23]
        right_hip = self.landmarks[24]

        # 높이(= y좌표) 차이
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
        hip_diff = abs(left_hip[1] - right_hip[1])

        # 간단히 두 차이를 합산
        total_diff = shoulder_diff + hip_diff

        # 차이가 0이면 완벽히 수평 => 점수 1
        # 차이가 50픽셀 이상이면 => 점수 0 (임의 스케일)
        max_tolerable = 50.0
        raw_score = 1.0 - (total_diff / max_tolerable)
        if raw_score < 0:
            raw_score = 0.0
        if raw_score > 1:
            raw_score = 1.0

        return raw_score
