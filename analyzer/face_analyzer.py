# analyzer/face_analyzer.py
import cv2
import mediapipe as mp
import numpy as np
from data_models.metrics import FaceMetrics

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
    def analyze_face(self, image: np.ndarray) -> FaceMetrics:
        """이미지에서 얼굴을 분석하여 메트릭을 반환"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 얼굴 영역 계산
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 얼굴 비율 계산
        face_width = (x_max - x_min) * width
        face_height = (y_max - y_min) * height
        face_area = face_width * face_height
        total_area = width * height
        face_ratio = (face_area / total_area) * 100
        
        # 얼굴 중심점 계산
        face_center = (
            (x_min + x_max) / 2,
            (y_min + y_max) / 2
        )
        
        # 얼굴 각도 계산 (눈 위치 기반)
        LEFT_EYE = [33, 133]
        RIGHT_EYE = [362, 263]
        
        left_eye = np.mean([[landmarks[idx].x, landmarks[idx].y] for idx in LEFT_EYE], axis=0)
        right_eye = np.mean([[landmarks[idx].x, landmarks[idx].y] for idx in RIGHT_EYE], axis=0)
        angle = np.degrees(np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        ))
        
        return FaceMetrics(
            face_ratio=face_ratio,
            face_center=face_center,
            face_angle=angle,
            face_area=face_area
        )
    
    def get_facial_landmarks(self, image: np.ndarray) -> list:
        """얼굴 랜드마크 추출"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return []
            
        return [(landmark.x, landmark.y, landmark.z) 
                for landmark in results.multi_face_landmarks[0].landmark]