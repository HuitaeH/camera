# utils/visualization.py
import cv2
import numpy as np
from data_models.metrics import ImageAnalysis, FaceMetrics, CompositionMetrics

class CompositionVisualizer:
    def __init__(self):
        self.colors = {
            'guides': (128, 128, 128),
            'face': (0, 255, 0),
            'text': (255, 255, 255)
        }
    
    def draw_composition_guides(self, image: np.ndarray) -> np.ndarray:
        """구도 가이드라인 그리기"""
        h, w = image.shape[:2]
        output = image.copy()
        
        # 3분할 선
        for i in range(1, 3):
            cv2.line(output, (w * i // 3, 0), (w * i // 3, h), self.colors['guides'], 1)
            cv2.line(output, (0, h * i // 3), (w, h * i // 3), self.colors['guides'], 1)
        
        # 황금비율 포인트
        golden_ratio = 1.618
        points = [
            (int(w/golden_ratio), int(h/golden_ratio)),
            (int(w - w/golden_ratio), int(h/golden_ratio)),
            (int(w/golden_ratio), int(h - h/golden_ratio)),
            (int(w - w/golden_ratio), int(h - h/golden_ratio))
        ]
        
        for point in points:
            cv2.circle(output, point, 3, self.colors['guides'], 1)
        
        return output
    
    def draw_face_analysis(self, image: np.ndarray, metrics: FaceMetrics) -> np.ndarray:
        """얼굴 분석 결과 시각화"""
        output = image.copy()
        h, w = image.shape[:2]
        
        # 얼굴 중심점
        center_x = int(metrics.face_center[0] * w)
        center_y = int(metrics.face_center[1] * h)
        cv2.circle(output, (center_x, center_y), 5, self.colors['face'], -1)
        
        # 각도 표시
        angle_line_length = 50
        end_x = center_x + int(angle_line_length * np.cos(np.radians(metrics.face_angle)))
        end_y = center_y + int(angle_line_length * np.sin(np.radians(metrics.face_angle)))
        cv2.line(output, (center_x, center_y), (end_x, end_y), self.colors['face'], 2)
        
        return output
    
    def draw_analysis_info(self, image: np.ndarray, analysis: ImageAnalysis) -> np.ndarray:
        """분석 정보 텍스트 표시"""
        output = image.copy()
        h, w = image.shape[:2]
        
        info_lines = [
            f"Face Ratio: {analysis.face_metrics.face_ratio:.1f}%",
            f"Angle: {analysis.face_metrics.face_angle:.1f}°",
            f"Rule of Thirds: {'Yes' if analysis.composition_metrics.rule_of_thirds else 'No'}",
            f"Symmetry: {analysis.composition_metrics.symmetry_score:.2f}",
            f"Quality Score: {analysis.quality_score:.2f}"
        ]
        
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(output, line, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return output
    
    def create_analysis_visualization(self, image: np.ndarray, analysis: ImageAnalysis) -> np.ndarray:
        """전체 분석 결과 시각화"""
        output = self.draw_composition_guides(image)
        output = self.draw_face_analysis(output, analysis.face_metrics)
        output = self.draw_analysis_info(output, analysis)
        return output