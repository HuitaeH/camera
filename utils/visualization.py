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
    
    def draw_dynamic_guides(self, image: np.ndarray, metrics: ImageAnalysis) -> np.ndarray:
        """움직이는 가이드라인 표시"""
        output = image.copy()
        h, w = image.shape[:2]
        
        # 현재 얼굴 위치
        current_x = int(metrics.face_metrics.face_center[0] * w)
        current_y = int(metrics.face_metrics.face_center[1] * h)
        
        # 가장 가까운 이상적 포인트 찾기
        ideal_points = [
            (w/3, h/3),      # 상단 좌측
            (2*w/3, h/3),    # 상단 우측
            (w/3, 2*h/3),    # 하단 좌측
            (2*w/3, 2*h/3),  # 하단 우측
            (w/2, h/2)       # 중앙
        ]
        
        nearest_point = min(ideal_points, 
                          key=lambda p: ((p[0]-current_x)**2 + (p[1]-current_y)**2)**0.5)
        
        # 가이드 화살표 그리기
        if ((nearest_point[0]-current_x)**2 + (nearest_point[1]-current_y)**2)**0.5 > 20:
            cv2.arrowedLine(output, 
                           (current_x, current_y),
                           (int(nearest_point[0]), int(nearest_point[1])),
                           (0, 255, 255), 2)
            
        return output
    
    def draw_pose_balance(self, image: np.ndarray, metrics: ImageAnalysis) -> np.ndarray:
        """포즈 균형도 시각화"""
        output = image.copy()
        h, w = image.shape[:2]
        
        if hasattr(metrics, 'pose_balance'):
            # 균형도 바 표시
            bar_width = 150
            bar_height = 20
            x = w - bar_width - 10
            y = 30
            
            # 배경 바
            cv2.rectangle(output, (x, y), (x + bar_width, y + bar_height), 
                         (50, 50, 50), -1)
            
            # 균형도 바
            balance_width = int(bar_width * metrics.pose_balance)
            cv2.rectangle(output, (x, y), (x + balance_width, y + bar_height),
                         (0, 255, 0), -1)
            
            # 텍스트
            cv2.putText(output, f"Pose Balance", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)

        return output    
    

    def draw_composition_quality_indicator(self, image: np.ndarray, metrics: ImageAnalysis) -> np.ndarray:
        """구도 품질 인디케이터"""
        output = image.copy()
        h, w = image.shape[:2]
        
        quality_score = metrics.quality_score
        
        # 원형 인디케이터
        center = (w - 50, h - 50)
        radius = 30
        
        # 배경 원
        cv2.circle(output, center, radius, (50, 50, 50), -1)
        
        # 품질 점수에 따른 색상 설정
        if quality_score > 0.8:
            color = (0, 255, 0)  # 녹색
        elif quality_score > 0.6:
            color = (0, 255, 255)  # 노란색
        else:
            color = (0, 0, 255)  # 빨간색
            
        # 점수 원호
        angle = int(360 * quality_score)
        cv2.ellipse(output, center, (radius, radius),
                   0, 0, angle, color, 3)
        
        # 텍스트 표시
        cv2.putText(output, f"{int(quality_score*100)}%",
                   (center[0]-20, center[1]+7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
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
        """전체 분석 결과 시각화 (업데이트)"""
        output = self.draw_composition_guides(image)
        output = self.draw_face_analysis(output, analysis.face_metrics)
        output = self.draw_pose_balance(output, analysis)
        output = self.draw_dynamic_guides(output, analysis)
        output = self.draw_composition_quality_indicator(output, analysis)
        output = self.draw_analysis_info(output, analysis)
        return output
    


    def draw_feedback_text(self, image: np.ndarray, metrics: ImageAnalysis) -> np.ndarray:
        """사용자 피드백 텍스트 표시"""
        output = image.copy()
        h, w = image.shape[:2]
        
        feedback_messages = []
        
        # 3분할 규칙
        if not metrics.composition_metrics.rule_of_thirds:
            feedback_messages.append("Move to thirds line")
            
        # 대칭성
        if metrics.composition_metrics.symmetry_score < 0.5:
            feedback_messages.append("Center the subject")
            
        # 얼굴 각도
        if abs(metrics.face_metrics.face_angle) > 15:
            feedback_messages.append("Level the face angle")
            
        # 포즈 균형
        if hasattr(metrics, 'pose_balance') and metrics.pose_balance < 0.6:
            feedback_messages.append("Balance the pose")
        
        # 피드백 메시지 표시
        for i, msg in enumerate(feedback_messages):
            y_pos = h - 100 + (i * 30)
            cv2.putText(output, msg, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
        
        return output