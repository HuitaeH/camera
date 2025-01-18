# data_models/metrics.py
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class FaceMetrics:
    face_ratio: float  # 얼굴이 차지하는 비율
    face_center: Tuple[float, float]  # 얼굴 중심점 (x, y)
    face_angle: float  # 얼굴 각도
    face_area: float  # 얼굴 영역 크기

@dataclass
class CompositionMetrics:
    rule_of_thirds: bool  # 3분할 구도 적용 여부
    symmetry_score: float  # 대칭성 점수
    center_weight: float  # 중심 구도 점수
    golden_ratio: float  # 황금비 점수

    # 추가 필드
    body_center_weight: float = 0.0
    pose_balance: float = 0.0

@dataclass
class ImageAnalysis:
    image_path: str
    face_metrics: Optional[FaceMetrics]
    composition_metrics: CompositionMetrics
    quality_score: float  # 전체 품질 점수

@dataclass
class AnalysisStatistics:
    avg_face_ratio: float
    avg_face_center: Tuple[float, float]
    preferred_angles: List[float]
    composition_preferences: dict
    quality_threshold: float