import numpy as np
from data_models.metrics import CompositionMetrics

class CompositionAnalyzer:
    def __init__(self):
        self.thirds_threshold = 0.1  # 3분할 선과의 거리 임계값
        self.golden_ratio = 1.618  # 황금비
    
    def analyze_composition(
        self,
        face_center: tuple,
        body_center: tuple = None,
        pose_balance: float = 0.0
    ) -> CompositionMetrics:
        """
        얼굴 위치 + (옵션) 몸 중심, 포즈 균형을 입력받아 구도 분석.
        face_center, body_center는 (x, y) 형태 (0~1 스케일).
        pose_balance는 0~1 범위 (1에 가까울수록 균형).
        """
        # ============== 얼굴 중심 기반 기존 계산 ==============
        x, y = face_center
        
        # 1) 3분할 구도
        thirds_x = abs(x - 1/3) < self.thirds_threshold or abs(x - 2/3) < self.thirds_threshold
        thirds_y = abs(y - 1/3) < self.thirds_threshold or abs(y - 2/3) < self.thirds_threshold
        rule_of_thirds = thirds_x or thirds_y
        
        # 2) 대칭성 점수(중심선과의 거리)
        symmetry_score = 1 - abs(x - 0.5) * 2
        
        # 3) 중심 가중치
        center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        center_weight = 1 - min(center_dist * 2, 1)
        
        # 4) 황금비 점수
        golden_points = [
            (1/self.golden_ratio, 1/self.golden_ratio),
            (1 - 1/self.golden_ratio, 1/self.golden_ratio),
            (1/self.golden_ratio, 1 - 1/self.golden_ratio),
            (1 - 1/self.golden_ratio, 1 - 1/self.golden_ratio)
        ]
        min_golden_dist = min(
            np.sqrt((x - px)**2 + (y - py)**2)
            for px, py in golden_points
        )
        golden_ratio_score = 1 - min(min_golden_dist * 2, 1)

        # ============== 추가: 몸 중심(body_center) ==============
        body_center_weight = 0.0
        if body_center is not None:
            bx, by = body_center
            b_dist = np.sqrt((bx - 0.5)**2 + (by - 0.5)**2)
            # 1에 가까울수록 중앙
            body_center_weight = 1 - min(b_dist * 2, 1)

        # ============== 추가: 포즈 균형(pose_balance) ==============
        # 이미 0~1로 들어온다고 가정
        # (1이면 매우 균형, 0이면 크게 기울어짐)
        
        return CompositionMetrics(
            rule_of_thirds=rule_of_thirds,
            symmetry_score=symmetry_score,
            center_weight=center_weight,
            golden_ratio=golden_ratio_score,
            body_center_weight=body_center_weight,
            pose_balance=pose_balance
        )
    
    def calculate_overall_score(self, composition_metrics: CompositionMetrics) -> float:
        """
        구도 점수 계산
        - 기존: 3분할(0.3), 대칭성(0.2), 중심(0.2), 황금비(0.3)
        - 새 필드 가중치: body_center_weight, pose_balance
        """
        weights = {
            'rule_of_thirds': 0.25,
            'symmetry': 0.15,
            'center_weight': 0.2,
            'golden_ratio': 0.2,
            'body_center_weight': 0.1,
            'pose_balance': 0.1
        }
        
        score = (
            weights['rule_of_thirds']    * float(composition_metrics.rule_of_thirds) +
            weights['symmetry']          * composition_metrics.symmetry_score +
            weights['center_weight']     * composition_metrics.center_weight +
            weights['golden_ratio']      * composition_metrics.golden_ratio +
            weights['body_center_weight']* composition_metrics.body_center_weight +
            weights['pose_balance']      * composition_metrics.pose_balance
        )
        
        return score
