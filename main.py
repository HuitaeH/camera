# main.py
from analyzer.multi_face_analyzer import MultiFaceAnalyzer  # FaceMesh 기반 다중 얼굴 분석
from analyzer.body_analyzer import BodyAnalyzer            # Pose 기반 몸(전신) 분석

import os
import sys
import io
import cv2
import argparse
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import List
import tensorflow as tf

# ========== ENV & LOGGING 설정 ==========
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 사용 (첫 번째 GPU 선택)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'  # GPU 활성화

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증가 허용
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 혹은 특정 메모리 제한을 두고 싶다면:
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB 제한
        # )
    except RuntimeError as e:
        print(f"GPU 설정 중 오류 발생: {e}")

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# ========== IMPORT (기존) ==========
from analyzer.face_analyzer import FaceAnalyzer
from analyzer.composition import CompositionAnalyzer
from data_models.metrics import ImageAnalysis, CompositionMetrics
from utils.visualization import CompositionVisualizer
from utils.image_utils import ImageProcessor

@contextmanager
def suppress_stdout_stderr():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


class PhotoCompositionAnalyzer:
    def __init__(self, body_analyzer=None):
        """
        body_analyzer: BodyAnalyzer 인스턴스 (Mediapipe Pose)
                       없으면(None) 몸 분석은 스킵
        """
        print("PhotoCompositionAnalyzer 초기화 중...")
        self.face_analyzer = FaceAnalyzer()             # 얼굴(FaceMesh) 분석
        self.composition_analyzer = CompositionAnalyzer() 
        self.visualizer = CompositionVisualizer()
        self.body_analyzer = body_analyzer              # 몸 분석기 주입
        print("PhotoCompositionAnalyzer 초기화 완료!")

    def analyze_frame(self, frame, image_path="(Camera Frame)"):
        """
        얼굴(FaceMesh) + (옵션) 몸(Pose) 분석 → 구도 점수 산출
        한 사람(얼굴 1개) 전용
        """
        # 1) 얼굴 분석
        face_metrics = self.face_analyzer.analyze_face(frame)
        if face_metrics is None:
            out_frame = frame.copy()
            cv2.putText(out_frame, "No face detected.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return out_frame, None

        # 2) 몸 중심 / 포즈 균형 (이미 main 루프에서 body_analyzer.detect_body(frame)는 호출했다고 가정)
        body_center = (0.5, 0.5)   # 기본값 (화면 중앙)
        pose_balance = 0.0
        if self.body_analyzer is not None:
            bc = self.body_analyzer.get_body_center(normalized=True)
            if bc:
                body_center = bc
            pose_balance = self.body_analyzer.get_pose_balance()

        # 3) 구도 분석
        composition_metrics = self.composition_analyzer.analyze_composition(
            face_center=face_metrics.face_center,  # (x,y) 0~1
            body_center=body_center, 
            pose_balance=pose_balance
        )
        overall_score = self.composition_analyzer.calculate_overall_score(composition_metrics)

        # 4) 결과 저장
        analysis_result = ImageAnalysis(
            image_path=image_path,
            face_metrics=face_metrics,
            composition_metrics=composition_metrics,
            quality_score=overall_score
        )

        # 5) 시각화
        out_frame = self.visualizer.create_analysis_visualization(frame, analysis_result)
        return out_frame, analysis_result

    def close(self):
        self.face_analyzer.face_mesh.close()


# ====== 전역 / 마우스 콜백 설정 ======
multi_face_analyzer = MultiFaceAnalyzer()  # FaceMesh 다중 얼굴 분석
analyzer = None             # PhotoCompositionAnalyzer (얼굴+몸 구도)
body_analyzer = None        # Pose 기반 전신 분석

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        multi_face_analyzer.select_face(x, y)


def main():
    # GPU 확인
    print("\nGPU 확인:")
    print("TensorFlow:", tf.config.list_physical_devices('GPU'))
    print("CUDA 사용 가능:", tf.test.is_built_with_cuda())
    print("GPU 사용 가능:", tf.test.is_built_with_gpu_support())

    global analyzer, body_analyzer
    parser = argparse.ArgumentParser(description='Photo Composition Analyzer (Faces + Body)')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory containing images to analyze')
    args = parser.parse_args()

    # 출력 디렉토리
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)

    # 1) 전신 분석기 초기화
    body_analyzer = BodyAnalyzer(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        gpu_id=0  # GPU 사용 설정 추가
    )

    # 2) 얼굴+몸 구도 분석기 (body_analyzer 주입)
    analyzer = PhotoCompositionAnalyzer(body_analyzer=body_analyzer)

    # ======= 여러 이미지 결과 누적 (옵션) =======
    composition_list: List[CompositionMetrics] = []
    overall_scores: List[float] = []

    # (1) 여러 이미지 분석 후 평균 계산
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"[오류] '{input_dir}' 폴더가 유효하지 않습니다.")
            sys.exit(1)

        image_files = ImageProcessor.get_image_files(input_dir)
        if len(image_files) == 0:
            print("[안내] 폴더 내에 이미지가 없습니다. -> 카메라 분석으로 이동합니다.")
        else:
            print(f"[안내] {len(image_files)}장 이미지 분석을 시작합니다...")

            for i, img_path in enumerate(image_files, 1):
                print(f"\n[{i}/{len(image_files)}] '{img_path.name}' 분석 중...")
                image = ImageProcessor.load_image(img_path)
                image = ImageProcessor.resize_image(image, max_size=1280)

                # - Pose detect (한 장 이미지에서도 가능)
                body_analyzer.detect_body(image)

                # - 얼굴+몸 구도 분석
                analyzed_frame, analysis_result = analyzer.analyze_frame(image, image_path=str(img_path))

                window_name = f"Analysis {i}/{len(image_files)}"
                cv2.imshow(window_name, analyzed_frame)

                if analysis_result:
                    composition_list.append(analysis_result.composition_metrics)
                    overall_scores.append(analysis_result.quality_score)

                print("결과 확인 후 아무 키를 누르면 다음 이미지로 넘어갑니다. (ESC: 전체 종료)")
                while True:
                    key = cv2.waitKey(10) & 0xFF
                    if key == 27:
                        print("사용자에 의해 프로그램이 종료되었습니다.")
                        cv2.destroyAllWindows()
                        analyzer.close()
                        sys.exit(0)
                    elif key != 255:
                        break

                cv2.destroyWindow(window_name)

    # (2) 평균값 계산
    if len(composition_list) > 0:
        sum_sym = sum(cm.symmetry_score for cm in composition_list)
        sum_center = sum(cm.center_weight for cm in composition_list)
        sum_gold = sum(cm.golden_ratio for cm in composition_list)
        sum_thirds = sum(cm.rule_of_thirds for cm in composition_list)
        n = len(composition_list)

        avg_symmetry = sum_sym / n
        avg_center = sum_center / n
        avg_golden = sum_gold / n
        avg_thirds = sum_thirds / n
        avg_score = sum(overall_scores) / n

        print("\n===== 여러 장 이미지 구도 통계 =====")
        print(f" - 평균 Symmetry Score      : {avg_symmetry:.2f}")
        print(f" - 평균 Center Weight      : {avg_center:.2f}")
        print(f" - 평균 Golden Ratio Score : {avg_golden:.2f}")
        print(f" - 3분할 구도 비율         : {(avg_thirds * 100):.1f}%")
        print(f" - 평균 종합 구도 점수     : {avg_score:.2f}")
        print("=================================")

    # =========== (3) 카메라 실시간 분석: 얼굴 + 몸 ===========
    print("\n카메라 초기화 중...")
    cap = cv2.VideoCapture("http://192.249.30.66:4747/video")
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    window_name = "Photo Composition Analyzer (Camera)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        print("카메라 실시간 분석 시작... ESC를 누르면 종료합니다.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패")
                break

            # 1) 전신 분석 (Mediapipe Pose) -> 스켈레톤 + bbox 시각화
            body_analyzer.detect_body(frame)
            body_analyzer.draw_body(frame, draw_bbox=True)

            # 2) 다중 얼굴(FaceMesh) 분석 -> 얼굴 bbox/랜드마크 + 클릭 포커스
            multi_face_analyzer.detect_faces(frame)

            # 선택된 얼굴 ROI만 구도 분석 (단일 얼굴)
            idx = multi_face_analyzer.selected_face_index
            if idx is not None and idx < len(multi_face_analyzer.face_info):
                (xmin, ymin, xmax, ymax) = multi_face_analyzer.face_info[idx]["bbox"]
                h, w = frame.shape[:2]
                xmin = max(0, xmin); ymin = max(0, ymin)
                xmax = min(w, xmax); ymax = min(h, ymax)

                # ROI
                face_roi = frame[ymin:ymax, xmin:xmax]

                # 구도 분석(ROI)
                analyzed_roi, analysis_result = analyzer.analyze_frame(face_roi, image_path="(Selected Face)")

                # ROI 결과를 우측 상단에 표시
                roi_h, roi_w = analyzed_roi.shape[:2]
                if roi_h > 0 and roi_w > 0 and roi_h < h and roi_w < w:
                    frame[0:roi_h, w - roi_w:w] = analyzed_roi

            # 모든 얼굴 시각화
            multi_face_analyzer.draw_faces(frame, draw_landmarks=True)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(10) & 0xFF == 27:
                print("사용자에 의해 프로그램이 종료되었습니다.")
                break

    except Exception as e:
        print(f"오류 발생: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        analyzer.close()
        print("프로그램 종료.")


if __name__ == "__main__":
    main()