# main.py
from analyzer.multi_face_analyzer import MultiFaceAnalyzer  
from analyzer.body_analyzer import BodyAnalyzer           

import os
import sys
import cv2
import logging
import warnings
import tensorflow as tf
from pathlib import Path

# GUI 관련 환경변수 제거
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 중 오류 발생: {e}")

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# ====== 전역 설정 ======
multi_face_analyzer = MultiFaceAnalyzer()  
body_analyzer = None        

def main():
    global body_analyzer

    # GPU 확인
    print("\nGPU 확인:")
    print("TensorFlow:", tf.config.list_physical_devices('GPU'))
    print("CUDA 사용 가능:", tf.test.is_built_with_cuda())
    print("GPU 사용 가능:", tf.test.is_built_with_gpu_support())

    # 출력 디렉토리 생성
    output_dir = Path("output_frames")
    output_dir.mkdir(exist_ok=True)

    # 전신 분석기 초기화
    body_analyzer = BodyAnalyzer(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 카메라 초기화
    print("\n카메라 초기화 중...")
    cap = cv2.VideoCapture(1)  # /dev/video1 사용
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 프로그램을 종료합니다.")
        sys.exit(1)

    frame_count = 0
    try:
        print("카메라 실시간 분석 시작... Ctrl+C로 종료하세요.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패")
                break

            # 1) 전신 분석 -> 스켈레톤 + bbox 시각화
            body_analyzer.detect_body(frame)
            body_analyzer.draw_body(frame, draw_bbox=True)

            # 2) 다중 얼굴 분석 -> 얼굴 bbox/랜드마크
            multi_face_analyzer.detect_faces(frame)
            multi_face_analyzer.draw_faces(frame, draw_landmarks=True)

            # 프레임 저장
            if frame_count % 30 == 0:  # 30프레임마다 저장
                output_path = output_dir / f"frame_{frame_count}.jpg"
                cv2.imwrite(str(output_path), frame)
                print(f"프레임 저장됨: {output_path}")
            
            frame_count += 1

    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 종료되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        cap.release()
        print("프로그램 종료.")

if __name__ == "__main__":
    main()