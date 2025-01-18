# utils/image_utils.py
import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple

class ImageProcessor:
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray:
        """이미지 로드 및 기본 전처리"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image

    @staticmethod
    def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """이미지 크기 조정"""
        height, width = image.shape[:2]
        
        # 이미지가 너무 크면 리사이즈
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size)
        
        return image

    @staticmethod
    def get_image_files(directory: Union[str, Path]) -> List[Path]:
        """디렉토리에서 이미지 파일 목록 가져오기"""
        directory = Path(directory)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)

    @staticmethod
    def extract_exif(image_path: Union[str, Path]) -> dict:
        """EXIF 데이터 추출"""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            image = Image.open(str(image_path))
            exif = {}
            
            if hasattr(image, '_getexif'):
                exif_data = image._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif[tag] = value
            
            return exif
        except Exception as e:
            print(f"Failed to extract EXIF data: {e}")
            return {}

    @staticmethod
    def save_image(image: np.ndarray, save_path: Union[str, Path], quality: int = 95):
        """이미지 저장"""
        try:
            cv2.imwrite(str(save_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return True
        except Exception as e:
            print(f"Failed to save image: {e}")
            return False

    @staticmethod
    def create_comparison_image(original: np.ndarray, analyzed: np.ndarray) -> np.ndarray:
        """원본과 분석된 이미지를 나란히 표시"""
        # 두 이미지의 높이를 맞춤
        h1, w1 = original.shape[:2]
        h2, w2 = analyzed.shape[:2]
        
        target_height = min(h1, h2)
        scale1 = target_height / h1
        scale2 = target_height / h2
        
        resized_original = cv2.resize(original, (int(w1 * scale1), int(h1 * scale1)))
        resized_analyzed = cv2.resize(analyzed, (int(w2 * scale2), int(h2 * scale2)))
        
        # 이미지 가로로 합치기
        return np.hstack((resized_original, resized_analyzed))