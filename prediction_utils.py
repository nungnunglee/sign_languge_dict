import sys
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Generator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import orjson
from PIL import ImageFont, ImageDraw, Image

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_DIR = r"C:\Users\dilab\PycharmProjects\hand_language_test\kwan"

# --- Constants ---
POSE_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), 
    (10, 11), (11, 24), (11, 22), (22, 23), (8, 12), (12, 13), (13, 14), 
    (14, 21), (14, 19), (19, 20), (1, 0), (0, 15), (0, 16), (15, 17), (16, 18)
]
HAND_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
FACE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24), (24, 25), (25, 26),
    (27, 28), (28, 29), (29, 30), (31, 32), (32, 33), (33, 34), (34, 35),
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)
]
KP_COUNTS = {"pose": 25, "face": 70, "hand_left": 21, "hand_right": 21}

# --- Import Transformer Model ---
try:
    # Assuming this file is in the same directory as real_time_predict.py
    # We need to add the parent directory to sys.path to import transformer_xl
    # Adjust this logic if the file structure is different
    BASE_DIR = Path(__file__).parent.absolute()
    if str(BASE_DIR) not in sys.path:
        sys.path.append(str(BASE_DIR))
    from transformer_xl import SignLanguageTransformerXL
except ImportError:
    logger.error("오류: 'transformer_xl' 모듈을 찾을 수 없습니다.")
    # Fallback or re-raise depending on how strict we want to be
    # For now, we'll let it fail when SignLanguagePredictor tries to use it if it's critical
    pass

class OpenPoseWrapper:
    """OpenPose 라이브러리 설정 및 키포인트 추출을 캡슐화하는 클래스"""
    def __init__(self, models_path: str):
        logger.info("OpenPose Wrapper 초기화 중...")
        
        # OpenPose Import Logic
        try:
            # Using the path passed or a default if needed, but here we assume the user provides the correct path
            # or we rely on the environment being set up.
            # However, the original code had specific logic to add DLL directories.
            # We need to replicate that if we want it to work out of the box.
            
            # Since we can't easily know where the build dir is without arguments, 
            # we might need to rely on the caller to set up paths OR assume the standard paths.
            # For now, let's assume the standard paths relative to the models_path or hardcoded as in the original.
            
            # The original code had:
            # OPENPOSE_DIR = "C:/Users/dilab/PycharmProjects/openpose/build/examples/tutorial_api_python"
            # We will try to infer or use the hardcoded one if imports fail.
            
            if sys.platform == "win32":
                # Attempt to import op. If it fails, try to add paths.
                try:
                    import pyopenpose as op
                except ImportError:
                    # Hardcoded fallback from original file
                    OPENPOSE_BUILD_PATH = "C:/Users/dilab/PycharmProjects/openpose/build"
                    OPENPOSE_DIR = f"{OPENPOSE_BUILD_PATH}/examples/tutorial_api_python"
                    
                    sys.path.append(OPENPOSE_DIR + '/../../python/openpose/Release')
                    os.add_dll_directory(OPENPOSE_DIR + '/../../x64/Release')
                    os.add_dll_directory(OPENPOSE_DIR + '/../../bin')
                    import pyopenpose as op
                    
            else:
                sys.path.append('../../python')
                from openpose import pyopenpose as op
                
        except ImportError as e:
            logger.error(f"Error: OpenPose 라이브러리를 찾을 수 없습니다.")
            raise e

        params = {
            "model_folder": models_path,
            "face": True,
            "hand": True,
            "disable_multi_thread": True
        }
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()
        logger.info("OpenPose Wrapper 초기화 완료.")

    def detect(self, frame: np.ndarray) -> Dict[str, list]:
        """프레임에서 키포인트를 감지하고 원본 1D 리스트로 반환합니다."""
        # OpenPose requires pyopenpose to be imported. 
        # Since we imported it inside __init__ (potentially), we need to make sure it's available.
        # A better way is to make 'op' a global or class member, but 'op' is a module.
        # Let's re-import or assume it's in sys.modules.
        
        if 'pyopenpose' in sys.modules:
            op = sys.modules['pyopenpose']
        elif 'openpose' in sys.modules:
            from openpose import pyopenpose as op
        else:
             # This should have been handled in __init__
             import pyopenpose as op

        datum = op.Datum()
        datum.cvInputData = frame
        self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        return {
            "pose": datum.poseKeypoints[0].tolist() if datum.poseKeypoints is not None and datum.poseKeypoints.shape[0] > 0 else [],
            "face": datum.faceKeypoints[0].tolist() if datum.faceKeypoints is not None and datum.faceKeypoints.shape[0] > 0 else [],
            "hand_left": datum.handKeypoints[0][0].tolist() if datum.handKeypoints[0] is not None and datum.handKeypoints[0].shape[0] > 0 else [],
            "hand_right": datum.handKeypoints[1][0].tolist() if datum.handKeypoints[1] is not None and datum.handKeypoints[1].shape[0] > 0 else [],
        }

class SignLanguagePredictor:
    """모델 로딩, 전처리, 상태(mems, buffer) 관리를 캡슐화하는 클래스"""
    
    def __init__(self, config_path, device: torch.device):
        self.device = device
        
        try:
            with open(config_path, "rb") as f:
                self.config = orjson.loads(f.read())
        except FileNotFoundError:
            logger.error(f"설정 파일(hparams.json)을 찾을 수 없습니다: {config_path}")
            raise
            
        self.chunk_size = self.config.get("chunk_size", 10)
        self.class_names = self._load_class_names()
        self.model = self._load_model()
        
        # 상태 변수 초기화
        self.keypoint_buffer: List[np.ndarray] = []
        self.mems: Optional[List[torch.Tensor]] = None
        
        logger.info(f"모델 로딩 완료. (Chunk: {self.chunk_size}, Classes: {len(self.class_names)})")

    def _load_class_names(self) -> List[str]:
        """hparams.json에서 클래스 이름을 로드. 없으면 경고 후 하드코딩된 값 사용."""
        # config_path might be relative or absolute. 
        # The morph_file path in config might be relative to the config file or absolute.
        # We should handle it carefully.
        
        morph_file = self.config.get("morph_file")
        if not os.path.isabs(morph_file):
             # Try to resolve relative to config file if possible, but we don't have config file path here easily unless we store it.
             # In the original code, it just opened it. Let's assume it's absolute or CWD is correct.
             pass

        with open(morph_file, encoding="utf-8") as f:
            allowed_list = sorted(orjson.loads(f.read()).get("allowed_labels"))
            if self.config.get("null_include"):
                allowed_list.insert(0, "null")
            return allowed_list

    def _load_model(self) -> SignLanguageTransformerXL:
        """설정 파일 기반으로 모델 아키텍처를 구성하고 가중치를 로드합니다."""
        model_path = self.config["model_save_path"]
        model_path = os.path.join(MODEL_DIR, model_path)
        model = SignLanguageTransformerXL(
            num_classes=self.config["num_classes"],
            input_features=self.config["input_features"], # 274
            model_dim=self.config["model_dim"],
            n_head=self.config["n_head"],
            n_layers=self.config["n_layers"],
            d_ff=self.config["d_ff"],
            mem_len=self.config.get("mem_len", 0)
        ).to(self.device)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    @staticmethod
    def preprocess(raw_kps: Dict[str, list]) -> np.ndarray:
        """키포인트 딕셔너리를 받아 (274,) 크기의 정규화된 벡터로 변환합니다."""
        
        def reshape_and_get_xy(kps_list: list, points: int) -> np.ndarray:
            if kps_list:
                try:
                    kps_3d = np.array(kps_list, dtype=np.float32).reshape(points, 3)
                    return kps_3d[:, :2]
                except ValueError:
                    return np.zeros((points, 2), dtype=np.float32)
            else:
                return np.zeros((points, 2), dtype=np.float32)

        pose_xy = reshape_and_get_xy(raw_kps["pose"], KP_COUNTS["pose"])
        face_xy = reshape_and_get_xy(raw_kps["face"], KP_COUNTS["face"])
        hand_left_xy = reshape_and_get_xy(raw_kps["hand_left"], KP_COUNTS["hand_left"])
        hand_right_xy = reshape_and_get_xy(raw_kps["hand_right"], KP_COUNTS["hand_right"])

        all_kps_xy = np.concatenate([pose_xy, face_xy, hand_left_xy, hand_right_xy], axis=0)

        mean = np.mean(all_kps_xy, axis=0)
        std = np.std(all_kps_xy, axis=0)
        epsilon = 1e-6 
        std[std < epsilon] = epsilon
        
        standardized_kps = (all_kps_xy - mean) / std
        return standardized_kps.flatten()

    def __call__(self, raw_kps: Dict[str, list]) -> Optional[List[Optional[Tuple[str, float]]]]:
        """
        처리된 키포인트를 입력받아 버퍼에 저장하고,
        버퍼가 가득 차면 추론을 실행하여 모든 프레임에 대한 (단어, 신뢰도) 리스트를 반환합니다.
        'null' 클래스가 예측된 경우 None을 반환합니다.
        
        Returns:
            None: 아직 추론할 때가 아님
            List[Optional[Tuple[str, float]]]: 청크 내 모든 프레임의 예측 결과 
                [(word, conf), ...] 또는 None ('null' 클래스인 경우)
        """
        # 1. 전처리
        processed_kps = self.preprocess(raw_kps)
        
        # 2. 버퍼링
        self.keypoint_buffer.append(processed_kps)
        
        # 3. 추론
        if len(self.keypoint_buffer) == self.chunk_size:
            with torch.no_grad():
                input_tensor = torch.tensor(
                    np.array(self.keypoint_buffer), dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                logits, new_mems = self.model(input_tensor, self.mems)
                # 모든 프레임에 대해 예측 수행 (chunk_size, num_classes)
                all_logits = logits[0]  # (chunk_size, num_classes)

                probs = F.softmax(all_logits, dim=1)  # (chunk_size, num_classes)
                
                # null 클래스 인덱스 찾기
                null_index = None
                try:
                    null_index = self.class_names.index('null')
                except ValueError:
                    pass  # null 클래스가 없으면 무시
                
                # 모든 프레임의 예측 결과를 리스트로 변환
                predictions = []
                for i in range(self.chunk_size):
                    frame_probs = probs[i]  # (num_classes,)
                    
                    # max 값과 인덱스 가져오기
                    max_conf, max_idx = torch.max(frame_probs, dim=0)
                    max_idx = max_idx.item()
                    max_conf = max_conf.item()
                    
                    # null이 max이고 null 클래스가 있는 경우, 다음으로 큰 값 선택
                    if null_index is not None and max_idx == null_index:
                        # null을 제외한 확률에서 max 찾기
                        non_null_probs = frame_probs.clone()
                        non_null_probs[null_index] = -1.0  # null 확률을 매우 낮게 설정
                        second_max_conf, second_max_idx = torch.max(non_null_probs, dim=0)
                        second_max_idx = second_max_idx.item()
                        second_max_conf = second_max_conf.item()
                        
                        # 다음으로 큰 값이 유효한 경우 사용
                        if second_max_conf > 0:
                            pred_index = second_max_idx
                            conf_value = second_max_conf
                        else:
                            # 유효한 값이 없으면 None
                            predictions.append(None)
                            continue
                    else:
                        pred_index = max_idx
                        conf_value = max_conf
                    
                    predicted_word = self.class_names[pred_index]
                    # 'null' 클래스는 제외 (이미 처리했지만 안전을 위해)
                    if predicted_word != 'null':
                        predictions.append((predicted_word, conf_value))
                    else:
                        predictions.append(None)
            
            # 4. 상태 업데이트
            self.mems = new_mems
            self.keypoint_buffer = [] # 버퍼 비우기
            
            return predictions
        
        return None # 아직 추론할 때가 아님

    def get_probability_distributions(self, raw_kps: Dict[str, list]) -> Optional[List[Dict[str, float]]]:
        """
        처리된 키포인트를 입력받아 버퍼에 저장하고,
        버퍼가 가득 차면 추론을 실행하여 모든 프레임에 대한 전체 확률 분포를 반환합니다.
        
        Returns:
            None: 아직 추론할 때가 아님
            List[Dict[str, float]]: 청크 내 모든 프레임의 확률 분포 
                [{"word1": prob1, "word2": prob2, ...}, ...]
        """
        # 1. 전처리
        processed_kps = self.preprocess(raw_kps)
        
        # 2. 버퍼링
        self.keypoint_buffer.append(processed_kps)
        
        # 3. 추론
        if len(self.keypoint_buffer) == self.chunk_size:
            with torch.no_grad():
                input_tensor = torch.tensor(
                    np.array(self.keypoint_buffer), dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                logits, new_mems = self.model(input_tensor, self.mems)
                # 모든 프레임에 대해 예측 수행 (chunk_size, num_classes)
                all_logits = logits[0]  # (chunk_size, num_classes)

                probs = F.softmax(all_logits, dim=1)  # (chunk_size, num_classes)
                
                # 모든 프레임의 확률 분포를 딕셔너리 리스트로 변환
                probability_distributions = []
                for i in range(self.chunk_size):
                    frame_probs = probs[i]  # (num_classes,)
                    
                    # 각 단어에 대한 확률을 딕셔너리로 변환
                    frame_dist = {}
                    for j, class_name in enumerate(self.class_names):
                        prob_value = frame_probs[j].item()
                        frame_dist[class_name] = prob_value
                    
                    probability_distributions.append(frame_dist)
            
            # 4. 상태 업데이트
            self.mems = new_mems
            self.keypoint_buffer = [] # 버퍼 비우기
            
            return probability_distributions
        
        return None # 아직 추론할 때가 아님


class VideoProcessor:
    """웹캠 또는 비디오 파일 입력을 관리하고 10FPS 샘플링을 수행합니다."""
    
    def __init__(self, video_path: Optional[str], target_fps: int = 10):
        self.target_fps = target_fps
        self.target_delay_ms = 1000 // target_fps
        self.is_webcam = (video_path is None)
        self.frame_count = 0

        if self.is_webcam:
            logger.info(f"웹캠 모드로 실행 (Target FPS: {target_fps})")
            self.cap = cv2.VideoCapture(1) # 1번 카메라
            if not self.cap.isOpened():
                raise IOError("웹캠(1번)을 열 수 없습니다. 0번으로 다시 시도해보세요.")
            self.frame_skip_interval = 1
        else:
            logger.info(f"비디오 파일 모드로 실행: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise IOError(f"비디오 파일을 열 수 없습니다: {video_path}")
            
            original_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if original_fps <= 0:
                logger.warning("비디오의 원본 FPS를 읽을 수 없습니다. 30.0으로 가정합니다.")
                original_fps = 30.0
            
            self.frame_skip_interval = max(1, int(round(original_fps / target_fps)))
            logger.info(f"원본 FPS: {original_fps:.2f} -> {self.frame_skip_interval}프레임당 1프레임 처리")

    def __iter__(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        """비디오/웹캠 프레임을 10FPS에 맞춰 반환하는 제너레이터"""
        while self.cap.isOpened():
            start_time_ticks = cv2.getTickCount()

            ret, frame = self.cap.read()
            if not ret:
                break # 비디오 끝 또는 오류

            should_process = (self.frame_count % self.frame_skip_interval == 0)
            
            if should_process:
                yield frame, self.frame_count
            
            self.frame_count += 1
            
            # FPS 제어
            wait_key_time = 1
            if self.is_webcam:
                # 웹캠은 실시간이므로 강제로 10FPS에 맞게 대기
                end_time_ticks = cv2.getTickCount()
                elapsed_time_ms = (end_time_ticks - start_time_ticks) / cv2.getTickFrequency() * 1000
                wait_time_ms = self.target_delay_ms - elapsed_time_ms
                wait_key_time = max(1, int(wait_time_ms))
            
            if cv2.waitKey(wait_key_time) & 0xFF == ord('q'):
                logger.info("'q' 키 입력으로 종료합니다.")
                break
                
    def release(self):
        if self.cap.isOpened():
            self.cap.release()
            logger.info("비디오 캡처 리소스 해제됨.")

def put_text_on_image(cv_image, text, position, font_path, font_size, font_color_bgr):
    """
    OpenCV 이미지(Numpy 배열)에 한글 텍스트를 입힙니다.
    """
    
    # 1. OpenCV 이미지(BGR)를 PIL 이미지(RGB)로 변환
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # 2. Pillow의 ImageDraw 객체 생성
    draw = ImageDraw.Draw(pil_image)
    
    # 3. 폰트 로드
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # print(f"Error: 폰트 파일을 찾을 수 없거나 읽을 수 없습니다. 경로: {font_path}")
        # 대체 폰트 사용 (Pillow 기본 폰트 - 한글 지원 안 됨)
        font = ImageFont.load_default()

    # 4. 텍스트 그리기
    font_color_rgb = (font_color_bgr[2], font_color_bgr[1], font_color_bgr[0])
    draw.text(position, text, font=font, fill=font_color_rgb)
    
    # 5. PIL 이미지(RGB)를 다시 OpenCV 이미지(BGR)로 변환
    result_image_rgb = np.array(pil_image)
    result_image_bgr = cv2.cvtColor(result_image_rgb, cv2.COLOR_RGB2BGR)
    
    return result_image_bgr

class Visualizer:
    """시각화 관련 유틸리티 함수 모음"""
    
    @staticmethod
    def display_prediction(image: np.ndarray, text: str):
        image = put_text_on_image(cv_image=image, 
                          text=text, 
                          position=(10, 10), 
                          font_path='C:/Windows/Fonts/malgun.ttf', 
                          font_size=40, 
                          font_color_bgr=(255, 255, 255))
        return image

    @staticmethod
    def draw_skeleton_part(background: np.ndarray, keypoints: List[float], pairs: List[Tuple[int, int]], color: Tuple[int, int, int]):
        if not keypoints:
            return
        
        try:
            kps = np.array(keypoints).reshape(-1, 3)
        except ValueError:
            logger.warning("스켈레톤 그리기 실패: Reshape 오류", exc_info=False)
            return
            
        for i, j in pairs:
            if i < len(kps) and j < len(kps):
                start_kp = kps[i]
                end_kp = kps[j]
                if start_kp[2] > 0 and end_kp[2] > 0:
                    start_point = (int(start_kp[0]), int(start_kp[1]))
                    end_point = (int(end_kp[0]), int(end_kp[1]))
                    cv2.line(background, start_point, end_point, color, 2)
        for i in range(len(kps)):
            if kps[i, 2] > 0:
                cv2.circle(background, (int(kps[i, 0]), int(kps[i, 1])), 5, color, -1)

    @staticmethod
    def draw_all_skeletons(frame: np.ndarray, raw_kps: Dict[str, list]):
        """모든 부위의 스켈레톤을 프레임에 그립니다."""
        Visualizer.draw_skeleton_part(frame, raw_kps["pose"], POSE_PAIRS, color=(255, 0, 0))
        Visualizer.draw_skeleton_part(frame, raw_kps["face"], FACE_PAIRS, color=(0, 255, 0))
        Visualizer.draw_skeleton_part(frame, raw_kps["hand_left"], HAND_PAIRS, color=(0, 0, 255))
        Visualizer.draw_skeleton_part(frame, raw_kps["hand_right"], HAND_PAIRS, color=(0, 255, 255))
