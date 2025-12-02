import os
import sys
import time
import logging
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Dict, Any
import cv2
import torch
import numpy as np

# Import from prediction_utils
try:
    from prediction_utils import (
        OpenPoseWrapper,
        SignLanguagePredictor,
        VideoProcessor,
        Visualizer
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from prediction_utils import (
        OpenPoseWrapper,
        SignLanguagePredictor,
        VideoProcessor,
        Visualizer
    )

# --- Constants & Defaults ---
DEFAULT_MODEL_CONFIG = r"C:\Users\dilab\PycharmProjects\hand_language_test\kwan\runs\transformer_xl\transformer_xl_20251202-140124\hparams.json"
DEFAULT_OPENPOSE_MODELS = r"C:\Users\dilab\PycharmProjects\hand_language_test\kwan\openpose\models"

# --- Data Classes ---

@dataclass
class FrameKeypoints:
    """한 프레임의 키포인트 정보 (OpenPose 형식)"""
    frame_index: int
    timestamp: float
    pose: List[List[float]]  # [[x, y, confidence], ...]
    face: List[List[float]]
    hand_left: List[List[float]]
    hand_right: List[List[float]]

@dataclass
class VideoInfo:
    """동영상 메타정보"""
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    codec: str

@dataclass
class TranslationResult:
    """번역 결과를 담는 데이터 클래스"""
    success: bool
    word: str
    keypoints: List[FrameKeypoints]
    video_info: VideoInfo
    annotated_video_path: str
    processing_time: float
    error: Optional[str] = None

# --- Exception Classes ---

class VideoProcessingError(Exception):
    pass

class ModelInferenceError(Exception):
    pass

# --- Main Interface Class ---

class SignLanguageTranslator:
    """
    수어 번역을 위한 메인 클래스.
    모델을 한 번만 로드하고 재사용하여 추론 성능을 최적화합니다.
    """
    def __init__(
        self, 
        model_config_path: str = DEFAULT_MODEL_CONFIG,
        openpose_models_path: str = DEFAULT_OPENPOSE_MODELS,
        device: Optional[str] = None
    ):
        """
        모델 초기화
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_config_path = model_config_path
        self.openpose_models_path = openpose_models_path
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.logger.info(f"Initializing SignLanguageTranslator on {self.device}...")
        
        # 모델 로드
        try:
            self.predictor = SignLanguagePredictor(
                config_path=self.model_config_path,
                device=self.device
            )
            self.openpose = OpenPoseWrapper(models_path=self.openpose_models_path)
            self.logger.info("All models loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise e

    def translate_sign_language(
        self,
        video_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> TranslationResult:
        """
        동영상을 분석하여 수어를 번역하고 키포인트를 추출합니다.
        """
        start_time = time.time()
        
        # 1. 입력 검증 및 준비
        if not os.path.exists(video_path):
            return TranslationResult(
                success=False, word="", keypoints=[], video_info=None, 
                annotated_video_path="", processing_time=0.0, 
                error=f"File not found: {video_path}"
            )
            
        os.makedirs(output_dir, exist_ok=True)
        annotated_video_path = os.path.join(output_dir, "annotated.mp4")
        
        try:
            if progress_callback:
                progress_callback(0, "Starting processing...")
                
            # 2. 비디오 프로세서 초기화
            # We use VideoProcessor for 10FPS sampling as required by the model
            video_processor = VideoProcessor(video_path=video_path, target_fps=10)
            
            # Get video info from the capture object inside processor
            cap = video_processor.cap
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames_original / original_fps if original_fps > 0 else 0
            
            # VideoWriter Setup
            # Resizing to 50% as in the original code
            final_width = int(original_width * 0.5)
            final_height = int(original_height * 0.5)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(annotated_video_path, fourcc, 10, (final_width, final_height))
            
            video_info = VideoInfo(
                width=original_width,
                height=original_height,
                fps=original_fps,
                duration=duration,
                total_frames=total_frames_original,
                codec="h264" # Assuming mp4v/h264
            )
            
            all_keypoints: List[FrameKeypoints] = []
            predictions: List[tuple] = [] # (word, confidence)
            
            # 3. 처리 루프
            processed_frames = 0
            # Estimate total processed frames (approximate since we skip frames)
            estimated_processed_frames = int(duration * 10) 
            
            # 상태 초기화 (새로운 비디오 처리 시 버퍼 등 초기화 필요)
            self.predictor.keypoint_buffer = []
            self.predictor.mems = None
            
            for frame, frame_idx in video_processor:
                # Progress update
                if progress_callback and estimated_processed_frames > 0:
                    # Map 0-100% range to frame progress
                    current_percent = int((processed_frames / estimated_processed_frames) * 100)
                    current_percent = min(99, current_percent)
                    progress_callback(current_percent, f"Processing frame {frame_idx}...")

                # 3-1. 키포인트 추출
                raw_keypoints = self.openpose.detect(frame)
                
                # Convert to FrameKeypoints
                # Helper to convert list of lists
                def to_list_of_lists(kps, count):
                    if not kps:
                        return []
                    try:
                        arr = np.array(kps).reshape(-1, 3)
                        return arr.tolist()
                    except:
                        return []

                frame_kp = FrameKeypoints(
                    frame_index=frame_idx,
                    timestamp=frame_idx / original_fps if original_fps > 0 else 0,
                    pose=to_list_of_lists(raw_keypoints["pose"], 25),
                    face=to_list_of_lists(raw_keypoints["face"], 70),
                    hand_left=to_list_of_lists(raw_keypoints["hand_left"], 21),
                    hand_right=to_list_of_lists(raw_keypoints["hand_right"], 21)
                )
                all_keypoints.append(frame_kp)
                
                # 3-2. 추론
                result = self.predictor(raw_keypoints)
                current_text = "Analyzing..."
                
                if result:
                    word, conf = result
                    predictions.append((word, conf))
                    current_text = f"{word} ({conf*100:.1f}%)"
                
                # 3-3. 시각화 및 저장
                frame = Visualizer.display_prediction(frame, current_text)
                Visualizer.draw_all_skeletons(frame, raw_keypoints)
                frame_resized = cv2.resize(frame, (final_width, final_height), interpolation=cv2.INTER_LINEAR)
                
                if writer.isOpened():
                    writer.write(frame_resized)
                    
                processed_frames += 1

            # 4. 정리 및 결과 집계
            video_processor.release()
            if writer.isOpened():
                writer.release()
                
            if progress_callback:
                progress_callback(100, "Finalizing results...")
                
            # Select best prediction
            final_word = ""
            if predictions:
                # Sort by confidence descending
                predictions.sort(key=lambda x: x[1], reverse=True)
                final_word = predictions[0][0]
                
            processing_time = time.time() - start_time
            
            return TranslationResult(
                success=True,
                word=final_word,
                keypoints=all_keypoints,
                video_info=video_info,
                annotated_video_path=annotated_video_path,
                processing_time=processing_time
            )

        except Exception as e:
            self.logger.error("Translation failed", exc_info=True)
            return TranslationResult(
                success=False,
                word="",
                keypoints=[],
                video_info=None, # type: ignore
                annotated_video_path="",
                processing_time=time.time() - start_time,
                error=str(e)
            )

if __name__ == "__main__":
    # Simple test if run directly
    test_video = "C:/Users/dilab/Desktop/hand_sen10/sung/1.mp4"
    output = "test_output"
    
    def print_progress(p, m):
        print(f"[{p}%] {m}")
    
    print("Initializing Translator...")
    translator = SignLanguageTranslator()
    
    print(f"Processing {test_video}...")
    result = translator.translate_sign_language(test_video, output, print_progress)
    
    print(f"Result: Success={result.success}, Word={result.word}")
    if result.success:
        print(f"Saved to {result.annotated_video_path}")
