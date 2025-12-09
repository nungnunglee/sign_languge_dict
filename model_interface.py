import os
import sys
import time
import logging
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Dict, Any, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
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
DEFAULT_MODEL_CONFIG = r"C:\Users\dilab\PycharmProjects\hand_language_test\kwan\runs\transformer_xl\transformer_xl_20251202-140124\hparams.json" # null 포함
# DEFAULT_MODEL_CONFIG = r"C:\Users\dilab\PycharmProjects\hand_language_test\kwan\runs\transformer_xl\transformer_xl_20251202-174419\hparams.json" # null 미포함
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

# --- Word Selection Algorithms ---

class WordSelectionAlgorithm(ABC):
    """최종 단어 선택 알고리즘의 베이스 클래스"""
    
    @abstractmethod
    def select_word(self, predictions: List[Tuple[str, float]], conf_threshold: float) -> str:
        """
        예측 결과 리스트에서 최종 단어를 선택
        
        Args:
            predictions: (word, confidence) 튜플 리스트
            conf_threshold: confidence threshold
            
        Returns:
            선택된 최종 단어 (없으면 빈 문자열)
        """
        pass


class GroupingAlgorithm(WordSelectionAlgorithm):
    """기본 그룹화 알고리즘 (기존 model_interface.py 방식)"""
    
    def select_word(self, predictions: List[Tuple[str, float]], conf_threshold: float) -> str:
        """연속된 같은 단어들을 그룹화하여 마지막 그룹의 단어 선택"""
        if not predictions:
            return ""
        
        # 1. confidence threshold 적용 및 null 제외
        filtered_predictions = [
            (word, conf) for word, conf in predictions 
            if word is not None and word != 'null' and conf >= conf_threshold
        ]
        
        if not filtered_predictions:
            return ""
        
        # 2. 연속된 같은 단어들을 그룹화
        groups = []
        current_group = [filtered_predictions[0]]
        
        for i in range(1, len(filtered_predictions)):
            current_word, _ = filtered_predictions[i]
            prev_word, _ = filtered_predictions[i-1]
            
            if current_word == prev_word:
                # 같은 단어면 현재 그룹에 추가
                current_group.append(filtered_predictions[i])
            else:
                # 다른 단어면 현재 그룹을 저장하고 새 그룹 시작
                groups.append(current_group)
                current_group = [filtered_predictions[i]]
        
        # 마지막 그룹 추가
        groups.append(current_group)
        
        # 3. 마지막 그룹의 단어 선택
        if groups:
            last_group = groups[-1]
            final_word = last_group[0][0]  # 같은 단어이므로 첫 번째 사용
            return final_word
        
        return ""


class SegmentScoringAlgorithm(WordSelectionAlgorithm):
    """세그먼트 기반 스코어링 알고리즘"""
    
    def __init__(
        self,
        w_conf: float = 1.0,
        w_dur: float = 0.5,
        w_last: float = 2.0,
        min_duration: int = 5,
        smoothing_window: int = 5,
        smoothing_method: str = 'majority'
    ):
        """
        Args:
            w_conf: confidence 가중치
            w_dur: duration 가중치
            w_last: 마지막 세그먼트 보너스 가중치
            min_duration: 최소 세그먼트 지속 시간 (프레임 수)
            smoothing_window: 스무딩 윈도우 크기
            smoothing_method: 'majority' (Majority Voting) 또는 'average' (Moving Average)
        """
        self.w_conf = w_conf
        self.w_dur = w_dur
        self.w_last = w_last
        self.min_duration = min_duration
        self.smoothing_window = smoothing_window
        self.smoothing_method = smoothing_method
    
    def _get_max_word_with_null_skip(self, probabilities: Dict[str, float], null_index: Optional[str] = 'null') -> Tuple[str, float]:
        """확률 분포에서 null이 max인 경우 다음으로 큰 값 선택"""
        if null_index and null_index in probabilities:
            non_null_probs = {k: v for k, v in probabilities.items() if k != null_index}
            if non_null_probs:
                max_word = max(non_null_probs.items(), key=lambda x: x[1])
                return max_word
        max_word = max(probabilities.items(), key=lambda x: x[1])
        return max_word
    
    def select_word(self, predictions: List[Tuple[str, float]], conf_threshold: float) -> str:
        """세그먼트 기반 스코어링으로 최종 단어 선택"""
        if not predictions:
            return ""
        
        # predictions를 프레임별 예측으로 변환 (단순화: predictions가 이미 프레임별 결과라고 가정)
        # 실제로는 predictions를 그대로 사용하되, None 처리 추가
        frame_predictions = []
        for word, conf in predictions:
            if word is not None and word != 'null' and conf >= conf_threshold:
                frame_predictions.append((word, conf))
            else:
                frame_predictions.append((None, 0.0))
        
        if not any(pred[0] for pred in frame_predictions):
            return ""
        
        # Smoothing (전처리)
        smoothed_predictions = self._apply_smoothing(frame_predictions)
        
        if not any(pred[0] for pred in smoothed_predictions):
            return ""
        
        # Segmentation (연속된 같은 예측 구간 그룹화)
        segments = self._create_segments(smoothed_predictions)
        
        if not segments:
            return ""
        
        # Filtering (최소 지속 시간보다 짧은 구간 제거)
        valid_segments = [seg for seg in segments if seg['duration'] >= self.min_duration]
        
        if not valid_segments:
            return ""
        
        # Scoring
        total_frames = len(smoothed_predictions)
        segment_scores = {}
        
        for i, seg in enumerate(valid_segments):
            word = seg['word']
            avg_conf = seg['avg_conf']
            duration = seg['duration']
            is_last = (i == len(valid_segments) - 1)
            
            # 정규화된 duration (전체 길이 대비)
            normalized_duration = duration / total_frames if total_frames > 0 else 0
            
            # 점수 계산
            score = (
                (avg_conf * self.w_conf) +
                (normalized_duration * self.w_dur) +
                (1.0 if is_last else 0.0) * self.w_last
            )
            
            # 같은 단어가 여러 세그먼트에 있으면 최고 점수 사용
            if word not in segment_scores or score > segment_scores[word]:
                segment_scores[word] = score
        
        if not segment_scores:
            return ""
        
        # 최고 점수 단어 선택
        final_word = max(segment_scores.items(), key=lambda x: x[1])[0]
        return final_word
    
    def _apply_smoothing(self, frame_predictions: List[Tuple[Optional[str], float]]) -> List[Tuple[Optional[str], float]]:
        """스무딩 적용"""
        smoothed_predictions = []
        
        if self.smoothing_method == 'majority':
            # Majority Voting
            for i in range(len(frame_predictions)):
                start_idx = max(0, i - self.smoothing_window // 2)
                end_idx = min(len(frame_predictions), i + self.smoothing_window // 2 + 1)
                window = frame_predictions[start_idx:end_idx]
                
                # 유효한 예측만 필터링
                valid_predictions = [(w, c) for w, c in window if w is not None]
                
                if valid_predictions:
                    # 가장 많이 나타난 단어 선택
                    word_counts = {}
                    word_confs = {}
                    for w, c in valid_predictions:
                        word_counts[w] = word_counts.get(w, 0) + 1
                        if w not in word_confs:
                            word_confs[w] = []
                        word_confs[w].append(c)
                    
                    # 가장 많이 나타난 단어 선택 (동점이면 평균 confidence가 높은 것)
                    max_count = max(word_counts.values())
                    candidates = [w for w, count in word_counts.items() if count == max_count]
                    
                    # 평균 confidence가 가장 높은 단어 선택
                    best_word = max(candidates, key=lambda w: sum(word_confs[w]) / len(word_confs[w]))
                    avg_conf = sum(word_confs[best_word]) / len(word_confs[best_word])
                    smoothed_predictions.append((best_word, avg_conf))
                else:
                    smoothed_predictions.append((None, 0.0))
        
        elif self.smoothing_method == 'average':
            # Moving Average (confidence만 평균화, 단어는 그대로)
            for i in range(len(frame_predictions)):
                start_idx = max(0, i - self.smoothing_window // 2)
                end_idx = min(len(frame_predictions), i + self.smoothing_window // 2 + 1)
                window = frame_predictions[start_idx:end_idx]
                
                # 현재 프레임의 단어 사용
                current_word, current_conf = frame_predictions[i]
                
                if current_word is not None:
                    # 윈도우 내 같은 단어들의 평균 confidence 계산
                    same_word_confs = [c for w, c in window if w == current_word]
                    if same_word_confs:
                        avg_conf = sum(same_word_confs) / len(same_word_confs)
                        smoothed_predictions.append((current_word, avg_conf))
                    else:
                        smoothed_predictions.append((current_word, current_conf))
                else:
                    smoothed_predictions.append((None, 0.0))
        else:
            # smoothing 없음
            smoothed_predictions = frame_predictions
        
        return smoothed_predictions
    
    def _create_segments(self, smoothed_predictions: List[Tuple[Optional[str], float]]) -> List[Dict[str, Any]]:
        """세그먼트 생성"""
        segments = []
        current_segment_word = None
        current_segment_confs = []
        current_segment_start = None
        
        for i in range(len(smoothed_predictions)):
            word, conf = smoothed_predictions[i]
            
            if word is not None:
                if word == current_segment_word:
                    # 같은 단어면 현재 세그먼트에 추가
                    current_segment_confs.append(conf)
                else:
                    # 다른 단어면 이전 세그먼트 저장
                    if current_segment_word is not None and current_segment_start is not None:
                        segments.append({
                            'word': current_segment_word,
                            'duration': len(current_segment_confs),
                            'avg_conf': sum(current_segment_confs) / len(current_segment_confs),
                            'start_idx': current_segment_start
                        })
                    
                    # 새 세그먼트 시작
                    current_segment_word = word
                    current_segment_confs = [conf]
                    current_segment_start = i
            else:
                # None이면 현재 세그먼트 종료 (있는 경우)
                if current_segment_word is not None and current_segment_start is not None:
                    segments.append({
                        'word': current_segment_word,
                        'duration': len(current_segment_confs),
                        'avg_conf': sum(current_segment_confs) / len(current_segment_confs),
                        'start_idx': current_segment_start
                    })
                    current_segment_word = None
                    current_segment_confs = []
                    current_segment_start = None
        
        # 마지막 세그먼트 추가
        if current_segment_word is not None and current_segment_start is not None:
            segments.append({
                'word': current_segment_word,
                'duration': len(current_segment_confs),
                'avg_conf': sum(current_segment_confs) / len(current_segment_confs),
                'start_idx': current_segment_start
            })
        
        return segments



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
        device: Optional[str] = None,
        word_selection_algorithm: Optional[WordSelectionAlgorithm] = None
    ):
        """
        모델 초기화
        
        Args:
            model_config_path: 모델 설정 파일 경로
            openpose_models_path: OpenPose 모델 경로
            device: 사용할 디바이스 ('cuda' 또는 'cpu')
            word_selection_algorithm: 최종 단어 선택 알고리즘 (None이면 기본 세그먼트 알고리즘 사용)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_config_path = model_config_path
        self.openpose_models_path = openpose_models_path
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.logger.info(f"Initializing SignLanguageTranslator on {self.device}...")
        
        # 최종 단어 선택 알고리즘 설정
        if word_selection_algorithm is not None:
            self.word_selection_algorithm = word_selection_algorithm
        else:
            # 기본값: 세그먼트 기반 스코어링 알고리즘 (튜닝된 최적 가중치)
            self.word_selection_algorithm = SegmentScoringAlgorithm(
                w_conf=5.1,
                w_dur=7.6,
                w_last=0.0,
                min_duration=9
            )
            algorithm_name = self.word_selection_algorithm.__class__.__name__
            self.logger.info(f"Using word selection algorithm: {algorithm_name}")
        
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
        progress_callback: Optional[Callable[[int, str], None]] = None,
        conf_threshold: float = 0.01
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
            # H.264 코덱 사용 (브라우저 호환성을 위해)
            # 'avc1' 또는 'H264' fourcc 사용 시도
            # Windows에서는 FFmpeg 백엔드가 필요할 수 있음
            try:
                # H.264 코덱 시도 (브라우저 호환성 최우선)
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264의 MP4 컨테이너용 fourcc
                writer = cv2.VideoWriter(annotated_video_path, fourcc, 10, (final_width, final_height))
                if not writer.isOpened():
                    raise ValueError("avc1 코덱으로 writer를 열 수 없음")
                self.logger.info("비디오 코덱 'avc1' (H.264) 사용")
            except Exception as e:
                self.logger.warning(f"avc1 코덱 실패, H264 시도: {e}")
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'H264')
                    writer = cv2.VideoWriter(annotated_video_path, fourcc, 10, (final_width, final_height))
                    if not writer.isOpened():
                        raise ValueError("H264 코덱으로 writer를 열 수 없음")
                    self.logger.info("비디오 코덱 'H264' 사용")
                except Exception as e2:
                    self.logger.warning(f"H264 코덱 실패, mp4v로 폴백: {e2}")
                    # 폴백: mp4v (브라우저 호환성 낮음, 나중에 FFmpeg로 변환 필요)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(annotated_video_path, fourcc, 10, (final_width, final_height))
                    self.logger.warning("비디오 코덱 'mp4v' 사용 (브라우저 호환성 낮음)")
            
            video_info = VideoInfo(
                width=original_width,
                height=original_height,
                fps=original_fps,
                duration=duration,
                total_frames=total_frames_original,
                codec="h264" # Assuming mp4v/h264
            )
            
            all_keypoints: List[FrameKeypoints] = []
            predictions: List[tuple] = [] # (word, confidence) - 모든 프레임의 예측
            frame_predictions: Dict[int, tuple] = {}  # frame_idx -> (word, conf) 매핑
            frames_data: List[tuple] = []  # (frame, frame_idx, raw_keypoints) 저장
            
            # 3-1. 첫 번째 단계: 키포인트 추출 및 추론 (비디오 작성 안 함)
            processed_frames = 0
            estimated_processed_frames = int(duration * 10) 
            
            # 상태 초기화
            self.predictor.keypoint_buffer = []
            self.predictor.mems = None
            frame_buffer: List[int] = []  # 현재 청크의 프레임 인덱스 추적
            
            if progress_callback:
                progress_callback(0, "Extracting keypoints and running inference...")
            
            for frame, frame_idx in video_processor:
                # Progress update
                if progress_callback and estimated_processed_frames > 0:
                    # 첫 번째 단계는 95%까지
                    current_percent = int((processed_frames / estimated_processed_frames) * 95)
                    current_percent = min(94, current_percent)
                    progress_callback(current_percent, f"Processing frame {frame_idx}...")

                # 키포인트 추출
                raw_keypoints = self.openpose.detect(frame)
                
                # Convert to FrameKeypoints
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
                
                # 프레임 데이터 저장 (나중에 비디오 생성 시 사용)
                frames_data.append((frame.copy(), frame_idx, raw_keypoints))
                
                # 프레임 인덱스를 버퍼에 추가
                frame_buffer.append(frame_idx)
                
                # 추론
                result = self.predictor(raw_keypoints)
                
                # 예측 결과가 있으면 (청크가 가득 찬 경우)
                if result:
                    # result는 청크 내 모든 프레임의 예측 리스트: [(word, conf), ...] 또는 None
                    # 프레임 버퍼와 매핑하여 각 프레임의 예측 저장 ('null' 제외)
                    for i, pred_result in enumerate(result):
                        if i < len(frame_buffer):
                            frame_idx_for_pred = frame_buffer[i]
                            # 'null' 클래스는 제외 (None인 경우 스킵)
                            if pred_result is not None:
                                word, conf = pred_result
                                frame_predictions[frame_idx_for_pred] = (word, conf)
                                predictions.append((word, conf))
                    # 버퍼 초기화 (다음 청크를 위해)
                    frame_buffer = []
                    
                processed_frames += 1
            
            # 마지막 청크 처리 (완전히 채워지지 않은 경우)
            if len(self.predictor.keypoint_buffer) > 0:
                # 남은 프레임이 있으면 마지막 예측 결과 사용 (또는 패딩)
                # 간단하게 마지막 예측을 나머지 프레임에도 적용
                if frame_predictions:
                    # 가장 최근 예측 결과를 사용
                    last_prediction = list(frame_predictions.values())[-1] if frame_predictions else None
                    if last_prediction:
                        for remaining_idx in frame_buffer:
                            frame_predictions[remaining_idx] = last_prediction
                            predictions.append(last_prediction)
            
            # 3-2. 두 번째 단계: 예측 결과를 사용하여 비디오 생성
            if progress_callback:
                progress_callback(95, "Generating annotated video...")
            
            for frame_idx_in_data, (frame, frame_idx, raw_keypoints) in enumerate(frames_data):
                # Progress update
                if progress_callback and len(frames_data) > 0:
                    # 두 번째 단계는 95-100%
                    current_percent = 95 + int((frame_idx_in_data / len(frames_data)) * 5)
                    current_percent = min(99, current_percent)
                    progress_callback(current_percent, f"Rendering frame {frame_idx}...")
                
                # 현재 프레임의 예측 결과 사용
                if frame_idx in frame_predictions:
                    word, conf = frame_predictions[frame_idx]
                    current_text = f"{word} ({conf*100:.1f}%)"
                else:
                    current_text = "Analyzing..."
                
                # 시각화 및 저장
                frame = Visualizer.display_prediction(frame, current_text)
                Visualizer.draw_all_skeletons(frame, raw_keypoints)
                frame_resized = cv2.resize(frame, (final_width, final_height), interpolation=cv2.INTER_LINEAR)
                
                if writer.isOpened():
                    writer.write(frame_resized)

            # 4. 정리 및 결과 집계
            video_processor.release()
            if writer.isOpened():
                writer.release()
                
            if progress_callback:
                progress_callback(100, "Finalizing results...")
                
            # Select final word: 설정된 알고리즘 사용
            final_word = self.word_selection_algorithm.select_word(predictions, conf_threshold)
                
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
