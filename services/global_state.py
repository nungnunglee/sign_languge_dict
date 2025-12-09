import os
import logging
from typing import Dict, Any, Optional
from model_interface import SignLanguageTranslator

# 로거 설정
logger = logging.getLogger(__name__)

# --- 1. Path Configuration ---

# 프로젝트 루트 경로 (services 폴더의 상위 폴더)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 사용자 업로드 영상 저장소
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# 수어 데이터셋 저장소 (Database)
# global_state를 경로의 'Single Source of Truth'로 만들기 위해 JSON_DIR도 여기서 정의합니다.
VIDEO_DIR = os.path.join(BASE_DIR, 'database', 'video')
JSON_DIR = os.path.join(BASE_DIR, 'database', 'json')

# 필수 폴더 자동 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 2. Global Runtime State ---

# 작업 상태 저장소 (Task ID -> Status Dict)
# Type: Dict[task_id, task_info]
TASK_STATUS: Dict[str, Dict[str, Any]] = {}

# --- 3. Model Initialization ---

# AI 모델을 전역에서 1회 로드 (Singleton Pattern)
# 앱 시작 시 모델이 메모리에 로드됩니다.
try:
    translator = SignLanguageTranslator()
except Exception as e:
    logger.error(f"Failed to initialize translator model: {e}")
    translator = None  # 모델 로드 실패 시 None 처리하여 앱 크래시 방지


# --- 4. Utility Functions ---

def update_task_status(
        file_id: str,
        progress: int,
        message: str,
        status: Optional[str] = None,
        **kwargs
) -> None:
    """
    작업 상태(TASK_STATUS)를 업데이트하는 유틸리티 함수

    Args:
        file_id (str): 작업 고유 ID
        progress (int): 진행률 (0~100)
        message (str): 상태 메시지
        status (str, optional): 작업 상태 코드 ('processing', 'completed', 'error' 등)
        **kwargs: 추가로 업데이트할 데이터 (예: result, word 등)
    """
    if file_id in TASK_STATUS:
        TASK_STATUS[file_id].update({
            'progress': progress,
            'message': message
        })

        if status:
            TASK_STATUS[file_id]['status'] = status

        # 기타 결과 데이터 병합
        if kwargs:
            TASK_STATUS[file_id].update(kwargs)