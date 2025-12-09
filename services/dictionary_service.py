import json
import logging
import os
from typing import List, Dict, Optional, Any

# global_state에서 경로 상수 임포트
from services.global_state import VIDEO_DIR, JSON_DIR

logger = logging.getLogger(__name__)


def find_video_file(target_filename: str) -> Optional[str]:
    """
    [Smart Matching]
    JSON 메타데이터의 파일명이 실제 폴더에 없을 경우,
    파일명의 ID 접두사를 이용하여 실제 파일을 검색합니다.
    (예: JSON의 'REAL01_meta.mp4' -> 실제 폴더의 'REAL01_F01.mp4' 매칭)
    """
    if not target_filename:
        return None

    target_path = os.path.join(VIDEO_DIR, target_filename)

    # 1. Exact Match (정확히 일치하는 파일이 있는 경우)
    if os.path.exists(target_path):
        return target_filename

    # 2. Fuzzy Match (Prefix Search)
    try:
        # 파일명에서 확장자 제거 및 ID 추출 로직
        # 예: 'NIA_SL_WORD0001_REAL01' -> 'NIA_SL_WORD0001'
        if "REAL" in target_filename:
            prefix = target_filename.split("REAL")[0]
        else:
            prefix = os.path.splitext(target_filename)[0]

        # VIDEO_DIR 내 모든 파일 스캔
        if os.path.exists(VIDEO_DIR):
            for file in os.listdir(VIDEO_DIR):
                if file.startswith(prefix) and file.endswith('.mp4'):
                    logger.info(f"🔄 Auto-matched: {target_filename} (Missing) -> {file} (Found)")
                    return file
    except Exception as e:
        logger.warning(f"Error during fuzzy match for {target_filename}: {e}")

    return None


def load_dictionary_from_files() -> List[Dict[str, Any]]:
    """
    JSON 디렉토리와 VIDEO 디렉토리를 스캔하여
    유효한(영상 파일이 존재하는) 단어 데이터 리스트를 생성합니다.
    """
    data_list = []

    # 디렉토리 존재 확인
    if not os.path.exists(JSON_DIR):
        logger.warning(f"JSON directory missing: {JSON_DIR}")
        return []
    if not os.path.exists(VIDEO_DIR):
        logger.warning(f"Video directory missing: {VIDEO_DIR}")
        return []

    # JSON 파일 순회
    for filename in os.listdir(JSON_DIR):
        if not filename.endswith('.json'):
            continue

        json_path = os.path.join(JSON_DIR, filename)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

                # 1. 데이터 파싱 (구조가 맞지 않으면 건너뜀)
                try:
                    # NIA 수어 데이터셋 표준 구조 가정
                    word = content['data'][0]['attributes'][0]['name']
                    meta_video_name = content['metaData']['name']
                except (KeyError, IndexError, TypeError):
                    continue

                # 2. 실제 비디오 파일 매칭
                actual_video_name = find_video_file(meta_video_name)

                # 영상이 없으면 사전에 등록하지 않음
                if not actual_video_name:
                    continue

                # 3. 데이터 등록
                file_id = os.path.splitext(actual_video_name)[0]
                data_list.append({
                    "id": file_id,
                    "word": word,
                    "video_filename": actual_video_name
                })

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format: {filename}")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")

    logger.info(f"✅ Dictionary loaded: {len(data_list)} words available.")
    return data_list


# --- Initialize Data ---
# 서버 시작 시 데이터 로드
DICTIONARY_DATA: List[Dict[str, Any]] = load_dictionary_from_files()