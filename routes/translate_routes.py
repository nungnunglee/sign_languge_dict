import json
import logging
import os
import threading
import time
import uuid
from typing import Generator, Dict, Any

from flask import Blueprint, request, jsonify, Response, send_from_directory

# Local Imports
from services.global_state import (
    TASK_STATUS, UPLOAD_FOLDER, VIDEO_DIR,
    translator, update_task_status
)
from services.dictionary_service import DICTIONARY_DATA

# 로거 및 Blueprint 설정
logger = logging.getLogger(__name__)
translate_bp = Blueprint('translate', __name__)


# --- Helper Functions ---

def _run_translation_task(file_id: str, task_data: Dict[str, Any]) -> None:
    """
    [Background Task] 수어 번역 AI 모델 실행
    """
    try:
        original_path = task_data['original_path']
        output_dir = os.path.dirname(original_path)

        # 모델 실행
        result = translator.translate_sign_language(
            video_path=original_path,
            output_dir=output_dir,
            progress_callback=lambda p, m: update_task_status(file_id, p, m)
        )

        # 결과 처리
        success_msg = f"✅ 번역 완료: {result.word}"
        error_msg = f"❌ 오류: {result.error}"

        final_msg = success_msg if result.success else error_msg
        status = 'completed' if result.success else 'error'

        update_task_status(
            file_id, 100, final_msg,
            status=status,
            word=result.word,
            annotated_path=result.annotated_video_path,
            error=result.error
        )
        logger.info(f"Task {file_id} completed successfully.")

    except Exception as e:
        logger.error(f"Translation Task Error ({file_id}): {e}")
        update_task_status(
            file_id, 0,
            f"❌ 시스템 오류: {str(e)}",
            status='error',
            error=str(e)
        )


# --- Routes ---

@translate_bp.route('/api/upload', methods=['POST'])
def upload_file() -> Response:
    """번역용 영상 파일 업로드 처리"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '파일이 없습니다.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '파일명이 비어있습니다.'}), 400

    try:
        file_id = str(uuid.uuid4())
        task_dir = os.path.join(UPLOAD_FOLDER, file_id)
        os.makedirs(task_dir, exist_ok=True)

        original_path = os.path.join(task_dir, 'original.mp4')
        file.save(original_path)

        # 초기 상태 등록
        TASK_STATUS[file_id] = {
            'progress': 0,
            'message': '업로드 완료, 대기 중...',
            'status': 'uploaded',
            'original_path': original_path,
            'annotated_path': os.path.join(task_dir, 'annotated.mp4'),
            'word': None,
            'error': None
        }

        return jsonify({'success': True, 'file_id': file_id, 'filename': file.filename})

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'success': False, 'error': '파일 저장 실패'}), 500


@translate_bp.route('/api/translate', methods=['POST'])
def start_translation() -> Response:
    """번역 작업 시작 요청"""
    data = request.get_json()
    file_id = data.get('file_id')

    if not file_id or file_id not in TASK_STATUS:
        return jsonify({'success': False, 'error': '유효하지 않은 Task ID'}), 404

    TASK_STATUS[file_id]['status'] = 'processing'

    # 백그라운드 스레드 시작
    thread = threading.Thread(
        target=_run_translation_task,
        args=(file_id, TASK_STATUS[file_id]),
        daemon=True
    )
    thread.start()

    return jsonify({'success': True, 'task_id': file_id})


@translate_bp.route('/api/translate/progress/<task_id>')
def stream_progress(task_id: str) -> Response:
    """
    SSE (Server-Sent Events) - 실시간 진행률 스트리밍
    """
    if task_id not in TASK_STATUS:
        return Response('Task not found', status=404)

    def generate() -> Generator[str, None, None]:
        last_progress = -1
        last_status = None

        while True:
            task = TASK_STATUS.get(task_id)
            if not task:
                break

            curr_status = task['status']
            curr_progress = task['progress']

            # 상태나 진행률이 변경되었을 때만 이벤트 전송
            if curr_status != last_status or curr_progress != last_progress:
                payload = {'progress': curr_progress, 'message': task['message']}
                yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

                last_status = curr_status
                last_progress = curr_progress

            # 완료 상태 처리
            if curr_status == 'completed':
                result_payload = {
                    'task_id': task_id,
                    'word': task.get('word'),
                    'game_result': task.get('result')  # 게임 결과 존재 시 포함
                }
                yield f"event: complete\ndata: {json.dumps(result_payload)}\n\n"
                break

            # 에러 상태 처리
            elif curr_status == 'error':
                error_payload = {'message': task.get('error', 'Unknown Error')}
                yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                break

            time.sleep(0.5)  # 폴링 간격

    return Response(generate(), mimetype='text/event-stream')


@translate_bp.route('/api/video/<video_type>/<identifier>')
def serve_video(video_type: str, identifier: str) -> Response:
    """
    비디오 스트리밍 엔드포인트

    Args:
        video_type: 'dictionary' | 'original' | 'annotated'
        identifier: dict_id 또는 task_id
    """
    file_path = None

    # 1. 사전 영상 요청인 경우
    if video_type == 'dictionary':
        item = next((i for i in DICTIONARY_DATA if i['id'] == identifier), None)
        if item:
            file_path = os.path.join(VIDEO_DIR, item['video_filename'])

    # 2. 사용자 업로드/분석 결과 영상 요청인 경우
    elif video_type in ['original', 'annotated']:
        task = TASK_STATUS.get(identifier)
        if task:
            file_path = task.get(f'{video_type}_path')

    # 3. 파일 존재 여부 확인 및 전송
    if file_path and os.path.exists(file_path):
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        return send_from_directory(directory, filename, mimetype='video/mp4')

    return jsonify({'error': 'Video not found'}), 404