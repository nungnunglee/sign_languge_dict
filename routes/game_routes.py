import logging
import os
import random
import threading
import uuid
from dataclasses import asdict
from typing import Dict, Any

from flask import Blueprint, request, jsonify, Response

# Local Imports
from services.global_state import TASK_STATUS, UPLOAD_FOLDER, translator, update_task_status
from services.dictionary_service import DICTIONARY_DATA

# 로거 및 Blueprint 설정
logger = logging.getLogger(__name__)
game_bp = Blueprint('game', __name__)


def _run_game_evaluation_task(file_id: str, task_data: Dict[str, Any], target_word: str) -> None:
    """
    [Background Task] 사용자 영상 채점 및 결과 업데이트
    """
    try:
        # translate_sign_language를 사용하여 비디오 분석
        video_path = task_data['original_path']
        output_dir = os.path.dirname(video_path)  # 업로드된 파일과 같은 디렉토리에 결과 저장
        
        translation_result = translator.translate_sign_language(
            video_path=video_path,
            output_dir=output_dir,
            progress_callback=lambda p, m: update_task_status(file_id, p, m)
        )
        
        if not translation_result.success:
            # 번역 실패 시
            update_task_status(
                file_id, 100,
                f"분석 실패: {translation_result.error or '알 수 없는 오류'}",
                status='error',
                error=translation_result.error or '알 수 없는 오류'
            )
            return
        
        # 인식된 단어와 정답 비교
        recognized_word = translation_result.word
        is_correct = (recognized_word == target_word)
        
        # 게임 결과 생성 (GameResult와 호환되는 형태)
        game_result = {
            'success': is_correct,
            'recognized_word': recognized_word,
            'score': 100 if is_correct else 0,
            'similarity': 1.0 if is_correct else 0.0,
            'processing_time': translation_result.processing_time
        }
        
        result_message = f"결과: {recognized_word}" + (" (정답!)" if is_correct else f" (정답: {target_word})")
        
        update_task_status(
            file_id, 100,
            result_message,
            status='completed',
            result=game_result
        )
    except Exception as e:
        logger.error(f"Game Evaluation Error: {e}", exc_info=True)
        update_task_status(file_id, 0, "채점 중 시스템 오류 발생", status='error', error=str(e))


@game_bp.route('/api/game/quiz', methods=['GET'])
def get_game_quiz() -> Response:
    """
    [Mode: Practice] AI 동작 인식 연습용 퀴즈 가져오기
    """
    if not DICTIONARY_DATA:
        return jsonify({'success': False, 'error': '사전 데이터가 없습니다.'}), 500

    item = random.choice(DICTIONARY_DATA)
    return jsonify({
        'success': True,
        'quiz': {
            'id': item['id'],
            'word': item['word'],
            'hint_video_url': f'/api/video/dictionary/{item["id"]}'
        }
    })


@game_bp.route('/api/game/multiple-choice', methods=['GET'])
def get_multichoice_quiz() -> Response:
    """
    [Mode: Multi-Choice] 4지선다 영상 퀴즈 생성
    """
    if len(DICTIONARY_DATA) < 4:
        return jsonify({'success': False, 'message': '데이터 부족 (최소 4개 필요)'}), 500

    # 정답 및 오답 보기 생성
    target = random.choice(DICTIONARY_DATA)
    others = [d for d in DICTIONARY_DATA if d['id'] != target['id']]
    wrong_options = random.sample(others, 3)

    # 보기 섞기
    options = [target] + wrong_options
    random.shuffle(options)

    return jsonify({
        'success': True,
        'quiz': {
            'target_word': target['word'],
            'target_id': target['id'],
            'options': [
                {
                    'id': opt['id'],
                    'video_url': f'/api/video/dictionary/{opt["id"]}'
                } for opt in options
            ]
        }
    })


@game_bp.route('/api/game/submit', methods=['POST'])
def submit_game_attempt() -> Response:
    """
    [Mode: Practice] 사용자 연습 영상 제출 및 채점 시작
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '파일이 없습니다.'}), 400

    file = request.files['file']
    target_word = request.form.get('target_word')

    try:
        file_id = str(uuid.uuid4())
        task_dir = os.path.join(UPLOAD_FOLDER, file_id)
        os.makedirs(task_dir, exist_ok=True)

        save_path = os.path.join(task_dir, 'original.mp4')
        file.save(save_path)

        # 태스크 초기화
        TASK_STATUS[file_id] = {
            'progress': 0,
            'message': '분석 준비 중...',
            'status': 'processing',
            'original_path': save_path
        }

        # 백그라운드 스레드 실행
        threading.Thread(
            target=_run_game_evaluation_task,
            args=(file_id, TASK_STATUS[file_id], target_word),
            daemon=True
        ).start()

        return jsonify({'success': True, 'task_id': file_id})

    except Exception as e:
        logger.error(f"Game Submit Error: {e}")
        return jsonify({'success': False, 'message': '서버 저장 중 오류'}), 500