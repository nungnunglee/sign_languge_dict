import os
import time
import json
import uuid
import logging
import shutil
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
import threading
from model_interface import SignLanguageTranslator, TranslationResult, VideoInfo, FrameKeypoints

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 현재 파일(app.py)의 디렉토리 (프로젝트 루트 디렉토리)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. 설정 및 상수 정의 ---
# 업로드된 파일이 저장될 폴더 (BASE_DIR에 위치)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 데이터베이스 경로 정의
DATABASE_DIR = os.path.join(BASE_DIR, 'database')
JSON_DIR = os.path.join(DATABASE_DIR, 'json')
VIDEO_DIR = os.path.join(DATABASE_DIR, 'video')

# 경로 존재 여부 확인 및 생성
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# 인메모리 상태 저장소
TASK_STATUS = {} # 비동기 번역 작업 상태 저장소 {task_id: {original_path: str, annotated_path: str, ...}}
DICTIONARY_DATA = [] # 수어 사전 데이터 {id: str, word: str, video_filename: str}

# 모델 인스턴스
translator = SignLanguageTranslator()

# --- 사전 데이터 로드 함수 ---
def load_dictionary_data():
    """
    database/json 폴더에서 NIA_SL_WORD{0001~0050}_REAL01_F_morpheme.json 파일을 읽어 
    수어 사전 데이터를 구성합니다.
    """
    dictionary = []
    # 1부터 50까지의 파일 번호를 순회
    for i in range(1, 51):
        # 4자리 ID 형식 (예: '0001')
        file_id = f"{i:04d}"
        
        # json 파일
        json_filename = f"NIA_SL_WORD{file_id}_REAL01_F_morpheme.json"
        json_filepath = os.path.join(JSON_DIR, json_filename)
        
        # mp4 파일
        video_filename = f"NIA_SL_WORD{file_id}_REAL02_F.mp4"
        video_filepath = os.path.join(VIDEO_DIR, video_filename)

        # JSON 파일과 대응되는 동영상 파일이 모두 존재하는지 확인
        if not os.path.exists(json_filepath):
            logger.warning(f"Dictionary Load: JSON file not found for ID {file_id}: {json_filepath}")
            continue
        
        if not os.path.exists(video_filepath):
            logger.warning(f"Dictionary Load: Video file not found for ID {file_id}: {video_filepath}")
            # 동영상이 없어도 검색 결과에는 포함시키기 위해 continue 대신 pass
            pass 

        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON 파일에서 단어 추출 (data[0].attributes[0].name)
            # JSON 예시 구조를 기반으로 경로를 수정합니다.
            word = data['data'][0]['attributes'][0]['name']
            
            dictionary.append({
                "id": file_id,
                "word": word,
                "video_filename": video_filename # 동영상 서빙을 위해 파일명 저장
            })
        except Exception as e:
            logger.error(f"Error processing JSON file {json_filepath}: {e}")

    logger.info(f"Loaded {len(dictionary)} dictionary entries from database.")
    return dictionary

# Flask 앱 인스턴스 생성
app = Flask(
    __name__, 
    template_folder=os.path.join(BASE_DIR, 'templates'), 
    static_folder=os.path.join(BASE_DIR, 'static')
)

# 전역 변수에 실제 데이터 로드
DICTIONARY_DATA = load_dictionary_data()

# --- 0. 메인 라우트 ---
@app.route('/')
def index():
    """메인 HTML 페이지 반환"""
    return render_template('index.html')

# --- 1. 파일 업로드 API ---
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """동영상 파일을 서버에 저장하고 task_id를 생성하여 반환"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '파일이 없습니다.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '파일 이름이 없습니다.'}), 400
    
    # 허용되는 파일 확장자 확인 (간단한 검증)
    allowed_extensions = {'mp4', 'mov', 'avi'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'success': False, 'error': '지원하지 않는 동영상 형식입니다.'}), 400

    # 고유 ID 및 저장 경로 설정
    file_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    
    # task_id로 임시 디렉토리 생성
    task_dir = os.path.join(UPLOAD_FOLDER, file_id)
    os.makedirs(task_dir, exist_ok=True)

    original_path = os.path.join(task_dir, 'original.mp4') # original 영상을 저장시킬 경로
    
    try:
        file.save(original_path) # original 영상 저장
    except Exception as e:
        logger.error(f"파일 저장 오류: {e}")
        # 파일 저장 실패 시 생성된 디렉토리 정리
        try:
            shutil.rmtree(task_dir)
        except Exception as cleanup_error:
            logger.warning(f"디렉토리 정리 실패: {cleanup_error}")
        return jsonify({'success': False, 'error': '파일 저장 중 오류가 발생했습니다.'}), 500

    # TASK_STATUS에 초기 정보 저장
    TASK_STATUS[file_id] = {
        'progress': 0,
        'message': '업로드 완료, 번역 대기 중',
        'status': 'uploaded',
        'original_path': original_path,
        'annotated_path': os.path.join(task_dir, 'annotated.mp4'), # 결과 영상 경로
        'word': None,
        'error': None
    }
    
    logger.info(f"파일 업로드 및 task_id 생성: {file_id}")

    return jsonify({
        'success': True,
        'file_id': file_id,
        'filename': filename,
        'message': '파일이 성공적으로 업로드되었습니다'
    })

# --- TASK_STATUS를 업데이트하는 유틸리티 함수 ---
def update_task_status(file_id, progress, message):
    if file_id in TASK_STATUS:
        TASK_STATUS[file_id]['progress'] = progress
        TASK_STATUS[file_id]['message'] = message
        # logging.info(f"업데이트 확인 {file_id}: {TASK_STATUS[file_id]['progress']}% - {TASK_STATUS[file_id]['message']}")
    else:
        logger.warning(f"Task ID {file_id}를 찾을 수 없습니다. 상태 업데이트 실패.")

# --- 실제 번역 작업을 실행하고 상태를 업데이트하는 함수 ---
def run_translation_in_thread(file_id, task_data):

    logger.info(f"현재 작업 {file_id}가 백그라운드에서 작업을 시작했습니다.")
    
    # progress 콜백 함수 정의
    def progress_callback(progress: int, message: str):
        update_task_status(file_id, progress, message) # TASK_STATUS를 업데이트
        logger.info(f"[작업 {file_id}의 진행] {progress}%: {message}")

    try:
        # model.py의 SignLanguageTranslator를 사용하여 번역 실행
        result = translator.translate_sign_language(
            video_path=task_data['original_path'],
            output_dir=os.path.dirname(task_data['original_path']),
            progress_callback=progress_callback # progress 콜백 함수
        )
        
        # 번역 결과 업데이트
        task_data['status'] = 'completed' if result.success else 'error'
        task_data['word'] = result.word # 번역된 단어
        task_data['annotated_path'] = result.annotated_video_path # annotated된 영상의 경로
        task_data['error'] = result.error

        # 최종 진행 상황 업데이트
        update_task_status(file_id, 100 if result.success else task_data['progress'], 
                           f"✅ 번역 완료: {result.word}" if result.success else f"❌ 번역 오류: {result.error}")
        
        logger.info(f"현재 작업 {file_id}의 번역 완료. 결과: {result.word}, 성공: {result.success}")

    except Exception as e:
        logger.error(f"Task {file_id} 처리 중 예외 발생: {e}", exc_info=True)
        task_data['status'] = 'error'
        task_data['error'] = str(e)
        task_data['word'] = "번역 실패"
        update_task_status(file_id, task_data['progress'], f"❌ 심각한 오류 발생: {str(e)}")

# --- 2. 비동기 번역 시작 API ---
@app.route('/api/translate', methods=['POST'])
def start_translation():
    """지정된 file_id에 대해 비동기 번역 작업을 시작"""
    data = request.json
    file_id = data.get('file_id')

    if file_id not in TASK_STATUS:
        return jsonify({'success': False, 'error': '유효하지 않은 파일 ID입니다.'}), 404
    
    task_data = TASK_STATUS[file_id]
    task_data['status'] = 'processing'
    task_data['progress'] = 0
    task_data['message'] = '번역 작업 시작 중...'
    
    # 스레드 설정 (데몬 스레드로 설정하여 메인 프로세스 종료 시 함께 종료)
    thread = threading.Thread(
        target=run_translation_in_thread, # run_translation_in_thread 함수 실행
        args=(file_id, task_data),
        daemon=True
    )
    # 스레드 시작
    thread.start()
    logger.info(f"현재 작업 {file_id} 의 번역 스레드가 시작되었습니다.")

    return jsonify({
        'success': True,
        'task_id': file_id,
        'message': '번역 작업이 백그라운드에서 시작되었습니다.'
    })


# --- 3. 번역 진행 상황 SSE 스트림 API ---
@app.route('/api/translate/progress/<task_id>')
def stream_progress(task_id):
    """SSE를 통해 번역 진행 상황을 실시간으로 스트리밍"""

    logger.info(f"번역 진행 상황을 실시간으로 스트리밍 하겠습니다.")
    if task_id not in TASK_STATUS:
        return Response(
            json.dumps({'error': '유효하지 않은 Task ID입니다.'}), 
            status=404, 
            mimetype='application/json'
        )

    def generate():
        previous_status = None
        
        while True:
            # 상태 변경이 있을 때만 이벤트 전송
            current_status = TASK_STATUS[task_id]['status']
            current_progress = TASK_STATUS[task_id]['progress']
            current_message = TASK_STATUS[task_id]['message']
            
            # 이전 상태와 비교하여 변경 사항이 있거나, 0.5초마다 강제 전송
            if current_status != previous_status or current_progress % 5 == 0 or current_progress == 0 or current_progress == 100:
                
                # 1. 진행 상황 이벤트 전송
                data = json.dumps({'progress': current_progress, 'message': current_message})
                yield f"event: progress\ndata: {data}\n\n"
                
                previous_status = current_status
            
            # 2. 완료 또는 에러 상태 체크 후 종료
            if current_status == 'completed':
                # 최종 완료 이벤트 전송
                word = TASK_STATUS[task_id]['word']
                complete_data = json.dumps({
                    'word': word,
                    'task_id': task_id,
                    'annotated_video_url': f'/api/video/annotated/{task_id}'
                })
                yield f"event: complete\ndata: {complete_data}\n\n"
                break
                
            elif current_status == 'error':
                # 최종 에러 이벤트 전송
                error_data = json.dumps({
                    'message': TASK_STATUS[task_id]['error'],
                    'task_id': task_id
                })
                yield f"event: error\ndata: {error_data}\n\n"
                break
                
            # 상태 폴링 간격
            time.sleep(0.3) 

    return Response(generate(), mimetype='text/event-stream')


# --- 4. 단어 검색 API (DICTIONARY_DATA 사용하도록 수정) ---
@app.route('/api/search', methods=['POST'])
def search_dictionary():
    """로드된 사전 데이터(DICTIONARY_DATA)에서 단어를 검색하여 반환"""
    data = request.json
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'success': True, 'results': []})

    # 부분 일치 검색 (대소문자 무시)
    results = []
    # DICTIONARY_DATA 구조: {id: str, word: str, video_filename: str}
    for item in DICTIONARY_DATA:
        if query.lower() in item['word'].lower():
            # 클라이언트에게는 video_filename 대신 video_id (여기서는 id)를 반환
            results.append({
                "id": item['id'], 
                "word": item['word'], 
                "video_url": f'/api/video/dictionary/{item["id"]}' # 클라이언트에서 사용할 URL
            })
            
    logger.info(f"'{query}' 검색 결과: {len(results)}개 항목")

    return jsonify({'success': True, 'results': results})


# --- 동영상 경로 반환 헬퍼 함수 ---
def get_video_path(video_type, identifier):
    """
    요청 타입과 ID에 따라 동영상 파일의 실제 경로를 반환합니다.
    """
    if video_type == 'dictionary':
        # 사전 검색: identifier는 DICTIONARY_DATA의 'id'
        for item in DICTIONARY_DATA:
            if item['id'] == identifier:
                # 동영상 파일은 VIDEO_DIR에 존재한다고 가정
                filepath = os.path.join(VIDEO_DIR, item['video_filename'])
                logger.info(f"이 단어의 비디오 경로는 dictionary_path: {filepath}")
                return filepath
        return None
    
    elif video_type in ['original', 'annotated']:
        # 번역 작업: identifier는 TASK_STATUS의 task_id
        if identifier not in TASK_STATUS:
            return None
            
        task_data = TASK_STATUS[identifier]
        if video_type == 'original': # 원본 영상을 찾는 경우
            logger.info(f"이 작업의 비디오 경로는 original_path: {task_data.get('original_path')}")
            return task_data.get('original_path')
        elif video_type == 'annotated': # annotated 영상을 찾는 경우
            logger.info(f"이 작업의 비디오 경로는 annotated_path: {task_data.get('annotated_path')}")
            return task_data.get('annotated_path')
            
    return None

# --- 5. 동영상 스트리밍 API ---
@app.route('/api/video/<video_type>/<identifier>')
def serve_video(video_type, identifier):
    """
    지정된 타입(original, annotated, dictionary)의 동영상 파일을 스트리밍
    """
    # 보안: video_type이 허용된 값인지 확인
    if video_type not in ['original', 'annotated', 'dictionary']:
        return jsonify({'error': '유효하지 않은 영상 타입입니다.'}), 400
    
    # 실제 파일 경로 획득
    filepath = get_video_path(video_type, identifier)
    
    if not filepath or not os.path.exists(filepath):
        # 파일이 없으면 404
        # NOTE: dictionary 타입의 경우 파일 이름만 있고 실제 파일은 없을 수 있음 (테스트 환경)
        logger.error(f"요청하신 영상 파일을 찾을 수 없습니다. (Type: {video_type}, ID: {identifier}, Path: {filepath})")
        # 실제 환경에서 dictionary 영상이 없으면 임시 placeholder를 반환하거나 404 처리
        if video_type == 'dictionary':
             # dictionary 영상이 없으면 임시로 빈 파일 생성 후 스트리밍을 시도하거나, 
             # 여기서는 단순하게 404를 반환하도록 하겠습니다.
             return jsonify({'error': f'사전 영상 파일을 찾을 수 없습니다: {identifier}'}), 404
        
        # 번역 결과 영상이 아직 완료되지 않았을 수도 있음
        if video_type == 'annotated' and identifier in TASK_STATUS and TASK_STATUS[identifier].get('status') == 'processing':
             return jsonify({'error': f'번역 작업이 아직 완료되지 않았습니다. 잠시 후 다시 시도해 주세요.'}), 404

        return jsonify({'error': f'요청하신 영상 파일을 찾을 수 없습니다: {identifier}'}), 404


    # 파일이 위치한 디렉토리를 기준으로 send_from_directory를 사용해야 합니다.
    # UPLOAD_FOLDER 또는 VIDEO_DIR 중 하나가 됩니다.
    base_dir = os.path.dirname(filepath)
    
    # send_from_directory는 Content-Type, Range Requests 등을 자동으로 처리해줌
    return send_from_directory(
        directory=base_dir, 
        path=os.path.basename(filepath),
        mimetype='video/mp4',
        as_attachment=False
    )


# --- 서버 실행 ---
if __name__ == '__main__':    
    logger.info("Flask 앱 시작...")
    app.run(debug=True, host='0.0.0.0', port=5000)