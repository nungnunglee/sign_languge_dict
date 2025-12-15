import logging
from typing import Optional
from flask import Flask

# 각 기능별 Blueprint 임포트
from routes.main_routes import main_bp
from routes.translate_routes import translate_bp
from routes.dictionary_routes import dict_bp
from routes.game_routes import game_bp


def create_app(config_object: Optional[object] = None) -> Flask:
    """
    Flask Application Factory

    앱 생성과 설정을 분리하여 테스트와 확장을 용이하게 합니다.
    """
    # 1. 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 2. 앱 생성
    app = Flask(__name__)

    # 3. 설정 적용
    if config_object:
        app.config.from_object(config_object)
    else:
        # 기본 설정 (실제 배포 시 config.py 분리 권장)
        app.config.update(
            SECRET_KEY='dev-key-please-change',
            UPLOAD_FOLDER='uploads',
            MAX_CONTENT_LENGTH=100 * 1024 * 1024  # 16MB 제한
        )

    # 4. Blueprint 등록
    _register_blueprints(app)

    logging.info("Flask application initialized.")
    return app


def _register_blueprints(app: Flask):
    """Blueprint 등록 헬퍼 함수"""
    app.register_blueprint(main_bp)
    app.register_blueprint(translate_bp)
    app.register_blueprint(dict_bp)
    app.register_blueprint(game_bp)


if __name__ == '__main__':
    # 로컬 개발 환경 실행
    app = create_app()
    logging.info("Starting Flask server...")

    app.run(debug=True, host='0.0.0.0', port=5000)