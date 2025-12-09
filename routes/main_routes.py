import logging
from flask import Blueprint, render_template

# 로거 설정
logger = logging.getLogger(__name__)

# Blueprint 생성
main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index() -> str:
    """
    메인 페이지(Index) 렌더링
    """
    logger.info("Main page accessed")
    return render_template('index.html')