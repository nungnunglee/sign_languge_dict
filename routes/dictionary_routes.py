import logging
from flask import Blueprint, request, jsonify, Response
from services.dictionary_service import DICTIONARY_DATA

# 로거 및 Blueprint 설정
logger = logging.getLogger(__name__)
dict_bp = Blueprint('dictionary', __name__)


@dict_bp.route('/api/search', methods=['POST'])
def search_dictionary() -> Response:
    """
    수어 사전 검색 API

    Request Body:
        { "query": "검색어" }

    Returns:
        JSON: { "success": bool, "results": List[dict] }
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip().lower()

        if not query:
            return jsonify({'success': True, 'results': []})

        # 부분 일치 검색 (대소문자 무시)
        results = [
            {
                "id": item['id'],
                "word": item['word'],
                "video_url": f'/api/video/dictionary/{item["id"]}'
            }
            for item in DICTIONARY_DATA
            if query in item['word'].lower()
        ]

        logger.info(f"Dictionary search: '{query}' -> {len(results)} results")
        return jsonify({'success': True, 'results': results})

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'success': False, 'error': 'Internal Server Error'}), 500