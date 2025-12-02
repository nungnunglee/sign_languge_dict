# 수어 번역 웹 서버 API 명세서

## 개요
수어 동영상을 업로드하고 번역하며, 단어로 수어 동영상을 검색할 수 있는 웹 서비스입니다.

## 통신 방식
- **HTTP**: 파일 업로드, 단어 검색, 동영상 다운로드
- **SSE (Server-Sent Events)**: 번역 진행 상황 실시간 전달

## API 엔드포인트

### 1. 루트 페이지
**GET /**  
서버의 루트 경로로 접속 시 GUI HTML 페이지를 반환합니다.

**응답:**
- Content-Type: `text/html`
- HTML 파일 (프론트엔드 GUI)

---

### 2. 동작 검색 - 동영상 업로드
**POST /api/upload**  
수어 동영상 파일을 서버에 업로드합니다.

**요청:**
- Content-Type: `multipart/form-data`
- Body:
  ```
  file: [동영상 파일]
  ```

**응답:**
```json
{
  "success": true,
  "file_id": "unique_file_id_12345",
  "filename": "original_filename.mp4",
  "message": "파일이 성공적으로 업로드되었습니다"
}
```

**에러 응답:**
```json
{
  "success": false,
  "error": "파일 형식이 올바르지 않습니다"
}
```

---

### 3. 번역 시작
**POST /api/translate**  
업로드된 동영상을 번역 작업을 시작합니다.

**요청:**
```json
{
  "file_id": "unique_file_id_12345"
}
```

**응답:**
```json
{
  "success": true,
  "task_id": "task_12345",
  "message": "번역 작업이 시작되었습니다"
}
```

**에러 응답:**
```json
{
  "success": false,
  "error": "파일을 찾을 수 없습니다"
}
```

---

### 4. 번역 진행 상황 (SSE)
**GET /api/translate/progress/{task_id}**  
번역 작업의 진행 상황을 실시간으로 전달합니다.

**요청:**
- Accept: `text/event-stream`
- Connection: `keep-alive`

**SSE 이벤트 스트림:**
```
event: progress
data: {"progress": 25, "status": "processing", "message": "키포인트 추출 중..."}

event: progress
data: {"progress": 50, "status": "processing", "message": "모델 분석 중..."}

event: progress
data: {"progress": 75, "status": "processing", "message": "결과 생성 중..."}

event: complete
data: {"progress": 100, "status": "completed", "word": "안녕하세요", "annotated_video_url": "/api/video/annotated/task_12345"}

event: error
data: {"status": "error", "message": "번역 중 오류가 발생했습니다"}
```

**이벤트 타입:**
- `progress`: 진행 중 (0-99%)
- `complete`: 완료 (100%)
- `error`: 오류 발생

---

### 5. 번역 결과 조회
**GET /api/translate/result/{task_id}**  
번역 작업의 최종 결과를 조회합니다.

**응답:**
```json
{
  "success": true,
  "status": "completed",
  "word": "안녕하세요",
  "annotated_video_url": "/api/video/annotated/task_12345",
  "original_video_url": "/api/video/original/task_12345"
}
```

**진행 중인 경우:**
```json
{
  "success": true,
  "status": "processing",
  "progress": 50
}
```

---

### 6. 원본 동영상 조회
**GET /api/video/original/{task_id}**  
업로드된 원본 동영상을 스트리밍합니다.

**응답:**
- Content-Type: `video/mp4` (또는 업로드된 형식)
- Content-Length: 파일 크기
- Accept-Ranges: `bytes`
- 동영상 바이너리 데이터

---

### 7. 키포인트 주석 동영상 조회
**GET /api/video/annotated/{task_id}**  
키포인트가 그려진 동영상을 스트리밍합니다.

**응답:**
- Content-Type: `video/mp4`
- Content-Length: 파일 크기
- Accept-Ranges: `bytes`
- 동영상 바이너리 데이터

**참고:** 클라이언트에서 키포인트 표시/숨김을 토글할 수 있도록, 원본 동영상과 주석 동영상 모두 제공합니다.

---

### 8. 단어 검색
**POST /api/search**  
단어로 해당하는 수어 동작 동영상을 검색합니다.

**요청:**
```json
{
  "word": "안녕하세요"
}
```

**응답:**
```json
{
  "success": true,
  "results": [
    {
      "word": "안녕하세요",
      "video_url": "/api/video/dictionary/안녕하세요_1",
      "description": "표준 수어 동작"
    },
    {
      "word": "안녕하세요",
      "video_url": "/api/video/dictionary/안녕하세요_2",
      "description": "대안 수어 동작"
    }
  ]
}
```

**결과 없음:**
```json
{
  "success": true,
  "results": [],
  "message": "검색 결과가 없습니다"
}
```

---

### 9. 사전 동영상 조회
**GET /api/video/dictionary/{word_id}**  
사전에 저장된 수어 동작 동영상을 스트리밍합니다.

**응답:**
- Content-Type: `video/mp4`
- Content-Length: 파일 크기
- Accept-Ranges: `bytes`
- 동영상 바이너리 데이터

---

## 데이터 흐름

### 시나리오 1: 동작 검색 및 번역
```
1. 클라이언트 → POST /api/upload
   서버: 파일 임시 저장, file_id 반환
   
2. 클라이언트: 업로드 성공 확인 (file_id 저장)
   
3. 클라이언트 → POST /api/translate (file_id)
   서버: 번역 작업 시작, task_id 반환
   
4. 클라이언트 → GET /api/translate/progress/{task_id} (SSE 연결)
   서버: 진행 상황 실시간 전송
   
5. 서버 → 클라이언트 (SSE): progress 이벤트들
   
6. 서버 → 클라이언트 (SSE): complete 이벤트 (word, annotated_video_url)
   
7. 클라이언트: word 표시, 동영상 플레이어에 URL 설정
   
8. 클라이언트 → GET /api/video/original/{task_id}
   클라이언트 → GET /api/video/annotated/{task_id}
   서버: 동영상 스트리밍
   
9. 클라이언트: 키포인트 표시/숨김 토글 (원본/주석 동영상 전환)
```

### 시나리오 2: 단어 검색
```
1. 클라이언트 → POST /api/search (word)
   서버: 사전에서 검색, 결과 반환
   
2. 클라이언트: 검색 결과 리스트 표시
   
3. 클라이언트 → GET /api/video/dictionary/{word_id}
   서버: 사전 동영상 스트리밍
   
4. 클라이언트: 동영상 재생
```

---

## 기술 고려사항

### HTTP vs SSE 선택 이유
- **HTTP POST**: 파일 업로드, 단어 검색 등 요청-응답 패턴에 적합
- **SSE**: 번역 진행 상황을 실시간으로 전달하기 위해 사용
  - WebSocket보다 단순하고 서버→클라이언트 단방향 통신에 적합
  - HTTP 기반이라 방화벽/프록시 호환성 좋음
  - 자동 재연결 지원

### 파일 저장 정책
- 업로드된 동영상은 **임시 저장**만 수행
- 번역 완료 후 일정 시간(예: 1시간) 후 자동 삭제
- task_id 기반으로 파일 관리
- 메모리 기반 또는 임시 디렉토리 사용

### 동영상 스트리밍
- HTTP Range Requests 지원 (부분 요청)
- Content-Range 헤더로 스트리밍 최적화
- 클라이언트에서 seek 가능하도록 구현

---

## 에러 처리

### 공통 에러 응답 형식
```json
{
  "success": false,
  "error": "에러 메시지",
  "error_code": "ERROR_CODE"
}
```

### 주요 에러 코드
- `FILE_NOT_FOUND`: 파일을 찾을 수 없음
- `INVALID_FILE_FORMAT`: 지원하지 않는 파일 형식
- `TRANSLATION_FAILED`: 번역 작업 실패
- `WORD_NOT_FOUND`: 검색 결과 없음
- `SERVER_ERROR`: 서버 내부 오류

---

## 보안 고려사항
- 인증 없음 (요구사항에 따라)
- 파일 크기 제한 필요 (예: 100MB)
- 파일 형식 검증 (mp4, avi, mov 등)
- CORS 설정 (필요시)
- Rate limiting (선택사항)

