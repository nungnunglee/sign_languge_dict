# 수어 마스터 AI

수어 동영상을 입력하여 단어를 검색하거나, 단어로 수어 동영상을 검색할 수 있는 수어 학습 플랫폼입니다. 게임 기능을 통해 재미있게 수어를 학습할 수 있습니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행](#설치-및-실행)
- [사용 방법](#사용-방법)
- [API 문서](#api-문서)
- [데이터베이스 구조](#데이터베이스-구조)

## ✨ 주요 기능

### 1. 수어 검색 (동영상 → 단어)
- 동영상 파일 업로드 또는 웹캠으로 실시간 촬영
- OpenPose를 통한 키포인트 추출
- Transformer-XL 모델을 활용한 수어 인식
- 실시간 진행 상황 표시 (SSE)
- 키포인트가 시각화된 결과 동영상 제공

### 2. 수어 사전 (단어 → 동영상)
- 단어 검색으로 해당 수어 동영상 찾기
- 50개의 수어 단어 데이터베이스 제공
- 동영상 재생 및 학습

### 3. 수어 게임
- **동영상 맞추기**: 동영상을 보고 어떤 단어인지 맞추기
- **동영상 찾기**: 단어를 보고 해당하는 수어 동영상 찾기
- 게임을 통한 재미있는 수어 학습

## 🛠 기술 스택

### 백엔드
- **Flask**: 웹 서버 프레임워크
- **PyTorch**: 딥러닝 모델 프레임워크
- **Transformer-XL**: 수어 인식을 위한 시퀀스 모델
  - AIHub 데이터셋으로 자체 학습
- **OpenPose**: 키포인트 추출
- **OpenCV**: 동영상 처리 및 시각화

### 프론트엔드
- **Vanilla JavaScript**: 프레임워크 없이 순수 JavaScript 사용
- **HTML5/CSS3**: 반응형 웹 디자인
- **Server-Sent Events (SSE)**: 실시간 진행 상황 스트리밍

### 데이터
- **AIHub 수어 데이터셋**: 모델 학습용
- **JSON + MP4**: 수어 사전 데이터 저장 형식

## 📁 프로젝트 구조

```
hand_server/
├── app.py                 # Flask 메인 서버
├── model_interface.py     # 모델 인터페이스 및 번역 로직
├── prediction_utils.py    # 예측 유틸리티 (OpenPose, 모델 추론)
├── transformer_xl.py      # Transformer-XL 모델 구현
├── weight_tuning.py       # 모델 가중치 튜닝
├── word_list.csv          # 수어 단어 목록 (50개)
│
├── templates/
│   └── index.html         # 메인 HTML 페이지
│
├── static/
│   ├── script.js          # 프론트엔드 JavaScript
│   └── style.css         # 스타일시트
│
├── database/
│   ├── json/              # 수어 사전 JSON 메타데이터
│   │   └── NIA_SL_WORD{0001~0050}_REAL02_F_morpheme.json
│   └── video/             # 수어 사전 동영상 파일
│       └── NIA_SL_WORD{0001~0050}_REAL02_F.mp4
│
├── uploads/               # 업로드된 동영상 임시 저장
│
└── docs/                  # 프로젝트 문서
    ├── API_SPECIFICATION.md
    ├── DATA_FLOW.md
    ├── FRONTEND_BACKEND_DIVISION.md
    └── MODEL_INTERFACE.md
```

## 🚀 설치 및 실행

### 필수 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장, CPU도 가능)
- OpenPose 모델 파일
- 학습된 Transformer-XL 모델 가중치

### 1. 저장소 클론

```bash
git clone <repository-url>
cd hand_server
```

### 2. Python 패키지 설치

```bash
pip install flask torch torchvision opencv-python numpy orjson pillow werkzeug
```

### 3. 모델 및 데이터 준비

#### OpenPose 모델
- OpenPose 모델 파일을 `model_interface.py`의 `DEFAULT_OPENPOSE_MODELS` 경로에 배치

#### Transformer-XL 모델
- 학습된 모델의 `hparams.json` 파일 경로를 `model_interface.py`의 `DEFAULT_MODEL_CONFIG`에 설정

#### 수어 사전 데이터
- `database/json/` 폴더에 JSON 메타데이터 파일 배치
- `database/video/` 폴더에 동영상 파일 배치
- 파일명 형식: `NIA_SL_WORD{0001~0050}_REAL02_F_morpheme.json` / `NIA_SL_WORD{0001~0050}_REAL02_F.mp4`

### 4. 서버 실행

```bash
python app.py
```

서버가 `http://0.0.0.0:5000`에서 실행됩니다.

브라우저에서 `http://localhost:5000`으로 접속하여 사용할 수 있습니다.

## 📖 사용 방법

### 수어 검색 (동영상 → 단어)

1. 메인 화면에서 **"수어 검색"** 카드 클릭
2. **파일 업로드** 또는 **웹캠 촬영** 선택
3. 동영상 업로드 또는 웹캠으로 촬영
4. **번역 시작** 버튼 클릭
5. 실시간 진행 상황 확인
6. 번역 결과 단어 및 키포인트 시각화 동영상 확인

### 수어 사전 (단어 → 동영상)

1. 메인 화면에서 **"수어 사전"** 카드 클릭
2. 검색창에 단어 입력 (예: "고민", "눈", "슬프다")
3. 검색 결과에서 동영상 선택하여 재생

### 수어 게임

1. 메인 화면에서 **"수어 게임"** 카드 클릭
2. 게임 모드 선택:
   - **동영상 맞추기**: 동영상을 보고 단어 선택
   - **동영상 찾기**: 단어를 보고 해당 동영상 선택
3. 정답 확인 및 점수 확인

## 📡 API 문서

자세한 API 명세는 [docs/API_SPECIFICATION.md](docs/API_SPECIFICATION.md)를 참고하세요.

### 주요 API 엔드포인트

- `POST /api/upload`: 동영상 파일 업로드
- `POST /api/translate`: 번역 작업 시작
- `GET /api/translate/progress/<task_id>`: 번역 진행 상황 (SSE)
- `POST /api/search`: 단어 검색
- `GET /api/video/<type>/<id>`: 동영상 스트리밍

## 💾 데이터베이스 구조

### 수어 사전 데이터

- **JSON 파일**: 각 단어의 메타데이터 (단어명, ID 등)
- **동영상 파일**: 해당 단어의 수어 동작 동영상

### 지원 단어 목록

현재 50개의 수어 단어를 지원합니다:
- 고민, 뻔뻔, 수어, 남아, 눈, 독신, 음료수, 발가락, 슬프다, 자극, 안타깝다, 어색하다, 여아, 외국인, 영아, 신사, 뉴질랜드, 나사렛대학교, 알아서, 장애인, 열아홉번째, 침착, 성실, 학교연혁, 싫어하다, 급하다, 필기시험, 병문안, 검사, 결승전1, 낚시터, 낚시대, 당뇨병, 독서, 매표소, 면역, 감기, 배드민턴, 변비, 병명, 보건소, 불면증, 불행, 붕대, 사위, 설사, 성병, 방충, 소화제, 손녀

전체 목록은 `word_list.csv`에서 확인할 수 있습니다.

## 🔧 설정

### 모델 경로 설정

`model_interface.py`에서 다음 경로를 수정할 수 있습니다:

```python
DEFAULT_MODEL_CONFIG = "path/to/hparams.json"
DEFAULT_OPENPOSE_MODELS = "path/to/openpose/models"
```

### 서버 설정

`app.py`에서 포트 및 호스트 설정:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

