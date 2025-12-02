# 모델/파이프라인 - 백엔드 인터페이스 명세서

## 개요
이 문서는 수어 번역 모델/파이프라인과 백엔드 웹 서버 간의 인터페이스를 정의합니다.
백엔드 개발자와 모델 개발자가 독립적으로 작업할 수 있도록 구체적인 입출력 형식을 명시합니다.

---

## 함수 시그니처 (Python 예시)

```python
def translate_sign_language(
    video_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> TranslationResult:
    """
    동영상을 분석하여 수어를 번역하고 키포인트를 추출합니다.
    
    Args:
        video_path: 입력 동영상 파일의 절대 경로 (str)
        output_dir: 결과 파일을 저장할 디렉토리 경로 (str)
        progress_callback: 진행 상황 콜백 함수 (progress: int, message: str) -> None
    
    Returns:
        TranslationResult: 번역 결과 객체
    
    Raises:
        FileNotFoundError: 동영상 파일을 찾을 수 없을 때
        VideoProcessingError: 동영상 처리 중 오류 발생 시
        ModelInferenceError: 모델 추론 중 오류 발생 시
    """
    pass
```

---

## 입력 사양

### 1. 동영상 파일 경로
- **타입**: `str` (절대 경로)
- **형식**: 파일 시스템 경로
- **예시**: 
  - Windows: `C:\tmp\hand_server\uploads\abc123.mp4`
  - Linux/Mac: `/tmp/hand_server/uploads/abc123.mp4`
- **요구사항**:
  - 파일이 존재해야 함
  - 읽기 권한 필요
  - 지원 형식: `.mp4`, `.avi`, `.mov`, `.mkv` 등

### 2. 출력 디렉토리 경로
- **타입**: `str` (절대 경로)
- **형식**: 디렉토리 경로 (존재하지 않으면 생성)
- **예시**: `/tmp/hand_server/translations/task_456/`
- **요구사항**:
  - 디렉토리가 없으면 자동 생성
  - 쓰기 권한 필요

### 3. 진행 상황 콜백 (선택사항)
- **타입**: `Callable[[int, str], None]` 또는 `None`
- **파라미터**:
  - `progress`: 0-100 사이의 정수
  - `message`: 진행 상황 설명 문자열
- **예시**:
```python
def progress_callback(progress: int, message: str):
    print(f"{progress}%: {message}")
    # 백엔드에서 SSE로 전송할 수 있도록 호출
```

---

## 출력 사양

### TranslationResult 객체 구조

```python
@dataclass
class TranslationResult:
    """번역 결과를 담는 데이터 클래스"""
    success: bool
    word: str  # 번역된 단어
    keypoints: List[FrameKeypoints]  # 프레임별 키포인트 리스트
    video_info: VideoInfo  # 동영상 메타정보
    annotated_video_path: str  # 키포인트가 그려진 동영상 경로
    processing_time: float  # 처리 시간 (초)
    error: Optional[str] = None  # 에러 메시지 (실패 시)

# 또는 Dict 형식으로 반환 가능
# {
#     "success": bool,
#     "word": str,
#     "keypoints": List[Dict],  # 각 프레임의 키포인트 딕셔너리
#     "video_info": Dict,
#     "annotated_video_path": str,
#     "processing_time": float,
#     "error": Optional[str]
# }
```

### FrameKeypoints 구조

```python
@dataclass
class FrameKeypoints:
    """한 프레임의 키포인트 정보 (OpenPose 형식)"""
    frame_index: int  # 프레임 인덱스 (0부터 시작)
    timestamp: float  # 타임스탬프 (초)
    
    # 각 키포인트는 [x, y, confidence] 형식의 리스트
    # x, y: 픽셀 좌표 (원본 동영상 해상도 기준)
    # confidence: 신뢰도 (0.0-1.0)
    
    pose: List[List[float]]  # 포즈 키포인트 (25개) - 각각 [x, y, confidence]
    face: List[List[float]]  # 얼굴 키포인트 (70개) - 각각 [x, y, confidence]
    hand_left: List[List[float]]  # 왼손 키포인트 (21개) - 각각 [x, y, confidence]
    hand_right: List[List[float]]  # 오른손 키포인트 (21개) - 각각 [x, y, confidence]
    
    # 감지되지 않은 경우 빈 리스트 []
    # 예: hand_left가 감지되지 않으면 hand_left = []
```

### 키포인트 데이터 형식

각 키포인트 그룹은 다음과 같은 형식입니다:

```python
# 포즈 키포인트 예시 (25개)
pose = [
    [320.5, 240.2, 0.98],  # 키포인트 0: [x, y, confidence]
    [350.1, 250.3, 0.95],  # 키포인트 1
    # ... 총 25개
]

# 얼굴 키포인트 예시 (70개)
face = [
    [400.0, 300.0, 0.97],  # 키포인트 0
    [410.0, 305.0, 0.96],  # 키포인트 1
    # ... 총 70개
]

# 손 키포인트 예시 (21개)
hand_left = [
    [200.0, 400.0, 0.99],  # 손목
    [210.0, 410.0, 0.98],  # 엄지 CMC
    # ... 총 21개
]

hand_right = [
    [600.0, 400.0, 0.99],  # 손목
    [590.0, 410.0, 0.98],  # 엄지 CMC
    # ... 총 21개
]
```

**참고**: 
- OpenPose 형식을 따릅니다
- `datum.poseKeypoints`: (사람 수, 25, 3) → 첫 번째 사람만 사용: `[0]`
- `datum.faceKeypoints`: (사람 수, 70, 3) → 첫 번째 사람만 사용: `[0]`
- `datum.handKeypoints[0]`: 왼손 (사람 수, 21, 3) → 첫 번째 사람만 사용: `[0]`
- `datum.handKeypoints[1]`: 오른손 (사람 수, 21, 3) → 첫 번째 사람만 사용: `[0]`

### VideoInfo 구조

```python
@dataclass
class VideoInfo:
    """동영상 메타정보"""
    width: int  # 동영상 너비 (픽셀)
    height: int  # 동영상 높이 (픽셀)
    fps: float  # 초당 프레임 수
    duration: float  # 동영상 길이 (초)
    total_frames: int  # 전체 프레임 수
    codec: str  # 비디오 코덱 (예: "h264", "vp9")
```

---

## JSON 출력 형식 (대안)

함수가 JSON 파일로 결과를 저장하는 경우:

### 파일 위치
- `{output_dir}/result.json`

### JSON 구조

```json
{
  "success": true,
  "word": "안녕하세요",
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "duration": 3.5,
    "total_frames": 105,
    "codec": "h264"
  },
  "keypoints": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "pose": [
        [320.5, 240.2, 0.98],
        [350.1, 250.3, 0.95],
        [380.2, 260.1, 0.97]
      ],
      "face": [
        [400.0, 300.0, 0.97],
        [410.0, 305.0, 0.96]
      ],
      "hand_left": [
        [200.0, 400.0, 0.99],
        [210.0, 410.0, 0.98],
        [220.0, 420.0, 0.97]
      ],
      "hand_right": [
        [600.0, 400.0, 0.99],
        [590.0, 410.0, 0.98],
        [580.0, 420.0, 0.97]
      ]
    },
    {
      "frame_index": 1,
      "timestamp": 0.033,
      "pose": [
        [322.1, 241.5, 0.97],
        [352.3, 251.2, 0.94]
      ],
      "face": [],
      "hand_left": [
        [202.1, 401.5, 0.98]
      ],
      "hand_right": []
    }
  ],
  "annotated_video_path": "/tmp/hand_server/translations/task_456/annotated.mp4",
  "processing_time": 12.5,
  "error": null
}
```

**참고**:
- 각 키포인트는 `[x, y, confidence]` 형식의 3개 요소 리스트
- `pose`: 최대 25개 키포인트 (각각 [x, y, confidence])
- `face`: 최대 70개 키포인트 (각각 [x, y, confidence])
- `hand_left`: 최대 21개 키포인트 (각각 [x, y, confidence])
- `hand_right`: 최대 21개 키포인트 (각각 [x, y, confidence])
- 감지되지 않은 경우 빈 리스트 `[]`
- 예시는 일부만 표시 (실제로는 모든 키포인트 포함)

---

## 키포인트 인덱스 매핑

OpenPose 형식을 따르므로, 리스트 인덱스로 키포인트를 식별합니다.

### 포즈 키포인트 (Pose) - 25개
인덱스 순서 (OpenPose BODY_25 형식):
```
0:  Nose
1:  Neck
2:  Right Shoulder
3:  Right Elbow
4:  Right Wrist
5:  Left Shoulder
6:  Left Elbow
7:  Left Wrist
8:  Mid Hip
9:  Right Hip
10: Right Knee
11: Right Ankle
12: Left Hip
13: Left Knee
14: Left Ankle
15: Right Eye
16: Left Eye
17: Right Ear
18: Left Ear
19: Left Big Toe
20: Left Small Toe
21: Left Heel
22: Right Big Toe
23: Right Small Toe
24: Right Heel
```

### 얼굴 키포인트 (Face) - 70개
OpenPose 얼굴 랜드마크 형식 (0-69 인덱스)

### 손 키포인트 (Hand) - 21개 (왼손/오른손 각각)
인덱스 순서 (OpenPose HAND 형식):
```
0:  Wrist
1:  Thumb CMC
2:  Thumb MCP
3:  Thumb IP
4:  Thumb Tip
5:  Index MCP
6:  Index PIP
7:  Index DIP
8:  Index Tip
9:  Middle MCP
10: Middle PIP
11: Middle DIP
12: Middle Tip
13: Ring MCP
14: Ring PIP
15: Ring DIP
16: Ring Tip
17: Pinky MCP
18: Pinky PIP
19: Pinky DIP
20: Pinky Tip
```

**사용 예시**:
```python
# 프레임 0의 왼손 손목 키포인트 접근
frame_0 = keypoints[0]
wrist = frame_0.hand_left[0]  # [x, y, confidence]

# 프레임 0의 오른손 엄지 끝 키포인트 접근
thumb_tip = frame_0.hand_right[4]  # [x, y, confidence]

# 포즈의 코 키포인트 접근
nose = frame_0.pose[0]  # [x, y, confidence]
```

**참고**: 
- 감지되지 않은 키포인트는 빈 리스트 `[]`로 반환
- 또는 해당 키포인트 그룹 전체가 빈 리스트일 수 있음
- 각 키포인트의 `confidence` 값이 낮으면 (예: < 0.3) 신뢰도 낮은 것으로 간주 가능

---

## 좌표계 규칙

### 2D 좌표 (x, y)
- **원점**: 동영상 프레임의 **왼쪽 상단 모서리**
- **X축**: 왼쪽에서 오른쪽으로 증가 (0 ~ width-1)
- **Y축**: 위에서 아래로 증가 (0 ~ height-1)
- **단위**: 픽셀 (부동소수점)
- **형식**: 각 키포인트는 `[x, y, confidence]` 리스트
  - `x`: X 좌표 (픽셀)
  - `y`: Y 좌표 (픽셀)
  - `confidence`: 신뢰도 (0.0-1.0)

**참고**: OpenPose는 3D 좌표를 제공하지 않으므로, 이 인터페이스는 2D 좌표만 사용합니다.

---

## 출력 파일

### 1. 주석 동영상 (Annotated Video)
- **경로**: `{output_dir}/annotated.mp4`
- **형식**: MP4 (H.264 코덱 권장)
- **내용**: 원본 동영상에 키포인트가 그려진 동영상
- **해상도**: 원본과 동일하거나 다를 수 있음 (변경 시 VideoInfo에 반영)

### 2. 결과 JSON (선택사항)
- **경로**: `{output_dir}/result.json`
- **형식**: UTF-8 인코딩 JSON
- **내용**: 위의 JSON 구조

---

## 에러 처리

### 예외 타입

```python
class VideoProcessingError(Exception):
    """동영상 처리 중 발생하는 오류"""
    pass

class ModelInferenceError(Exception):
    """모델 추론 중 발생하는 오류"""
    pass

class KeypointExtractionError(Exception):
    """키포인트 추출 중 발생하는 오류"""
    pass
```

### 에러 응답 형식

함수가 예외를 발생시키지 않고 결과 객체로 반환하는 경우:

```python
TranslationResult(
    success=False,
    word="",
    keypoints=[],
    video_info=None,
    annotated_video_path="",
    processing_time=0.0,
    error="동영상 파일을 읽을 수 없습니다: 파일이 손상되었습니다"
)
```

---

## 진행 상황 콜백 사용 예시

```python
def translate_with_progress(video_path: str, output_dir: str):
    """진행 상황을 콜백으로 전달하는 예시"""
    
    def on_progress(progress: int, message: str):
        # 백엔드에서 SSE로 클라이언트에 전송
        send_sse_progress(task_id, progress, message)
    
    result = translate_sign_language(
        video_path=video_path,
        output_dir=output_dir,
        progress_callback=on_progress
    )
    
    return result
```

### 진행 상황 단계 제안

```python
# 0-10%: 동영상 로드
progress_callback(5, "동영상 파일 로드 중...")

# 10-30%: 프레임 추출
progress_callback(15, "프레임 추출 중...")
progress_callback(25, "프레임 추출 완료")

# 30-60%: 키포인트 추출
progress_callback(35, "키포인트 추출 중...")
progress_callback(50, "키포인트 추출 완료")

# 60-80%: 모델 추론
progress_callback(65, "수어 인식 모델 실행 중...")
progress_callback(75, "모델 추론 완료")

# 80-95%: 키포인트 그리기
progress_callback(85, "키포인트 주석 추가 중...")
progress_callback(90, "동영상 인코딩 중...")

# 95-100%: 완료
progress_callback(100, "번역 완료")
```

---

## 실제 데이터 변환 예시

모델/파이프라인에서 OpenPose `datum` 객체를 이 인터페이스 형식으로 변환:

```python
def convert_datum_to_frame_keypoints(datum, frame_index: int, timestamp: float) -> FrameKeypoints:
    """OpenPose datum 객체를 FrameKeypoints로 변환"""
    
    # 포즈 키포인트 변환 (25개)
    pose = datum.poseKeypoints[0].tolist() if (
        datum.poseKeypoints is not None and 
        datum.poseKeypoints.shape[0] > 0
    ) else []
    
    # 얼굴 키포인트 변환 (70개)
    face = datum.faceKeypoints[0].tolist() if (
        datum.faceKeypoints is not None and 
        datum.faceKeypoints.shape[0] > 0
    ) else []
    
    # 왼손 키포인트 변환 (21개)
    hand_left = datum.handKeypoints[0][0].tolist() if (
        datum.handKeypoints[0] is not None and 
        datum.handKeypoints[0].shape[0] > 0
    ) else []
    
    # 오른손 키포인트 변환 (21개)
    hand_right = datum.handKeypoints[1][0].tolist() if (
        datum.handKeypoints[1] is not None and 
        datum.handKeypoints[1].shape[0] > 0
    ) else []
    
    return FrameKeypoints(
        frame_index=frame_index,
        timestamp=timestamp,
        pose=pose,
        face=face,
        hand_left=hand_left,
        hand_right=hand_right
    )
```

## 백엔드 통합 예시

### Python 백엔드 (Flask/FastAPI)

```python
from model_pipeline import translate_sign_language, TranslationResult
import os
import json

def process_translation_task(task_id: str, file_id: str):
    """번역 작업을 처리하는 백엔드 함수"""
    
    # 파일 경로 구성
    video_path = f"/tmp/hand_server/uploads/{file_id}.mp4"
    output_dir = f"/tmp/hand_server/translations/{task_id}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 진행 상황 콜백
    def progress_callback(progress: int, message: str):
        # SSE로 클라이언트에 전송
        send_sse_event(task_id, "progress", {
            "progress": progress,
            "message": message
        })
    
    try:
        # 모델/파이프라인 호출
        result: TranslationResult = translate_sign_language(
            video_path=video_path,
            output_dir=output_dir,
            progress_callback=progress_callback
        )
        
        if result.success:
            # 성공 시 결과 저장 (JSON 형식으로 저장)
            save_translation_result(task_id, result)
            
            # 완료 이벤트 전송
            send_sse_event(task_id, "complete", {
                "progress": 100,
                "word": result.word,
                "annotated_video_url": f"/api/video/annotated/{task_id}"
            })
        else:
            # 실패 시 에러 전송
            send_sse_event(task_id, "error", {
                "message": result.error
            })
            
    except Exception as e:
        # 예외 처리
        send_sse_event(task_id, "error", {
            "message": f"번역 중 오류 발생: {str(e)}"
        })

def save_translation_result(task_id: str, result: TranslationResult):
    """번역 결과를 JSON 파일로 저장"""
    output_dir = f"/tmp/hand_server/translations/{task_id}/"
    
    result_dict = {
        "success": result.success,
        "word": result.word,
        "video_info": {
            "width": result.video_info.width,
            "height": result.video_info.height,
            "fps": result.video_info.fps,
            "duration": result.video_info.duration,
            "total_frames": result.video_info.total_frames,
            "codec": result.video_info.codec
        },
        "keypoints": [
            {
                "frame_index": kp.frame_index,
                "timestamp": kp.timestamp,
                "pose": kp.pose,
                "face": kp.face,
                "hand_left": kp.hand_left,
                "hand_right": kp.hand_right
            }
            for kp in result.keypoints
        ],
        "annotated_video_path": result.annotated_video_path,
        "processing_time": result.processing_time,
        "error": result.error
    }
    
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
```

---

## 테스트용 더미 함수

백엔드 개발자가 모델이 준비되기 전에 테스트할 수 있는 더미 함수:

```python
import time
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable

@dataclass
class FrameKeypoints:
    frame_index: int
    timestamp: float
    pose: List[List[float]]  # [[x, y, confidence], ...] 최대 25개
    face: List[List[float]]  # [[x, y, confidence], ...] 최대 70개
    hand_left: List[List[float]]  # [[x, y, confidence], ...] 최대 21개
    hand_right: List[List[float]]  # [[x, y, confidence], ...] 최대 21개

@dataclass
class VideoInfo:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    duration: float = 3.0
    total_frames: int = 90
    codec: str = "h264"

@dataclass
class TranslationResult:
    success: bool
    word: str
    keypoints: List[FrameKeypoints]
    video_info: VideoInfo
    annotated_video_path: str
    processing_time: float
    error: Optional[str] = None

def translate_sign_language_dummy(
    video_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> TranslationResult:
    """테스트용 더미 함수 (OpenPose 형식)"""
    
    # 진행 상황 시뮬레이션
    if progress_callback:
        progress_callback(10, "동영상 로드 중...")
        time.sleep(0.5)
        progress_callback(30, "키포인트 추출 중...")
        time.sleep(1.0)
        progress_callback(60, "모델 추론 중...")
        time.sleep(1.0)
        progress_callback(90, "결과 생성 중...")
        time.sleep(0.5)
        progress_callback(100, "완료")
    
    # 더미 키포인트 데이터 생성 (OpenPose 형식)
    keypoints = []
    for i in range(90):  # 90 프레임
        # 더미 포즈 키포인트 (25개 중 일부만 생성)
        pose = [
            [400.0 + i * 0.1, 300.0 + i * 0.1, 0.98],  # Nose (인덱스 0)
            [400.0 + i * 0.1, 350.0 + i * 0.1, 0.97],  # Neck (인덱스 1)
            [450.0 + i * 0.1, 400.0 + i * 0.1, 0.96],  # Right Shoulder (인덱스 2)
            [500.0 + i * 0.1, 450.0 + i * 0.1, 0.95],  # Right Elbow (인덱스 3)
            [550.0 + i * 0.1, 500.0 + i * 0.1, 0.94],  # Right Wrist (인덱스 4)
        ]
        
        # 더미 얼굴 키포인트 (70개 중 일부만 생성)
        face = [
            [400.0 + i * 0.1, 300.0 + i * 0.1, 0.97],
            [410.0 + i * 0.1, 305.0 + i * 0.1, 0.96],
            [390.0 + i * 0.1, 305.0 + i * 0.1, 0.96],
        ]
        
        # 더미 왼손 키포인트 (21개 중 일부만 생성)
        hand_left = [
            [300.0 + i * 0.1, 500.0 + i * 0.1, 0.99],  # Wrist (인덱스 0)
            [310.0 + i * 0.1, 510.0 + i * 0.1, 0.98],  # Thumb CMC (인덱스 1)
            [320.0 + i * 0.1, 520.0 + i * 0.1, 0.97],  # Thumb MCP (인덱스 2)
        ]
        
        # 더미 오른손 키포인트 (21개 중 일부만 생성)
        hand_right = [
            [500.0 - i * 0.1, 500.0 + i * 0.1, 0.99],  # Wrist (인덱스 0)
            [490.0 - i * 0.1, 510.0 + i * 0.1, 0.98],  # Thumb CMC (인덱스 1)
            [480.0 - i * 0.1, 520.0 + i * 0.1, 0.97],  # Thumb MCP (인덱스 2)
        ]
        
        frame_kp = FrameKeypoints(
            frame_index=i,
            timestamp=i / 30.0,
            pose=pose,
            face=face,
            hand_left=hand_left,
            hand_right=hand_right
        )
        keypoints.append(frame_kp)
    
    # 결과 생성
    result = TranslationResult(
        success=True,
        word="안녕하세요",
        keypoints=keypoints,
        video_info=VideoInfo(),
        annotated_video_path=os.path.join(output_dir, "annotated.mp4"),
        processing_time=3.0
    )
    
    # JSON 저장 (선택사항)
    os.makedirs(output_dir, exist_ok=True)
    result_dict = {
        "success": result.success,
        "word": result.word,
        "video_info": asdict(result.video_info),
        "keypoints": [
            {
                "frame_index": kp.frame_index,
                "timestamp": kp.timestamp,
                "pose": kp.pose,
                "face": kp.face,
                "hand_left": kp.hand_left,
                "hand_right": kp.hand_right
            }
            for kp in result.keypoints
        ],
        "annotated_video_path": result.annotated_video_path,
        "processing_time": result.processing_time,
        "error": result.error
    }
    
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    return result
```

---

## 체크리스트

### 모델/파이프라인 개발자
- [ ] 함수 시그니처 구현
- [ ] 입력 파일 경로 검증
- [ ] 출력 디렉토리 생성
- [ ] 키포인트 추출 및 형식 준수
- [ ] 진행 상황 콜백 호출
- [ ] 주석 동영상 생성
- [ ] JSON 결과 저장 (선택사항)
- [ ] 에러 처리 및 예외 발생

### 백엔드 개발자
- [ ] 함수 호출 코드 작성
- [ ] 진행 상황 콜백을 SSE로 변환
- [ ] 결과 파싱 및 저장
- [ ] 에러 핸들링
- [ ] 더미 함수로 테스트

---

## 추가 고려사항

### 성능
- 대용량 동영상 처리 시간 고려
- 메모리 사용량 최적화
- 배치 처리 지원 (선택사항)

### 확장성
- 여러 모델 버전 지원
- 설정 파일로 파라미터 조정
- 로깅 시스템 연동

### 호환성
- Python 버전 명시 (예: Python 3.8+)
- 의존성 패키지 목록 제공
- 가상 환경 설정 가이드

