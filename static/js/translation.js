import { state, elements } from './state.js';
import { showToast, showError, setStep, fetchAPI, navigateTo } from './utils.js';

export function initTranslation() {
    // 1. Mode Toggles
    elements.btnModeFile?.addEventListener('click', () => toggleInputMode('file')); // 입력 모드를 파일로 변경
    elements.btnModeCam?.addEventListener('click', () => {
        toggleInputMode('cam'); // 입력 모드를 웹캠으로 변경
        setIdleState(); // 웹캠 모드 진입 시 초기 상태로 설정
    });
    elements.uploadStartButton?.addEventListener('click', uploadVideoToServer);

    // 2. File Input & Drag-Drop
    if (elements.dropArea) {
        elements.dropArea.addEventListener('click', () => elements.fileInput.click());
        setupDragAndDrop();
    }
    elements.fileInput?.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));

    // 3. Webcam Controls

    // 녹화/중지 버튼
    elements.recordToggleBtn?.addEventListener('click', () => {
        if(state.recordedBlob) {
            console.log("Recorded state: Restarting webcam stream.")
            startWebcamStream();
        } else if (!state.isRecording) {
            console.log("녹화 시작.")
            startRecording(); // 녹화 시작
        } else {
            console.log("녹화 중지.")
            stopRecording(); // 녹화 중지
        }
    });
    // 단어 번역 버튼
    elements.translateBtn?.addEventListener('click', startTranslationFromWebcam);

    // 4. Keypoint Toggle
    elements.keypointToggle?.addEventListener('change', handleKeypointToggle);

    // 5/ go translation
    elements.go_translation?.addEventListener('click', () => navigateTo('translation'));
}

// 파일 업로드 모드 드랙그 앤 드랍
function setupDragAndDrop() {
    const area = elements.dropArea;
    ['dragenter', 'dragover'].forEach(evt => {
        area.addEventListener(evt, e => {
            e.preventDefault();
            area.classList.add('border-indigo-400', 'bg-indigo-50');
        });
    });
    ['dragleave', 'drop'].forEach(evt => {
        area.addEventListener(evt, e => {
            e.preventDefault();
            area.classList.remove('border-indigo-400', 'bg-indigo-50');
        });
    });
    area.addEventListener('drop', e => handleFileSelect(e.dataTransfer.files[0])); // 입력된 파일 처리 함수
}

// 파일 입력 모드 변경
function toggleInputMode(mode) {
    state.inputType = mode;
    const isFile = mode === 'file';

    // Update UI Styles
    updateModeButtonStyles(isFile);
    elements.modeFileArea.classList.toggle('hidden', !isFile);
    elements.modeCamArea.classList.toggle('hidden', isFile);

    isFile ? stopWebcamStream() : startWebcamStream();
}

function updateModeButtonStyles(isFile) {
    const setStyle = (el, active) => {
        el.classList.toggle('bg-white', active);
        el.classList.toggle('shadow-sm', active);
        el.classList.toggle('text-indigo-600', active);
    };
    setStyle(elements.btnModeFile, isFile);
    setStyle(elements.btnModeCam, !isFile);
}

// 파일 업로드 관련 함수

// --- 파일 입력 시 처리하는 함수 ---
function handleFileSelect(file) {
    if (!file) return;                                                                // 파일의 존재 확인
    if (!file.type.startsWith('video/')) return showError('동영상 파일만 가능합니다.'); // 입력된 파일이 비디오 타입인지 확인

    state.uploadedFile = { file: file, filename: file.name };
    // 상태 텍스트를 파일 이름으로
    if (elements.statusText) elements.statusText.textContent = file.name;
    // 파일 입력 시 파일 상태 박스 보이기
    if (elements.fileStatusBox) elements.fileStatusBox.classList.remove('hidden');
    // 파일 입력 시 업로드 시작 버튼 초기화
    if (elements.uploadStartButton) {
        elements.uploadStartButton.textContent = '번역하기';
        elements.uploadStartButton.disabled = false; // 업로드 시작 버튼을 활성화
    }
    showToast(`파일 선택됨: ${file.name}`);
}

// 녹화 관련 함수

// --- 웹캠 스트리밍 시작 함수 ---
function startWebcamStream() {
    // 1. 무한 루프 방지를 위해 녹화 데이터 초기화
    if (state.recordedBlob) {
        // 녹화 데이터와 파일 정보 초기화
        state.recordedBlob = null;
        state.uploadedFile = { file: null, filename: null }; 
    }

    // 2. 녹화 영상 URL 해제 및 UI 초기화
    // if (elements.videoPreview.src) {
    //     URL.revokeObjectURL(elements.videoPreview.src);
    //     elements.videoPreview.src = '';
    //     elements.videoPreview.loop = false;
    //     elements.videoPreview.srcObject = null;
    // }
    // 💡 2. 녹화 영상 URL 해제 및 UI 정리 (recordedVideoPlayer 정리)
    if (elements.recordedVideoPlayer && elements.recordedVideoPlayer.src) {
        URL.revokeObjectURL(elements.recordedVideoPlayer.src);
        elements.recordedVideoPlayer.src = '';
        elements.recordedVideoPlayer.pause();
        // 녹화 플레이어 숨김
        elements.recordedVideoPlayer.classList.add('hidden');
    }

    // 카메라 연결 시도 전 UI 상태 초기화
    elements.camStatusText.textContent = '카메라 연결 시도 중...';
    elements.camPlaceholder.classList.remove('hidden'); 
    elements.videoPreview.classList.add('hidden');
    elements.recordToggleBtn.disabled = true; // 시도 중 녹화 버튼 비활성화

    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
            state.webcamStream = stream; // 스트림 객체 할당
            elements.videoPreview.srcObject = stream; // 스트림 연결
            
            // 성공 시 UI 업데이트
            elements.videoPreview.classList.remove('hidden');
            elements.camPlaceholder.classList.add('hidden');
            elements.recordToggleBtn.disabled = false; // 녹화 시작 버튼 활성화
            elements.camStatusText.textContent = '녹화 준비 완료'; // 캠 상태 설명

            setIdleState(); // 👈 스트림 연결 성공 시 '녹화 시작' 상태로 전환
        })
        .catch(() => {
            // 실패 시 UI 업데이트
            state.webcamStream = null;
            elements.videoPreview.classList.add('hidden');
            elements.camPlaceholder.classList.remove('hidden');
            elements.recordToggleBtn.disabled = true; // 버튼 비활성화 유지
            elements.camStatusText.textContent = '카메라 연결 실패'; // 캠 상태 설명

            showError('카메라 권한이 필요합니다.');
        });
}

// --- 웹캠 스트리밍 중지 ---
export function stopWebcamStream() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(t => t.stop());
        state.webcamStream = null;
    }
}

window.stopWebcamStream = stopWebcamStream;

// --- 녹화 시작 함수 ---
function startRecording() {
    
    if (!state.webcamStream) {
        console.error("웹캠 스트림 (state.webcamStream)이 null 또는 undefined입니다.");
        showError("웹캠 스트림이 활성화되지 않았습니다. 녹화를 시작할 수 없습니다.");
        return; 
    }

    state.recordedChunks = [];
    try {
        state.mediaRecorder = new MediaRecorder(state.webcamStream); 
    } catch (e) {
        console.error("MediaRecorder 초기화 실패:", e);
        showError("브라우저가 동영상 녹화를 지원하지 않습니다.");
        return;
    }

    state.mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) state.recordedChunks.push(e.data);
    };

    state.mediaRecorder.onstop = () => {
        const blob = new Blob(state.recordedChunks, { type: 'video/mp4' });
        const fileObject = new File([blob], `cam_${Date.now()}.mp4`, { type: 'video/mp4' });

        // 1. 녹화된 Blob과 File을 state에 저장
        state.recordedBlob = blob; 
        state.uploadedFile = { file: fileObject, filename: fileObject.name };
        
        // 2. 웹캠 스트림 중지
        // 녹화 후 웹캠 스트림을 중지해야 녹화된 영상만 재생 가능
        state.webcamStream.getTracks().forEach(track => track.stop());

        // 3. 웹캠 영역에 녹화된 영상 연결 및 재생
        const videoUrl = URL.createObjectURL(blob);
        // elements.videoPreview.srcObject = null; // 기존 스트림 해제
        // elements.videoPreview.src = videoUrl;
        // elements.videoPreview.loop = true; // 반복 재생
        // elements.videoPreview.play().catch(e => console.error("Video play failed:", e));
        elements.recordedVideoPlayer.srcObject = null; 
        elements.recordedVideoPlayer.src = videoUrl;
        elements.recordedVideoPlayer.loop = true; // 반복 재생
        elements.recordedVideoPlayer.controls = true; // 컨트롤바 표시
    
        // 비디오 플레이어가 준비되면 재생
        elements.recordedVideoPlayer.play().catch(e => console.error("Video play failed:", e));

        // 4. 버튼 상태 변경
        setRecordedState(); // 녹화된 상태로
    };

    try {
        state.mediaRecorder.start();
        
        // 녹화 시작 성공 시 UI 업데이트 및 상태 변경
        elements.recIndicator.classList.remove('hidden');
        elements.camStatusText.textContent = '녹화 중...'; // 캠 상태 설명
        
        setRecordingState(); // 녹화 중 상태로
    } catch (e) {
        console.error("MediaRecorder.start() 실패:", e);
        showError("녹화 시작 중 오류가 발생했습니다.");
        setIdleState(); 
    }
}

// --- 녹화 중지 함수 ---
function stopRecording() {
    if (state.mediaRecorder?.state === 'recording') {
        state.mediaRecorder.stop();
        elements.recIndicator.classList.add('hidden');
        elements.camStatusText.textContent = '녹화 완료';
    } else {
        showToast('녹화 중이 아닙니다.', 'warning');
    }
}

// --- 녹화 버튼 상태 관리 함수 ---

/**
 * 버튼 상태를 "대기/재녹화" 상태(초기)로 설정합니다. (녹화 시작/단어 번역 비활성화)
 */
function setIdleState() {
    // 녹화 중지 상태
    state.isRecording = false;
    // 녹화/중지 버튼 UI
    elements.recordToggleBtn.innerHTML = '<i class="fas fa-video mr-2"></i> 녹화 시작';
    elements.recordToggleBtn.classList.replace('bg-red-500', 'bg-indigo-500');
    elements.recordToggleBtn.classList.replace('shadow-red-500/30', 'shadow-indigo-500/30');
    
    // 영상 번역 버튼 UI 및 비활성화
    elements.translateBtn.disabled = true;
    elements.translateBtn.classList.replace('bg-indigo-500', 'bg-slate-300');
    elements.translateBtn.classList.replace('text-white', 'text-slate-500');
    elements.translateBtn.classList.add('cursor-not-allowed');

    // 영상 플레이어 초기화(캠 프리뷰를 보여주고, 녹화 영상을 숨김)
    elements.videoPreview.classList.remove('hidden');
    elements.recordedVideoPlayer.classList.add('hidden');
    elements.recordedVideoPlayer.src = ''; // 영상 초기화
    elements.recordedVideoPlayer.srcObject = null;
}

/**
 * 버튼 상태를 "녹화 중" 상태로 설정합니다. (녹화 중지/단어 번역 비활성화)
 */
function setRecordingState() {
    // 녹화 상태
    state.isRecording = true;
    // 녹화/중지 버튼 UI
    elements.recordToggleBtn.innerHTML = '<i class="fas fa-stop-circle mr-2"></i> 녹화 중지';
    elements.recordToggleBtn.classList.replace('bg-indigo-500', 'bg-red-500');
    elements.recordToggleBtn.classList.replace('shadow-indigo-500/30', 'shadow-red-500/30');

    // 영상 번역 버튼 UI 및 비활성화
    elements.translateBtn.disabled = true;
    elements.translateBtn.classList.replace('bg-indigo-500', 'bg-slate-300');
    elements.translateBtn.classList.replace('text-white', 'text-slate-500');
    elements.translateBtn.classList.add('cursor-not-allowed');
}

/**
 * 버튼 상태를 "녹화 완료" 상태로 설정합니다. (재녹화 시작/단어 번역 활성화)
 */
function setRecordedState() {
    // 녹화 상태
    state.isRecording = false; // 녹화는 끝남
    // 녹화/중지 버튼 UI
    elements.recordToggleBtn.innerHTML = '<i class="fas fa-video mr-2"></i> 다시 녹화'; 
    elements.recordToggleBtn.classList.replace('bg-red-500', 'bg-indigo-500');
    elements.recordToggleBtn.classList.replace('shadow-red-500/30', 'shadow-indigo-500/30');

    // 영상 번역 버튼 UI 및 활성화
    elements.translateBtn.disabled = false;
    elements.translateBtn.classList.replace('bg-slate-300', 'bg-indigo-500');
    elements.translateBtn.classList.replace('text-slate-500', 'text-white');
    elements.translateBtn.classList.remove('cursor-not-allowed');

    elements.videoPreview.classList.add('hidden'); // 캠 프리뷰 숨기기
    elements.recordedVideoPlayer.classList.remove('hidden'); // 녹화된 영상 보이기
}

// --- 녹화 완료 후 번역 시작 함수 추가 ---

/**
 * 녹화된 웹캠 영상을 서버로 전송하고 번역을 요청합니다.
 */
function startTranslationFromWebcam() {
    console.log("--- startTranslationFromWebcam 호출됨 ---");

    if (!state.uploadedFile) {
        console.warn("uploadedFile이 없어 번역을 시작할 수 없습니다.");
        showError("녹화된 영상이 없습니다. 먼저 녹화를 완료해주세요.");
        return;
    }
    
    // 단어 번역 버튼 비활성화 (번역 중임을 표시)
    if (elements.translateBtn) {
        elements.translateBtn.disabled = true;
        elements.translateBtn.classList.replace('bg-indigo-500', 'bg-slate-300');
        elements.translateBtn.classList.add('cursor-not-allowed');
    }

    // Blob 데이터를 FormData로 변환하여 서버로 전송
    // const formData = new FormData();
    // formData.append('file', state.uploadedFile, state.uploadedFile.name); 

    // uploadVideoToServer는 FormData를 받도록 구현되어 있어야 합니다.
    uploadVideoToServer();
}

// 번역 작업 2단계

// --- 서버에 영상을 업로드 하는 함수 ---
async function uploadVideoToServer() {
    if (!state.uploadedFile || !state.uploadedFile.file) return showError('파일이 없습니다.');

    setStep(2); // 번역 작업을 2단계로

    elements.uploadStartButton.disabled = true; // 업로드 시작 버튼을 비활성와

    const formData = new FormData();
    formData.append('file', state.uploadedFile.file);

    try {
        // 업로드 Route
        const uploadData = await fetchAPI('/api/upload', { method: 'POST', body: formData });
        state.uploadedFile.fileId = uploadData.file_id;

        // 번역 요청 Route
        const transData = await fetchAPI('/api/translate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ file_id: uploadData.file_id })
        });

        monitorProgress(transData.task_id); // 작업 과정 모니터링 함수 호출

    } catch (e) {
        handleProcessError(e.message); // 작업 과정 중 오류 발생 시 호출하는 함수
    }
}

// --- 작업과정 모니터링 함수 ---
function monitorProgress(taskId) {
    if (state.eventSource) state.eventSource.close();
    state.eventSource = new EventSource(`/api/translate/progress/${taskId}`); // 모니터링 Route

    // 작업 중
    state.eventSource.addEventListener('progress', e => {
        const d = JSON.parse(e.data);
        if (elements.progressBar) elements.progressBar.style.width = `${d.progress}%`;
        if (elements.progressMessage) elements.progressMessage.textContent = `${d.progress}% - ${d.message}`;
    });

    // 작업 완료
    state.eventSource.addEventListener('complete', e => {
        state.eventSource.close();
        const d = JSON.parse(e.data);
        handleCompletion(taskId, d.word); // 작업 완료되었으니 다음 함수 호출
    });

    // 작업 오류
    state.eventSource.addEventListener('error', () => {
        state.eventSource.close();
        handleProcessError('서버 연결 중단');  // 작업 과정 중 오류 발생 시 호출하는 함수
    });
}

// 번역 작업 3단계

// 번역 작업이 완료 시 작업하는 함수
function handleCompletion(taskId, word) {
    if (elements.progressBar) elements.progressBar.style.width = '100%';
    if (elements.resultWord) elements.resultWord.textContent = word;

    const annotatedUrl = `/api/video/annotated/${taskId}`; // 작업된 영상 위치
    const originalUrl = `/api/video/original/${taskId}`; // 원본 영상 위치

    const player = elements.resultVideoPlayer;
    player.dataset.annotatedUrl = annotatedUrl;
    player.dataset.originalUrl = originalUrl;
    player.src = originalUrl;

    if (elements.keypointToggle) {
        elements.keypointToggle.checked = false;
    }

    // Save History
    state.fileHistory.unshift({
        type: 'translation',
        filename: state.uploadedFile.filename || 'Webcam Video',
        date: new Date().toLocaleTimeString(),
        url: annotatedUrl,
        word: word
    });

    setStep(3); // 번역 작업을 3단계로

    player.classList.remove('hidden');
    if (elements.resultVideoPlaceholder) elements.resultVideoPlaceholder.classList.add('hidden');
    player.play().catch(() => {});

    if (elements.uploadStartButton) {
        elements.uploadStartButton.disabled = false; // 업로드 시작 버튼 활성화
        elements.uploadStartButton.textContent = '다시 번역하기'; // 이미 작업해본 파일이 그대로 있으니 텍스트를 '다시 번역하기'로 변경
    }
}

function handleKeypointToggle(e) {
    const player = elements.resultVideoPlayer;
    if (!player.dataset.annotatedUrl) return;

    const currentTime = player.currentTime;
    const wasPlaying = !player.paused;

    player.src = e.target.checked ? player.dataset.annotatedUrl : player.dataset.originalUrl;
    player.currentTime = currentTime;
    if (wasPlaying) player.play();
}

// 작업 오류러 중단하는 함수
function handleProcessError(msg) {

    showError(msg); // 에러 메시지

    setStep(1); // 1단계로 초기화

    elements.uploadStartButton.disabled = false; // 업로드 시작 버튼 활성화
}