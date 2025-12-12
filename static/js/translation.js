import { state, elements } from './state.js';
import { showToast, showError, setStep, fetchAPI, navigateTo } from './utils.js';

export function initTranslation() {
    // 1. Mode Toggles
    elements.btnModeFile?.addEventListener('click', () => toggleInputMode('file'));
    elements.btnModeCam?.addEventListener('click', () => toggleInputMode('cam'));
    elements.uploadStartButton?.addEventListener('click', uploadVideoToServer);

    // 2. File Input & Drag-Drop
    if (elements.dropArea) {
        elements.dropArea.addEventListener('click', () => elements.fileInput.click());
        setupDragAndDrop();
    }
    elements.fileInput?.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));

    // 3. Webcam Controls
    elements.btnStartRecord?.addEventListener('click', startRecording);
    elements.btnStopRecord?.addEventListener('click', stopRecording);

    // 4. Keypoint Toggle
    elements.keypointToggle?.addEventListener('change', handleKeypointToggle);

    // 5/ go translation
    elements.go_translation?.addEventListener('click', () => navigateTo('translation'));
}

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
    area.addEventListener('drop', e => handleFileSelect(e.dataTransfer.files[0]));
}

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
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
            state.webcamStream = stream;
            elements.videoPreview.srcObject = stream;
            elements.videoPreview.classList.remove('hidden');
            elements.camPlaceholder.classList.add('hidden');
            elements.btnStartRecord.disabled = false;
            elements.camStatusText.textContent = '녹화 준비 완료';
        })
        .catch(() => {
            showError('카메라 권한이 필요합니다.');
            elements.camStatusText.textContent = '카메라 연결 실패';
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
    if (!state.webcamStream) return;
    state.recordedChunks = [];
    state.mediaRecorder = new MediaRecorder(state.webcamStream);

    state.mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) state.recordedChunks.push(e.data);
    };

    state.mediaRecorder.onstop = () => {
        const blob = new Blob(state.recordedChunks, { type: 'video/mp4' });
        const file = new File([blob], `cam_${Date.now()}.mp4`, { type: 'video/mp4' });
        toggleInputMode('file');
        handleFileSelect(file);
    };

    state.mediaRecorder.start();
    elements.btnStartRecord.classList.add('hidden');
    elements.btnStopRecord.classList.remove('hidden');
    elements.recIndicator.classList.remove('hidden');
    elements.camStatusText.textContent = '녹화 중...';
}

// --- 녹화 중지 함수 ---
function stopRecording() {
    if (state.mediaRecorder?.state === 'recording') state.mediaRecorder.stop();
    elements.btnStartRecord.classList.remove('hidden');
    elements.btnStopRecord.classList.add('hidden');
    elements.recIndicator.classList.add('hidden');
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
    player.src = annotatedUrl;

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