import { state, elements } from './state.js';
import { showToast, showError, setStep, fetchAPI } from './utils.js';

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

function handleFileSelect(file) {
    if (!file) return;
    if (!file.type.startsWith('video/')) return showError('동영상 파일만 가능합니다.');

    state.uploadedFile = { file: file, filename: file.name };
    if (elements.statusText) elements.statusText.textContent = file.name;
    if (elements.fileStatusBox) elements.fileStatusBox.classList.remove('hidden');
    showToast(`파일 선택됨: ${file.name}`);
}

// --- Webcam Logic ---
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

export function stopWebcamStream() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(t => t.stop());
        state.webcamStream = null;
    }
}

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

function stopRecording() {
    if (state.mediaRecorder?.state === 'recording') state.mediaRecorder.stop();
    elements.btnStartRecord.classList.remove('hidden');
    elements.btnStopRecord.classList.add('hidden');
    elements.recIndicator.classList.add('hidden');
}

// --- Server & Result ---
async function uploadVideoToServer() {
    if (!state.uploadedFile || !state.uploadedFile.file) return showError('파일이 없습니다.');

    setStep(2);
    elements.uploadStartButton.disabled = true;

    const formData = new FormData();
    formData.append('file', state.uploadedFile.file);

    try {
        // Upload
        const uploadData = await fetchAPI('/api/upload', { method: 'POST', body: formData });
        state.uploadedFile.fileId = uploadData.file_id;

        // Request Translation
        const transData = await fetchAPI('/api/translate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ file_id: uploadData.file_id })
        });

        monitorProgress(transData.task_id);

    } catch (e) {
        handleProcessError(e.message);
    }
}

function monitorProgress(taskId) {
    if (state.eventSource) state.eventSource.close();
    state.eventSource = new EventSource(`/api/translate/progress/${taskId}`);

    state.eventSource.addEventListener('progress', e => {
        const d = JSON.parse(e.data);
        if (elements.progressBar) elements.progressBar.style.width = `${d.progress}%`;
        if (elements.progressMessage) elements.progressMessage.textContent = `${d.progress}% - ${d.message}`;
    });

    state.eventSource.addEventListener('complete', e => {
        state.eventSource.close();
        const d = JSON.parse(e.data);
        handleCompletion(taskId, d.word);
    });

    state.eventSource.addEventListener('error', () => {
        state.eventSource.close();
        handleProcessError('서버 연결 중단');
    });
}

function handleCompletion(taskId, word) {
    if (elements.progressBar) elements.progressBar.style.width = '100%';
    if (elements.resultWord) elements.resultWord.textContent = word;

    const annotatedUrl = `/api/video/annotated/${taskId}`;
    const originalUrl = `/api/video/original/${taskId}`;

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

    setStep(3);
    player.classList.remove('hidden');
    if (elements.resultVideoPlaceholder) elements.resultVideoPlaceholder.classList.add('hidden');
    player.play().catch(() => {});

    elements.uploadStartButton.disabled = false;
    elements.uploadStartButton.textContent = '다시 번역하기';
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

function handleProcessError(msg) {
    showError(msg);
    setStep(1);
    elements.uploadStartButton.disabled = false;
}