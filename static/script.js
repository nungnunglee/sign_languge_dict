const state = {
    uploadedFile: { fileId: null, filename: null },
    currentTaskId: null,
    inputType: 'file',
    webcamStream: null,
    mediaRecorder: null,
    recordedChunks: [],
    recordedBlob: null,
    eventSource: null,
    showKeypoints: false,
};

const elements = {
    // 화면 요소
    introScreen: document.getElementById('intro-screen'),
    translationApp: document.getElementById('translation-app'),
    dictionaryApp: document.getElementById('dictionary-app'),
    gameApp: document.getElementById('game-app'),

    // 카드 메뉴
    cardTranslate: document.getElementById('card-translate'),
    cardDictionary: document.getElementById('card-dictionary'),
    cardGame: document.getElementById('card-game'),

    // 번역 앱
    btnModeFile: document.getElementById('btn-mode-file'),
    btnModeCam: document.getElementById('btn-mode-cam'),
    modeFileArea: document.getElementById('mode-file-area'),
    modeCamArea: document.getElementById('mode-cam-area'),
    dropArea: document.getElementById('drop-area'),
    fileInput: document.getElementById('video-file-input'),
    statusText: document.querySelector('.status-text span'),
    videoPreview: document.getElementById('cam-preview'),
    btnStartRecord: document.getElementById('btn-start-record'),
    btnStopRecord: document.getElementById('btn-stop-record'),
    recIndicator: document.getElementById('recording-indicator'),
    camStatusText: document.getElementById('cam-status-text'),
    uploadButton: document.getElementById('upload-button'),
    step1: document.getElementById('step-upload'),
    step2: document.getElementById('step-translate'),
    step3: document.getElementById('step-result'),
    nav1: document.getElementById('nav-1'),
    nav2: document.getElementById('nav-2'),
    nav3: document.getElementById('nav-3'),
    taskIdDisplay: document.getElementById('task-id-display'),
    progressBarFill: document.getElementById('progress-bar-fill'),
    progressMessage: document.getElementById('progress-message'),
    translationWord: document.getElementById('translation-word'),

    // 결과 화면 요소
    resultVideoPlaceholder: document.getElementById('result-video-placeholder'),
    keypointToggle: document.getElementById('keypoint-toggle'),
    btnSearchDict: document.getElementById('btn-search-dict'), // [신규]

    // 사전 앱
    dictSearchInput: document.getElementById('dict-search-input'),
    dictSearchBtn: document.getElementById('dict-search-btn'),
    dictResultList: document.getElementById('dict-result-list'),
    dictVideoArea: document.getElementById('dict-video-area'),
    dictPlayingWord: document.getElementById('dict-playing-word'),

    // 토스트
    toastContainer: document.getElementById('toast-container'),
};

// ==========================================
// 1. 네비게이션 & 초기화
// ==========================================

function hideAllApps() {
    elements.introScreen.classList.add('hidden');
    elements.translationApp.classList.add('hidden');
    elements.dictionaryApp.classList.add('hidden');
    elements.gameApp.classList.add('hidden');
    stopWebcamStream();
}

elements.cardTranslate.addEventListener('click', () => {
    hideAllApps();
    elements.translationApp.classList.remove('hidden');
    switchMode('file');
});

elements.cardDictionary.addEventListener('click', () => {
    hideAllApps();
    elements.dictionaryApp.classList.remove('hidden');
    elements.dictSearchInput.value = '';
    elements.dictResultList.innerHTML = '<div class="empty-state">검색어를 입력해주세요.</div>';
    elements.dictVideoArea.classList.add('hidden');
});

elements.cardGame.addEventListener('click', () => {
    hideAllApps();
    elements.gameApp.classList.remove('hidden');
});

window.goHome = function() {
    hideAllApps();
    elements.introScreen.classList.remove('hidden');
    resetUploadState();
    goToStep(1);
    if(state.eventSource) state.eventSource.close();
};

function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'toast error';
    toast.innerHTML = `⚠️ <span>${message}</span>`;
    elements.toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ==========================================
// 2. 번역 앱 로직
// ==========================================
function resetUploadState() {
    elements.uploadButton.disabled = true;
    elements.uploadButton.textContent = "다음 단계로";
    elements.statusText.textContent = "여기를 클릭하여 파일 선택";
    state.recordedBlob = null;
    elements.fileInput.value = "";
}

elements.btnModeFile.addEventListener('click', () => switchMode('file'));
elements.btnModeCam.addEventListener('click', () => switchMode('cam'));
async function switchMode(mode) {
    state.inputType = mode;
    resetUploadState();
    if (mode === 'file') {
        elements.btnModeFile.classList.add('active');
        elements.btnModeCam.classList.remove('active');
        elements.modeFileArea.classList.remove('hidden');
        elements.modeCamArea.classList.add('hidden');
        stopWebcamStream();
    } else {
        elements.btnModeFile.classList.remove('active');
        elements.btnModeCam.classList.add('active');
        elements.modeFileArea.classList.add('hidden');
        elements.modeCamArea.classList.remove('hidden');
        await initWebcam();
    }
}
async function initWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        state.webcamStream = stream;
        elements.videoPreview.srcObject = stream;
        elements.btnStartRecord.disabled = false;
        elements.btnStartRecord.classList.remove('hidden');
        elements.btnStopRecord.classList.add('hidden');
        elements.camStatusText.textContent = "준비됨";
    } catch (err) {
        showError("카메라 접근 권한이 필요합니다.");
    }
}
function stopWebcamStream() {
    if (state.webcamStream) {
        state.webcamStream.getTracks().forEach(track => track.stop());
        state.webcamStream = null;
        elements.videoPreview.srcObject = null;
    }
}
elements.btnStartRecord.addEventListener('click', () => {
    state.recordedChunks = [];
    try {
        state.mediaRecorder = new MediaRecorder(state.webcamStream, { mimeType: 'video/webm' });
    } catch(e) { state.mediaRecorder = new MediaRecorder(state.webcamStream); }

    state.mediaRecorder.ondataavailable = (e) => { if(e.data.size > 0) state.recordedChunks.push(e.data); };
    state.mediaRecorder.onstop = () => {
        state.recordedBlob = new Blob(state.recordedChunks, { type: 'video/webm' });
        elements.camStatusText.textContent = "촬영 완료";
        elements.uploadButton.disabled = false;
    };
    state.mediaRecorder.start();
    elements.btnStartRecord.classList.add('hidden');
    elements.btnStopRecord.classList.remove('hidden');
    elements.recIndicator.classList.remove('hidden');
});
elements.btnStopRecord.addEventListener('click', () => {
    state.mediaRecorder.stop();
    elements.btnStartRecord.classList.remove('hidden');
    elements.btnStopRecord.classList.add('hidden');
    elements.recIndicator.classList.add('hidden');
});
elements.dropArea.addEventListener('click', () => elements.fileInput.click());
elements.fileInput.addEventListener('change', () => {
    if (elements.fileInput.files.length > 0) {
        elements.statusText.textContent = elements.fileInput.files[0].name;
        elements.uploadButton.disabled = false;
    }
});

elements.uploadButton.addEventListener('click', handleUpload);

async function handleUpload() {
    let fileToUpload = state.inputType === 'file' ? elements.fileInput.files[0] : new File([state.recordedBlob], "webcam.webm");
    if (!fileToUpload) return showError("파일이나 영상이 없습니다.");

    elements.uploadButton.textContent = "업로드 중...";
    elements.uploadButton.disabled = true;
    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
            state.uploadedFile = { fileId: data.file_id };
            goToStep(2);
            startTranslation();
            stopWebcamStream();
        } else { throw new Error(data.error); }
    } catch (e) {
        showError("업로드 실패: " + e.message);
        elements.uploadButton.disabled = false;
        elements.uploadButton.textContent = "다시 시도";
    }
}

function goToStep(step) {
    [elements.step1, elements.step2, elements.step3].forEach(el => el.classList.remove('active'));
    [elements.nav1, elements.nav2, elements.nav3].forEach(el => el.classList.remove('active', 'completed'));

    if(step === 1) { elements.step1.classList.add('active'); elements.nav1.classList.add('active'); }
    if(step === 2) { elements.step2.classList.add('active'); elements.nav1.classList.add('completed'); elements.nav2.classList.add('active'); }
    if(step === 3) { elements.step3.classList.add('active'); elements.nav1.classList.add('completed'); elements.nav2.classList.add('completed'); elements.nav3.classList.add('active'); }
}

async function startTranslation() {
    try {
        const res = await fetch('/api/translate', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ file_id: state.uploadedFile.fileId })
        });
        const data = await res.json();
        if (data.success) connectSSE(data.task_id);
        else throw new Error(data.error);
    } catch (e) { showError(e.message); goToStep(1); }
}

function connectSSE(taskId) {
    state.currentTaskId = taskId;
    elements.taskIdDisplay.textContent = taskId;
    if (state.eventSource) state.eventSource.close();
    state.eventSource = new EventSource(`/api/translate/progress/${taskId}`);

    state.eventSource.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data);
        elements.progressBarFill.style.width = `${data.progress}%`;
        elements.progressMessage.textContent = data.message;
    });

    state.eventSource.addEventListener('complete', (e) => {
        const data = JSON.parse(e.data);
        state.eventSource.close();

        elements.translationWord.textContent = data.word;
        elements.progressBarFill.style.width = `100%`;
        elements.progressMessage.textContent = "완료!";

        state.showKeypoints = false;
        elements.keypointToggle.checked = false;
        updateResultVideo();

        setTimeout(() => goToStep(3), 800);
    });
}

// 결과 화면 - 키포인트 토글
elements.keypointToggle.addEventListener('change', (e) => {
    state.showKeypoints = e.target.checked;
    updateResultVideo();
});

function updateResultVideo() {
    elements.resultVideoPlaceholder.innerHTML = state.showKeypoints
        ? `<span style="color:#e74c3c">💀 Keypoint 영상 재생 중...</span><br><small>(${state.currentTaskId})</small>`
        : `<span style="color:#3498db">▶ 원본 영상 재생 중...</span><br><small>(${state.currentTaskId})</small>`;
}

// [신규] 결과 화면 -> 사전 자동 검색 연결
elements.btnSearchDict.addEventListener('click', () => {
    const rawWord = elements.translationWord.textContent;
    const searchKeyword = rawWord.split('(')[0].trim(); // 괄호 제거

    if (!searchKeyword || searchKeyword === "---") return showError("검색할 단어가 없습니다.");

    hideAllApps();
    elements.dictionaryApp.classList.remove('hidden');

    elements.dictSearchInput.value = searchKeyword;
    handleDictionarySearch();
});

// ==========================================
// 3. 사전 검색 기능
// ==========================================

elements.dictSearchBtn.addEventListener('click', handleDictionarySearch);
elements.dictSearchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleDictionarySearch();
});

async function handleDictionarySearch() {
    const query = elements.dictSearchInput.value.trim();
    if (!query) return showError("검색어를 입력해주세요.");

    elements.dictSearchBtn.disabled = true;
    elements.dictResultList.innerHTML = '<div class="empty-state">검색 중...</div>';

    try {
        const res = await fetch('/api/search', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query })
        });
        const data = await res.json();
        renderSearchResults(data.results);
    } catch (e) {
        showError("검색 중 오류가 발생했습니다.");
    } finally {
        elements.dictSearchBtn.disabled = false;
    }
}

function renderSearchResults(results) {
    elements.dictResultList.innerHTML = '';
    if (results.length === 0) {
        elements.dictResultList.innerHTML = '<div class="empty-state">검색 결과가 없습니다.</div>';
        return;
    }

    results.forEach(item => {
        const div = document.createElement('div');
        div.className = 'result-item';
        div.innerHTML = `
            <span class="result-word">${item.word}</span>
            <span class="play-icon-btn">▶</span>
        `;
        div.addEventListener('click', () => playDictionaryVideo(item));
        elements.dictResultList.appendChild(div);
    });
}

function playDictionaryVideo(item) {
    elements.dictVideoArea.classList.remove('hidden');
    elements.dictPlayingWord.textContent = item.word;

    const placeholder = elements.dictVideoArea.querySelector('.placeholder-video');
    placeholder.innerHTML = `▶ 사전 영상 재생 중...<br><small>${item.video_url}</small>`;

    elements.dictVideoArea.scrollIntoView({ behavior: 'smooth' });
}