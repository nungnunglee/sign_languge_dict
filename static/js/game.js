import { state, elements } from './state.js';
import { showToast, showError, navigateTo, fetchAPI } from './utils.js';

// --- Configuration ---
const CONFIG = {
    API: {
        QUIZ: '/api/game/quiz',
        MULTI_QUIZ: '/api/game/multiple-choice',
        SUBMIT: '/api/game/submit',
        PROGRESS: '/api/translate/progress/'
    },
    LIMITS: { COUNTDOWN: 3, MAX_LEVEL: 10 },
    MIME_TYPE: 'video/mp4',
    ANIMATION: { STAR_COUNT: 12 }
};

const localState = {
    isRecording: false,
    selectedUploadFile: null
};

// === 1. Initialization & Menu ===

export function initGame() {
    state.game.totalScore = 0;
    state.game.level = 1;

    updateTotalScoreUI();
    updateLevelUI();
    toggleScreen('menu');
}

// Global Exports for HTML Interaction
window.startGameMode = (mode) => {
    state.game.mode = mode; // 'practice' | 'multichoice'
    if (mode === 'practice') {
        startGameWebcam();
        loadNextQuestion();
    } else if (mode === 'multichoice') {
        loadNextQuestion();
    }
};

// === 2. Quiz Logic ===

export async function loadNextQuestion() {
    // Screen Transition
    state.game.mode === 'practice'
        ? (toggleScreen('quiz'), resetUploadState())
        : toggleScreen('multichoice');

    updateLevelUI();

    try {
        const apiUrl = state.game.mode === 'practice' ? CONFIG.API.QUIZ : CONFIG.API.MULTI_QUIZ;
        const data = await fetchAPI(apiUrl);

        if (state.game.mode === 'practice') setupPracticeQuiz(data.quiz);
        else setupMultiChoiceQuiz(data.quiz);

    } catch (err) {
        showError('퀴즈 데이터를 불러오는데 실패했습니다.');
        console.error(err);
    }
}

function setupMultiChoiceQuiz(quiz) {
    const { target_word, target_id, options } = quiz;
    state.game.currentWord = target_word;
    state.game.currentQuizId = target_id;

    elements.multiTargetWord.textContent = target_word;
    elements.multiOptionsContainer.innerHTML = '';

    options.forEach(option => {
        const card = createOptionCard(option);
        elements.multiOptionsContainer.appendChild(card);
    });
}

function createOptionCard(option) {
    const card = document.createElement('div');
    card.className = "relative bg-slate-900 rounded-xl overflow-hidden aspect-video cursor-pointer hover:ring-4 hover:ring-indigo-400 transition-all group shadow-md select-none active:scale-95";
    card.innerHTML = `
        <video src="${option.video_url}" class="w-full h-full object-contain pointer-events-none" autoplay loop muted playsinline></video>
        <div class="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex flex-col items-center justify-center z-10">
            <i class="fa-regular fa-hand-pointer text-white opacity-0 group-hover:opacity-100 text-4xl drop-shadow-lg transform scale-50 group-hover:scale-100 transition-all mb-2"></i>
            <span class="text-white text-xs opacity-0 group-hover:opacity-100 font-medium bg-black/40 px-2 py-1 rounded">꾹 눌러서 확대</span>
        </div>
    `;

    // Attach Long Press Logic
    addLongPressEvent(
        card,
        () => checkMultiChoiceAnswer(option.id), // Short Click
        () => openVideoPreview(option.video_url) // Long Press
    );
    return card;
}

function setupPracticeQuiz(quiz) {
    const { word, id, hint_video_url } = quiz;
    state.game.currentWord = word;
    state.game.currentQuizId = id;
    state.game.hintUrl = hint_video_url;

    elements.targetWord.textContent = word;
    elements.hintArea.classList.add('hidden');
}

function checkMultiChoiceAnswer(selectedId) {
    const isCorrect = selectedId === state.game.currentQuizId;
    showGameResult({
        success: isCorrect,
        recognized_word: isCorrect ? state.game.currentWord : "오답",
        isMultiChoice: true
    });
}

// === 3. Media & Recording (Practice Mode) ===

export async function startGameWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        state.game.stream = stream;
        elements.gameCamPreview.srcObject = stream;
    } catch (err) {
        showError('카메라 권한이 필요합니다.');
    }
}

function toggleRecording() {
    if (!state.game.stream) return showToast("카메라가 연결되지 않았습니다.");
    localState.isRecording ? stopRecording() : startCountdown();
}

function startCountdown() {
    elements.gameCamPreview.classList.remove('hidden');
    elements.gameCountdown.classList.remove('hidden');
    document.getElementById('game-file-preview').classList.add('hidden');

    let count = CONFIG.LIMITS.COUNTDOWN;
    elements.gameCountdown.textContent = count;

    const timer = setInterval(() => {
        count--;
        if (count > 0) {
            elements.gameCountdown.textContent = count;
        } else {
            clearInterval(timer);
            elements.gameCountdown.classList.add('hidden');
            startMediaRecorder();
        }
    }, 1000);
}

function startMediaRecorder() {
    state.game.chunks = [];
    state.game.mediaRecorder = new MediaRecorder(state.game.stream);

    state.game.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.game.chunks.push(e.data);
    };

    state.game.mediaRecorder.onstop = () => {
        const blob = new Blob(state.game.chunks, { type: CONFIG.MIME_TYPE });
        const file = new File([blob], 'game_attempt.mp4', { type: CONFIG.MIME_TYPE });
        preparePreview(file);
    };

    state.game.mediaRecorder.start();
    localState.isRecording = true;
    updateRecordButtonUI(true);
    elements.gameRecIndicator.classList.remove('hidden');
}

function stopRecording() {
    if (state.game.mediaRecorder?.state === 'recording') {
        state.game.mediaRecorder.stop();
        localState.isRecording = false;
        updateRecordButtonUI(false);
        elements.gameRecIndicator.classList.add('hidden');
    }
}

function preparePreview(file) {
    localState.selectedUploadFile = file;
    const previewVideo = document.getElementById('game-file-preview');

    elements.gameCamPreview.classList.add('hidden');
    previewVideo.classList.remove('hidden');
    previewVideo.src = URL.createObjectURL(file);
    previewVideo.play();

    toggleControlButtons('upload');
}

function resetUploadState() {
    localState.selectedUploadFile = null;
    const previewVideo = document.getElementById('game-file-preview');
    previewVideo.pause();
    previewVideo.src = "";
    previewVideo.classList.add('hidden');

    elements.gameCamPreview.classList.remove('hidden');
    toggleControlButtons('default');
    updateRecordButtonUI(false);
}

// === 4. Submission & Progress ===

async function submitGameFile() {
    const file = localState.selectedUploadFile;
    if (!file) return showToast("제출할 영상이 없습니다.");

    toggleScreen('loading');
    updateProgressBar(0, '서버 전송 중...');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_word', state.game.currentWord);

    try {
        const data = await fetchAPI(CONFIG.API.SUBMIT, { method: 'POST', body: formData });
        monitorProgress(data.task_id, file.name);
    } catch (err) {
        showError('업로드 오류 발생');
        resetUploadState();
        toggleScreen('quiz');
    }
}

function monitorProgress(taskId, originalFilename) {
    const eventSource = new EventSource(`${CONFIG.API.PROGRESS}${taskId}`);

    eventSource.addEventListener('progress', (e) => {
        const { progress, message } = JSON.parse(e.data);
        updateProgressBar(progress, message);
    });

    eventSource.addEventListener('complete', (e) => {
        eventSource.close();
        const { game_result } = JSON.parse(e.data);

        // Save History
        state.fileHistory.unshift({
            type: 'game',
            filename: originalFilename || 'Game Attempt',
            date: new Date().toLocaleTimeString(),
            url: `/api/video/original/${taskId}`,
            word: game_result.success ? '성공' : '실패'
        });

        setTimeout(() => showGameResult(game_result), 500);
    });

    eventSource.addEventListener('error', () => {
        eventSource.close();
        showError('채점 중 오류 발생');
        toggleScreen('quiz');
    });
}

// === 5. UI Helpers & Animations ===

function showGameResult(result) {
    toggleScreen('result');
    const { success, recognized_word, isMultiChoice } = result;

    // Update Result UI
    if (success) {
        setResultCardStyle(true, '성공!', 'text-emerald-600', 'border-emerald-400', 'bg-emerald-50');
        elements.scoreValue.innerHTML = '<i class="fa-solid fa-star text-5xl"></i>';
        elements.resultDesc.innerHTML = isMultiChoice
            ? `<b>"${state.game.currentWord}"</b>의 올바른 영상을 찾으셨습니다!`
            : `AI가 <b>"${recognized_word}"</b>(으)로 인식했습니다.<br>정확합니다!`;

        state.game.totalScore += 10;
        triggerStarAnimation(CONFIG.ANIMATION.STAR_COUNT);
        updateTotalScoreUI();
    } else {
        setResultCardStyle(false, '아쉬워요', 'text-orange-600', 'border-orange-400', 'bg-orange-50');
        elements.scoreValue.innerHTML = '<i class="fa-regular fa-face-frown text-5xl"></i>';
        elements.resultDesc.innerHTML = isMultiChoice
            ? `선택하신 영상은 <b>"${state.game.currentWord}"</b>이 아닙니다.`
            : `AI가 <b>"${recognized_word}"</b>(으)로 인식했습니다.<br>정답은 "${state.game.currentWord}" 입니다.`;
    }

    setupNextButton();
}

function setResultCardStyle(success, title, titleColor, borderColor, bgColor) {
    elements.resultTitle.textContent = title;
    elements.resultTitle.className = `text-3xl font-bold mb-2 ${titleColor}`;
    elements.scoreCircle.className = `w-40 h-40 rounded-full border-[10px] mx-auto flex flex-col items-center justify-center mb-6 shadow-xl ${borderColor} ${titleColor} ${bgColor}`;
}

function setupNextButton() {
    const nextBtn = document.getElementById('btn-next-question');
    const isFinal = state.game.level >= CONFIG.LIMITS.MAX_LEVEL;

    nextBtn.textContent = isFinal ? "최종 결과 보기" : "다음 문제";
    nextBtn.onclick = isFinal ? showFinalResult : () => {
        state.game.level += 1;
        loadNextQuestion();
    };
}

function showFinalResult() {
    toggleScreen('final');
    document.getElementById('game-final-score-value').textContent = state.game.totalScore;
    document.getElementById('game-final-score-text').textContent = Math.floor(state.game.totalScore / 10);

    // Stop Webcam
    if (state.game.stream) {
        state.game.stream.getTracks().forEach(track => track.stop());
        state.game.stream = null;
    }
}

// --- Utils: Long Press, Animation, Toggles ---

function toggleScreen(screenName) {
    [
        elements.gameMenuScreen, elements.gameMultiChoiceScreen, elements.screenQuiz,
        elements.screenLoading, elements.screenResult, elements.screenFinal
    ].forEach(el => el?.classList.add('hidden'));

    const target = {
        menu: elements.gameMenuScreen,
        multichoice: elements.gameMultiChoiceScreen,
        quiz: elements.screenQuiz,
        loading: elements.screenLoading,
        result: elements.screenResult,
        final: elements.screenFinal
    }[screenName];

    if (target) target.classList.remove('hidden');
}

function toggleControlButtons(mode) {
    const def = document.getElementById('game-controls-default');
    const upl = document.getElementById('game-controls-upload');
    const isUpload = mode === 'upload';

    def.classList.toggle('hidden', isUpload);
    upl.classList.toggle('hidden', !isUpload);
}

function updateRecordButtonUI(isRecording) {
    const btn = elements.btnGameRecord;
    const icon = document.getElementById('icon-record');
    const text = document.getElementById('text-record');
    const baseClass = "group relative flex flex-col items-center justify-center text-white shadow-lg transition-all duration-500 ease-in-out overflow-hidden";

    if (isRecording) {
        btn.className = `${baseClass} w-full p-4 rounded-2xl bg-red-500 hover:bg-red-600 shadow-red-200 animate-pulse`;
        icon.className = 'fa-solid fa-stop text-2xl mb-1';
        text.textContent = '녹화 중지';
    } else {
        btn.className = `${baseClass} w-24 h-24 rounded-full mx-auto bg-orange-500 hover:bg-orange-600 shadow-orange-200 active:scale-95 hover:scale-105`;
        icon.className = 'fa-solid fa-video text-2xl mb-1 group-hover:scale-110';
        text.textContent = '녹화하기';
    }
}

function updateTotalScoreUI() { elements.gameTotalScore.textContent = state.game.totalScore; }
function updateLevelUI() { elements.gameLevelBadge.textContent = `LEVEL ${state.game.level}`; }
function updateProgressBar(pct, msg) {
    elements.gameProgressBar.style.width = `${pct}%`;
    elements.gameProgressMsg.textContent = msg;
}

// Star Animation Logic (Keep as is mostly, just cleaned up)
function triggerStarAnimation(count) {
    const source = elements.scoreCircle;
    const target = document.querySelector('.fa-star'); // Target icon in header
    if (!source || !target) return;

    const sRect = source.getBoundingClientRect();
    const tRect = target.getBoundingClientRect();
    const startX = sRect.left + sRect.width / 2;
    const startY = sRect.top + sRect.height / 2;
    const endX = tRect.left + tRect.width / 2;
    const endY = tRect.top + tRect.height / 2;

    const fragment = document.createDocumentFragment();

    for (let i = 0; i < count; i++) {
        const star = document.createElement('i');
        star.classList.add('fa-solid', 'fa-star', 'flying-star');
        star.style.left = `${startX}px`;
        star.style.top = `${startY}px`;

        const duration = 1000 + Math.random() * 500;
        const delay = Math.random() * 300;

        star.style.setProperty('--destX', `${endX - startX}px`);
        star.style.setProperty('--destY', `${endY - startY}px`);
        star.style.setProperty('--spreadX', `${(Math.random() - 0.5) * 150}px`);
        star.style.setProperty('--spreadY', `${(Math.random() - 0.5) * 150}px`);
        star.style.animation = `fly-to-score ${duration}ms ease-in-out ${delay}ms forwards`;

        fragment.appendChild(star);
        setTimeout(() => star.remove(), duration + delay + 100);
    }
    document.body.appendChild(fragment);
}

// Long Press Handler
function addLongPressEvent(element, onClick, onLongPress) {
    let timer;
    let isLongPress = false;
    const DURATION = 500;

    const start = (e) => {
        if (e.type === 'mousedown' && e.button !== 0) return;
        isLongPress = false;
        timer = setTimeout(() => {
            isLongPress = true;
            if (navigator.vibrate) navigator.vibrate(50);
            onLongPress();
        }, DURATION);
    };

    const end = () => {
        if (timer) clearTimeout(timer);
        if (!isLongPress) onClick();
    };

    const cancel = () => {
        if (timer) clearTimeout(timer);
        isLongPress = true; // Prevent click on cancel
    };

    element.addEventListener('mousedown', start);
    element.addEventListener('mouseup', end);
    element.addEventListener('mouseleave', cancel);
    element.addEventListener('touchstart', start, { passive: true });
    element.addEventListener('touchend', end);
    element.addEventListener('touchcancel', cancel);
    element.addEventListener('touchmove', cancel);
}

// Modal & Other Global Exports
function openVideoPreview(url) {
    if (elements.previewModal && elements.previewModalVideo) {
        elements.previewModalVideo.src = url;
        elements.previewModal.classList.remove('hidden');
        elements.previewModalVideo.play().catch(() => {});
    }
}

window.toggleGameRecording = toggleRecording;
window.cancelUpload = resetUploadState;
window.confirmUpload = submitGameFile;
window.initGameRound = () => {
    state.game.mode === 'practice' ? (toggleScreen('quiz'), resetUploadState()) : toggleScreen('multichoice');
};
window.triggerFileUpload = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/*';
    input.onchange = (e) => {
        const file = e.target.files?.[0];
        if (file) preparePreview(file);
    };
    input.click();
};
window.toggleHint = () => {
    const h = elements.hintArea;
    const v = elements.hintVideo;
    h.classList.toggle('hidden');
    if (!h.classList.contains('hidden') && state.game.hintUrl) {
        v.src = state.game.hintUrl;
        v.play();
    } else {
        v.pause();
    }
};
window.closeVideoPreview = () => {
    if (elements.previewModal) {
        elements.previewModal.classList.add('hidden');
        elements.previewModalVideo.pause();
        elements.previewModalVideo.src = "";
    }
};