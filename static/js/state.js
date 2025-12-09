export const state = {
    // --- File & Input State ---
    uploadedFile: { fileId: null, filename: null },
    currentTaskId: null,
    inputType: 'file', // 'file' | 'cam'

    // --- Recorder & Stream ---
    webcamStream: null,
    mediaRecorder: null,
    recordedChunks: [],
    eventSource: null,

    // --- History ---
    fileHistory: [],

    // --- Game State ---
    game: {
        mode: null, // 'practice' (AI) | 'multichoice' (4지선다)
        currentWord: null,
        currentQuizId: null,
        hintUrl: null,
        stream: null,
        mediaRecorder: null,
        chunks: [],
        totalScore: 0,
        level: 1
    }
};

export const elements = {
    // --- Global Layout ---
    introScreen: document.getElementById('intro-screen'),
    translationApp: document.getElementById('translation-app'),
    dictionaryApp: document.getElementById('dictionary-app'),
    gameApp: document.getElementById('game-app'),

    // --- Sidebar ---
    sidebar: document.getElementById('file-sidebar'),
    sidebarOverlay: document.getElementById('sidebar-overlay'),
    sidebarList: document.getElementById('sidebar-list'),

    // --- Main Navigation Cards ---
    cardTranslate: document.getElementById('card-translate'),
    cardDictionary: document.getElementById('card-dictionary'),
    cardGame: document.getElementById('card-game'),

    // --- Translation Module ---
    btnModeFile: document.getElementById('btn-mode-file'),
    btnModeCam: document.getElementById('btn-mode-cam'),
    modeFileArea: document.getElementById('mode-file-area'),
    modeCamArea: document.getElementById('mode-cam-area'),
    dropArea: document.getElementById('drop-area'),
    fileInput: document.getElementById('video-file-input'),
    fileStatusBox: document.getElementById('file-status-box'),
    statusText: document.querySelector('.status-text'),
    uploadStartButton: document.getElementById('upload-start-btn'),

    // Translation Camera & Result
    videoPreview: document.getElementById('cam-preview'),
    camPlaceholder: document.getElementById('cam-placeholder'),
    btnStartRecord: document.getElementById('btn-start-record'),
    btnStopRecord: document.getElementById('btn-stop-record'),
    recIndicator: document.getElementById('recording-indicator'),
    camStatusText: document.getElementById('cam-status-text'),
    progressBar: document.getElementById('progress-bar'),
    progressMessage: document.getElementById('progress-message'),
    resultWord: document.getElementById('result-word'),
    resultVideoPlayer: document.getElementById('result-video-player'),
    resultVideoPlaceholder: document.getElementById('result-video-placeholder'),
    keypointToggle: document.getElementById('keypoint-toggle'),

    // --- Dictionary Module ---
    dictSearchInput: document.getElementById('dict-search-input'),
    dictSearchBtn: document.getElementById('dict-search-btn'),
    dictResultList: document.getElementById('dict-result-list'),
    dictVideoArea: document.getElementById('dict-video-area'),
    dictVideoPlayer: document.getElementById('dict-video-player'),
    dictVideoPlaceholder: document.getElementById('dict-video-placeholder'),
    dictPlayingWord: document.getElementById('dict-playing-word'),

    // --- Game Module ---
    // Screens
    gameMenuScreen: document.getElementById('game-menu-screen'),
    gameMultiChoiceScreen: document.getElementById('game-multichoice-screen'),
    screenQuiz: document.getElementById('game-quiz-screen'),
    screenLoading: document.getElementById('game-loading-screen'),
    screenResult: document.getElementById('game-result-screen'),
    screenFinal: document.getElementById('game-final-screen'),

    // Elements
    multiTargetWord: document.getElementById('multi-target-word'),
    multiOptionsContainer: document.getElementById('multi-options-container'),
    targetWord: document.getElementById('game-target-word'),
    hintArea: document.getElementById('game-hint-area'),
    hintVideo: document.getElementById('game-hint-video'),
    gameCamPreview: document.getElementById('game-cam-preview'),
    gameCountdown: document.getElementById('game-countdown'),
    gameRecIndicator: document.getElementById('game-rec-indicator'),
    btnGameRecord: document.getElementById('btn-game-record'),

    // Status
    gameProgressBar: document.getElementById('game-progress-bar'),
    gameProgressMsg: document.getElementById('game-progress-msg'),
    gameTotalScore: document.getElementById('game-total-score'),
    gameLevelBadge: document.getElementById('game-level-badge'),
    scoreValue: document.getElementById('game-score-value'),
    scoreCircle: document.getElementById('game-score-circle'),
    resultTitle: document.getElementById('game-result-title'),
    resultDesc: document.getElementById('game-result-desc'),

    // Video Preview Modal
    previewModal: document.getElementById('video-preview-modal'),
    previewModalVideo: document.getElementById('preview-modal-video'),
};