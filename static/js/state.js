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

    // --- 메인 이동 카드 ---
    cardTranslate: document.getElementById('card-translate'),   // 번역 카드
    cardDictionary: document.getElementById('card-dictionary'), // 사전 카드
    cardGame: document.getElementById('card-game'),             // 게임 카드

    // --- 번역 1단계 모드 변경 ---
    btnModeFile: document.getElementById('btn-mode-file'),          // 모드 변경 버튼(파일)
    btnModeCam: document.getElementById('btn-mode-cam'),            // 모드 변경 버튼(캠)

    // --- 파일 업로드 ---
    modeFileArea: document.getElementById('mode-file-area'),        // 파일 업로드 모드 영역
    dropArea: document.getElementById('drop-area'),                 // 드랍 영역
    fileInput: document.getElementById('video-file-input'),         // 파일 입력
    fileStatusBox: document.getElementById('file-status-box'),      // 현재 파일 상태
    statusText: document.querySelector('.status-text'),             // 현재 파일 상태 텍스트

    uploadStartButton: document.getElementById('upload-start-btn'), // 파일 업로드 버튼

    // --- 카메라 ---
    modeCamArea: document.getElementById('mode-cam-area'),          // 캠 모드 영역
    videoPreview: document.getElementById('cam-preview'),           // 캠 보여주기 영역
    recordedVideoPlayer: document.getElementById('recorded-video-player'),
    camPlaceholder: document.getElementById('cam-placeholder'),     // 
    recIndicator: document.getElementById('recording-indicator'),
    camStatusText: document.getElementById('cam-status-text'),      // 캠 상태 텍스트
    
    btnStartRecord: document.getElementById('btn-start-record'),    // 녹화 시작 버튼
    btnStopRecord: document.getElementById('btn-stop-record'),      // 녹화 중지 버튼

    recordToggleBtn: document.getElementById('record-toggle-btn'),  // 녹화/중지 버튼
    translateBtn: document.getElementById('translate-btn'),         // 웹캠 영상 번역

    // --- 번역 2단계 분석 단계 ---
    progressMessage: document.getElementById('progress-message'),   // 진행 메시지: 진행 상황을 알려주기 위한 메시지
    progressBar: document.getElementById('progress-bar'),           // 진행바

    // --- 번역 3단계 결과 단계 ---
    resultWord: document.getElementById('result-word'),                             // 번역 결과 단어
    resultVideoPlayer: document.getElementById('result-video-player'),              // 결과 비디오
    resultVideoPlaceholder: document.getElementById('result-video-placeholder'),    // 결과 비디오 영역
    keypointToggle: document.getElementById('keypoint-toggle'),                     // 키포인트 토글 스위치
    go_translation: document.getElementById('go_translation'),                      // 메인화면으로 가는 버튼

    // --- 사전 모듈 ---
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