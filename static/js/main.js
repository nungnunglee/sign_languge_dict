import { elements, state } from './state.js';
import { navigateTo, showToast } from './utils.js';

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Modules

    // 2. Event Listeners for Navigation
    elements.cardTranslate?.addEventListener('click', () => navigateTo('translation'));
    elements.cardDictionary?.addEventListener('click', () => navigateTo('dictionary'));

    // 3. Initial Route
});

// --- Global Window Actions ---
// HTML의 onclick 속성에서 접근하기 위해 window 객체에 할당

window.goHome = () => {
    stopWebcamStream(); // 번역기 웹캠 정지

    [elements.resultVideoPlayer, elements.dictVideoPlayer].forEach(v => v?.pause());

    // Stop Game Stream
    if (state.game.stream) {
        state.game.stream.getTracks().forEach(t => t.stop());
        state.game.stream = null;
    }

    closeSidebar();
};

window.loadNextQuestion = loadNextQuestion;

// --- Sidebar Logic ---
window.openSidebar = () => {
    renderSidebarList();
    elements.sidebar.classList.remove('translate-x-[150%]');
    elements.sidebarOverlay.classList.remove('hidden');
    setTimeout(() => elements.sidebarOverlay.classList.remove('opacity-0'), 10);
};

window.closeSidebar = () => {
    elements.sidebar.classList.add('translate-x-[150%]');
    elements.sidebarOverlay.classList.add('opacity-0');
    setTimeout(() => elements.sidebarOverlay.classList.add('hidden'), 300);
};

window.deleteHistoryItem = (index) => {
    if (confirm("이 기록을 삭제하시겠습니까?")) {
        state.fileHistory.splice(index, 1);
        renderSidebarList();
        showToast("기록이 삭제되었습니다.");
    }
};

function renderSidebarList() {
    elements.sidebarList.innerHTML = '';

    if (state.fileHistory.length === 0) {
        elements.sidebarList.innerHTML = `
            <div class="flex flex-col items-center justify-center h-40 text-slate-400">
                <i class="fa-regular fa-folder-open text-3xl mb-2 opacity-50"></i>
                <span class="text-sm">보관된 파일이 없습니다.</span>
            </div>`;
        return;
    }

    state.fileHistory.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'relative bg-slate-50 p-3 rounded-2xl border border-slate-100 hover:border-indigo-200 transition-colors group pr-8 mb-2';

        const isTranslation = item.type === 'translation';
        const iconClass = isTranslation ? 'fa-hands-asl-interpreting text-indigo-500' : 'fa-gamepad text-orange-500';
        const typeLabel = isTranslation ? '번역' : '게임';

        div.innerHTML = `
            <button onclick="deleteHistoryItem(${index})" class="absolute top-2 right-2 text-slate-300 hover:text-red-500 transition-colors p-1" title="삭제">
                <i class="fa-solid fa-trash-can"></i>
            </button>
            <div class="flex items-center gap-3 mb-2">
                <div class="w-8 h-8 rounded-full bg-white flex items-center justify-center shadow-sm">
                    <i class="fa-solid ${iconClass} text-xs"></i>
                </div>
                <div class="flex-1 min-w-0">
                    <p class="text-xs font-bold text-slate-500 mb-0.5">${typeLabel}</p>
                    <h4 class="text-sm font-semibold text-slate-800 truncate" title="${item.filename}">${item.filename}</h4>
                </div>
            </div>
            ${item.word ? `<div class="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded inline-block mb-2">결과: ${item.word}</div>` : ''}
            <div class="text-xs text-slate-400 flex justify-between items-center">
                <span>${item.date}</span>
                <a href="${item.url}" target="_blank" class="text-indigo-500 hover:underline">보기</a>
            </div>
        `;
        elements.sidebarList.appendChild(div);
    });
}