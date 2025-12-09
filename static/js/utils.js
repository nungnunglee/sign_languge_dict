import { elements } from './state.js';

// --- API Helper ---
export async function fetchAPI(url, options = {}) {
    try {
        const response = await fetch(url, options);
        const data = await response.json();

        if (!response.ok || (data && data.success === false)) {
            throw new Error(data.message || data.error || 'Server Error');
        }
        return data;
    } catch (error) {
        throw error;
    }
}

// --- UI Feedback ---
export function showToast(message, type = 'success') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) return;

    const toast = document.createElement('div');
    const isError = type === 'error';

    toast.className = `toast ${isError ? 'error' : 'success'}`;
    const iconHtml = isError
        ? '<i class="fa-solid fa-circle-exclamation text-white"></i>'
        : '<i class="fa-solid fa-circle-check text-emerald-400"></i>';

    toast.innerHTML = `
        <span class="icon">${iconHtml}</span>
        <span class="font-medium text-sm text-white">${message}</span>
    `;

    toastContainer.appendChild(toast);

    // Animation
    requestAnimationFrame(() => {
        // Force reflow handled by browser optimization usually, but timeout ensures transition
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(20px)';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    });
}

export function showError(message) {
    showToast(message, 'error');
}

// --- Navigation ---
export function navigateTo(appName) {
    // Hide all apps
    [elements.introScreen, elements.translationApp, elements.dictionaryApp, elements.gameApp]
        .forEach(el => el?.classList.add('hidden'));

    // Show selected app
    switch (appName) {
        case 'intro':
            elements.introScreen.classList.remove('hidden');
            break;
        case 'translation':
            elements.translationApp.classList.remove('hidden');
            setStep(1);
            break;
        case 'dictionary':
            elements.dictionaryApp.classList.remove('hidden');
            resetDictionaryView();
            break;
        case 'game':
            elements.gameApp.classList.remove('hidden');
            break;
    }
}

function resetDictionaryView() {
    if (elements.dictVideoArea) elements.dictVideoArea.classList.add('hidden');
    if (elements.dictVideoPlayer) {
        elements.dictVideoPlayer.pause();
        elements.dictVideoPlayer.src = "";
    }
}

// --- Translation Stepper ---
export function setStep(stepNumber) {
    // Content Visibility
    [1, 2, 3].forEach(num => {
        const el = document.getElementById(`step-${num}-content`);
        if (el) el.classList.toggle('hidden', num !== stepNumber);
    });

    // Indicator Style
    [1, 2, 3].forEach(num => {
        const el = document.getElementById(`step-${num}`);
        if (el) el.classList.toggle('active', num === stepNumber);
    });

    // Reset Progress Bar on Step 2
    if (stepNumber === 2 && elements.progressBar) {
        elements.progressBar.style.width = '0%';
        elements.progressMessage.textContent = '작업 대기 중...';
    }
}