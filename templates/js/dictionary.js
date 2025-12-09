import { elements } from './state.js';
import { showError, fetchAPI } from './utils.js';

export function initDictionary() {
    elements.dictSearchBtn?.addEventListener('click', searchDictionary);
    elements.dictSearchInput?.addEventListener('keydown', e => {
        if (e.key === 'Enter') searchDictionary();
    });
}

async function searchDictionary() {
    const query = elements.dictSearchInput.value.trim();
    if (!query) return showError("검색어를 입력해주세요.");

    elements.dictSearchBtn.disabled = true;

    try {
        const data = await fetchAPI('/api/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query })
        });
        renderResults(data.results);
    } catch (e) {
        showError("검색 중 오류가 발생했습니다.");
    } finally {
        elements.dictSearchBtn.disabled = false;
    }
}

function renderResults(results) {
    elements.dictResultList.innerHTML = '';

    if (!results || results.length === 0) {
        elements.dictResultList.innerHTML = `
            <div class="text-center py-10 text-slate-400">
                <i class="fa-solid fa-circle-question text-2xl mb-2 opacity-50"></i><br>
                검색 결과가 없습니다.
            </div>`;
        return;
    }

    results.forEach(item => {
        const div = document.createElement('div');
        div.className = 'result-item'; // CSS defined in dictionary.css
        div.innerHTML = `
            <span class="font-bold text-slate-700">${item.word}</span>
            <button class="w-8 h-8 rounded-full bg-emerald-50 text-emerald-500 hover:bg-emerald-500 hover:text-white transition-colors flex items-center justify-center">
                <i class="fa-solid fa-play ml-0.5"></i>
            </button>
        `;

        div.addEventListener('click', () => {
            playDictionaryVideo(item.word, item.video_url);
            // Highlight active item
            document.querySelectorAll('.result-item').forEach(el => el.classList.remove('playing'));
            div.classList.add('playing');
        });

        elements.dictResultList.appendChild(div);
    });
}

function playDictionaryVideo(word, url) {
    elements.dictVideoArea.classList.remove('hidden');
    elements.dictPlayingWord.textContent = word;

    const player = elements.dictVideoPlayer;
    player.src = url;
    player.classList.remove('hidden');
    elements.dictVideoPlaceholder.classList.add('hidden');

    player.play().catch(e => console.warn("Autoplay blocked:", e));
}