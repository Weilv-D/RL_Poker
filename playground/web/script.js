// System State
let systemState = {
    polling: null,
    selectedCards: new Set(),
    lastData: null,
    isConfiguring: true,
    aiMoving: false,
    lastAiTrigger: 0
};

const UI = {
    overlay: document.getElementById('overlay'),
    main: document.getElementById('main-ui'),
    ckptSelect: document.getElementById('ckpt-select'),
    startBtn: document.getElementById('btn-start'),
    configBtn: document.getElementById('btn-config'),
    
    status: document.getElementById('status-display'),
    turn: document.getElementById('turn-display'),
    
    hand: document.getElementById('hand-container'),
    table: document.getElementById('played-container'),
    tableMsg: document.getElementById('table-message'),
    logs: document.getElementById('log-panel'),
    
    playBtn: document.getElementById('btn-play'),
    passBtn: document.getElementById('btn-pass'),
    prestart: document.getElementById('prestart-controls'),
    shuffleBtn: document.getElementById('btn-shuffle'),
    beginBtn: document.getElementById('btn-begin'),
};

const SEAT_MAP = {
    1: { el: document.getElementById('opp-right'), count: document.getElementById('count-right'), nameEl: document.getElementById('name-right') },
    2: { el: document.getElementById('opp-top'), count: document.getElementById('count-top'), nameEl: document.getElementById('name-top') },
    3: { el: document.getElementById('opp-left'), count: document.getElementById('count-left'), nameEl: document.getElementById('name-left') }
};

// --- Initialization ---

async function init() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        
        UI.ckptSelect.innerHTML = '';
        data.checkpoints.forEach(path => {
            const opt = document.createElement('option');
            opt.value = path;
            
            // Format name nicely: star_008_step_110879895 -> 星宿 · 捌
            const filename = path.split('/').pop().replace('.pt', '');
            let displayName = filename;
            const match = filename.match(/star_(\d+)_step_(\d+)/);
            if (match) {
                const num = parseInt(match[1]);
                const cnNums = ['零','壹','贰','叁','肆','伍','陆','柒','捌','玖','拾','拾壹','拾贰'];
                const cnNum = num < cnNums.length ? cnNums[num] : num;
                displayName = `星宿 · ${cnNum}`; 
            }
            
            opt.text = displayName;
            if (path === data.current_checkpoint) opt.selected = true;
            UI.ckptSelect.appendChild(opt);
        });
        
    } catch (e) {
        console.error("Config fetch failed", e);
    }
}

// Start Game
UI.startBtn.addEventListener('click', async () => {
    const ckpt = UI.ckptSelect.value;
    const mode = document.getElementById('mode-select').value;
    
    UI.startBtn.disabled = true;
    UI.startBtn.textContent = "载入中";
    
    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ checkpoint: ckpt, mode: mode })
        });
        
        await fetch('/api/reset', { method: 'POST' });
        
        UI.overlay.classList.add('hidden');
        UI.main.classList.remove('hidden');
        
        systemState.isConfiguring = false;
        startLoop();
        
    } catch (e) {
        console.error(e);
        alert("启动失败");
    } finally {
        UI.startBtn.disabled = false;
        UI.startBtn.textContent = "入 局";
    }
});

UI.configBtn.addEventListener('click', () => {
    systemState.isConfiguring = true;
    clearInterval(systemState.polling);
    UI.main.classList.add('hidden');
    UI.overlay.classList.remove('hidden');
    init(); 
});


// --- Game Loop ---

function startLoop() {
    if (systemState.polling) clearInterval(systemState.polling);
    fetchState(); 
    systemState.polling = setInterval(fetchState, 1000);
}

async function fetchState() {
    if (systemState.isConfiguring) return;
    try {
        const res = await fetch('/api/state');
        const data = await res.json();
        render(data);
        
        // 如果是 AI 回合且游戏未结束，自动触发 AI 移动
        if (data.game_started && !data.game_over && data.current_player !== 0 && !systemState.aiMoving) {
            const now = Date.now();
            // 调整 AI 出牌思考间隔 (2000ms)
            if (now - systemState.lastAiTrigger < 2000) return;
            systemState.lastAiTrigger = now;
            await triggerAIMove();
        }
    } catch (e) {
        // console.warn("Polling error");
    }
}

async function triggerAIMove() {
    if (systemState.aiMoving) return;
    systemState.aiMoving = true;
    try {
        const res = await fetch('/api/ai_move');
        const data = await res.json();
        if (data.did_move) {
            // AI 移动成功，刷新状态
            setTimeout(fetchState, 300); // 延迟刷新以便观察
        }
    } catch (e) {
        console.error("AI move error", e);
    } finally {
        systemState.aiMoving = false;
    }
}

// --- Rendering ---

const SUITS = ['♥', '♦', '♣', '♠'];
const RANKS = ['3','4','5','6','7','8','9','10','J','Q','K','A','2'];

function render(data) {
    // 保存最新状态
    systemState.lastData = data;
    if (!data.game_started) {
        systemState.selectedCards.clear();
    }
    
    // Status
    if (!data.game_started && !data.game_over) {
        UI.status.textContent = "待 开";
        UI.turn.textContent = "未开始";
        UI.turn.style.color = 'var(--ink-secondary)';
    } else {
        UI.status.textContent = data.game_over ? "终 局" : "对 弈";
        const turnName = data.current_player === 0 ? "本 尊" : `${data.player_names ? data.player_names[data.current_player] : '对手'}`;
        UI.turn.textContent = `执手  ${turnName}`;
        UI.turn.style.color = data.current_player === 0 ? 'var(--accent-red)' : 'var(--ink-secondary)';
    }
    
    // Highlight Active
    [1, 2, 3].forEach(seat => {
        SEAT_MAP[seat].el.classList.remove('active');
        if (data.current_player === seat) SEAT_MAP[seat].el.classList.add('active');
        SEAT_MAP[seat].count.textContent = data.cards_remaining[seat];
        
        if (data.player_names && data.player_names[seat]) {
             SEAT_MAP[seat].nameEl.textContent = data.player_names[seat];
        }
    });
    
    // 更新玩家手牌数量
    const countMe = document.getElementById('count-me');
    if (countMe) {
        countMe.textContent = data.cards_remaining[0];
    }

    const nameMe = document.getElementById('name-me');
    if (nameMe && data.player_names && data.player_names[0]) {
        nameMe.textContent = data.player_names[0];
    }
    
    renderTable(data);
    const isMyTurn = data.game_started && data.current_player === 0 && !data.game_over;
    renderHand(data.my_hand, isMyTurn);
    const canPass = isMyTurn && !data.is_new_lead;
    updateControls(isMyTurn, data.game_over, canPass);
    updatePrestartControls(data);
    renderLogs(data.history);
}

function renderTable(data) {
    const cards = data.table_cards;
    UI.table.innerHTML = '';
    UI.tableMsg.textContent = '';
    
    if (!data.game_started) {
        UI.tableMsg.textContent = "待 开";
        return;
    }
    if (data.is_new_lead) {
        UI.tableMsg.textContent = "先 手";
        return;
    }
    
    if (!cards || cards.length === 0) {
        if (data.consecutive_passes > 0) {
             UI.tableMsg.textContent = "过";
        }
        return;
    }
    
    cards.forEach(c => {
        UI.table.appendChild(createCard(c));
    });
}

function renderHand(handIds, isMyTurn) {
    // 1. Filter selection to valid cards
    const handSet = new Set(handIds);
    systemState.selectedCards = new Set(
        Array.from(systemState.selectedCards).filter(c => handSet.has(c))
    );

    // 2. Sync DOM Smartly (keyed by card index)
    const existingCards = Array.from(UI.hand.children);
    const existingMap = new Map();
    existingCards.forEach(el => {
        existingMap.set(parseInt(el.dataset.idx), el);
    });

    // Check if we need full structure update (if order changed or new cards)
    const currentStr = handIds.join(',');
    const domStr = existingCards.map(el => el.dataset.idx).join(',');
    
    // If structure is same, just update attributes to avoid layout thrashing
    const structureChanged = currentStr !== domStr;

    if (structureChanged) {
        UI.hand.innerHTML = '';
        handIds.forEach(c => {
            // Reuse if exists (preserves transient states but order might force re-append)
            // Ideally we clone or just create new if simple. 
            // For simple list, let's just create. To animate properly, we'd need FLIP.
            // But simple "don't destroy if same" is often enough.
            let el = existingMap.get(c);
            if (!el) {
                el = createCard(c);
            }
            // Bind events (refresh ensures closures are current)
            updateCardState(el, c, isMyTurn);
            UI.hand.appendChild(el);
        });
    } else {
        // Just update states
        existingCards.forEach(el => {
            const c = parseInt(el.dataset.idx);
            updateCardState(el, c, isMyTurn);
        });
    }
}

function updateCardState(el, c, isMyTurn) {
    const isSelected = systemState.selectedCards.has(c);
    if (isSelected) el.classList.add('selected');
    else el.classList.remove('selected');
    
    if (isMyTurn) {
        el.onclick = () => toggleCard(c, el);
        el.style.cursor = 'pointer';
    } else {
        el.onclick = null;
        el.style.cursor = 'default';
    }
}

function createCard(idx) {
    const el = document.createElement('div');
    el.className = 'card';
    el.dataset.idx = idx; // Key for diffing
    
    let rank, suit, colorClass;
    
    if (idx >= 52) {
        rank = idx === 52 ? '小王' : '大王';
        suit = idx === 52 ? '★' : '☀'; 
        colorClass = idx === 52 ? 'suit-black' : 'suit-red';
    } else {
        const rVal = idx % 13;
        const sVal = Math.floor(idx / 13);
        rank = RANKS[rVal];
        suit = SUITS[sVal];
        colorClass = (sVal === 0 || sVal === 1) ? 'suit-red' : 'suit-black';
    }
    
    el.setAttribute('data-suit', suit);
    
    // Adjust font size for Jokers in Chinese
    const rankStyle = idx >= 52 ? 'font-size: 0.9rem; writing-mode: vertical-rl; text-orientation: upright;' : '';

    el.innerHTML = `
        <div class="card-top ${colorClass}" style="${rankStyle}">
            ${rank}
            <span class="card-suit-sm">${suit}</span>
        </div>
        <div class="card-bottom ${colorClass}" style="${rankStyle}">
            ${rank}
            <span class="card-suit-sm">${suit}</span>
        </div>
    `;
    
    return el;
}

function toggleCard(idx, el) {
    if (systemState.selectedCards.has(idx)) {
        systemState.selectedCards.delete(idx);
        el.classList.remove('selected');
    } else {
        systemState.selectedCards.add(idx);
        el.classList.add('selected');
    }
    const last = systemState.lastData;
    if (!last) {
        updateControls(true);
        return;
    }
    const isTurn = last.current_player === 0 && !last.game_over;
    const canPass = isTurn && !last.is_new_lead;
    updateControls(isTurn, last.game_over, canPass);
}

function updateControls(isTurn, isGameOver = false, canPass = true) {
    if (isGameOver) {
        UI.playBtn.textContent = "再 局";
        UI.playBtn.disabled = false;
        UI.playBtn.onclick = restartGame;
        UI.passBtn.disabled = true;
        return;
    }
    
    // 恢复正常出牌按钮
    UI.playBtn.textContent = "出 牌";
    UI.playBtn.onclick = async () => {
        const cards = Array.from(systemState.selectedCards);
        const ok = await sendAction('play', cards);
        if (ok) systemState.selectedCards.clear();
    };
    
    if (!isTurn) {
        UI.playBtn.disabled = true;
        UI.passBtn.disabled = true;
        return;
    }
    
    UI.passBtn.disabled = !canPass;
    UI.playBtn.disabled = systemState.selectedCards.size === 0;
}

async function restartGame() {
    try {
        await fetch('/api/reset', { method: 'POST' });
        systemState.selectedCards.clear();
        fetchState();
    } catch (e) {
        console.error(e);
    }
}

function updatePrestartControls(data) {
    const show = !data.game_started && !data.game_over;
    if (show) {
        UI.prestart.classList.remove('hidden');
        UI.beginBtn.disabled = false;
        UI.shuffleBtn.disabled = false;
    } else {
        UI.prestart.classList.add('hidden');
    }
}

function renderLogs(logs) {
    // Show full history
    UI.logs.innerHTML = logs.map(l => `<div class="log-line">${l}</div>`).join('');
    // Auto scroll to bottom
    UI.logs.scrollTop = UI.logs.scrollHeight;
}

// --- Actions ---

UI.passBtn.onclick = async () => {
    const ok = await sendAction('pass');
    if (ok) systemState.selectedCards.clear();
};

UI.shuffleBtn.onclick = async () => {
    try {
        const res = await fetch('/api/deal', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert(err.detail || "重发失败");
            return;
        }
        systemState.selectedCards.clear();
        fetchState();
    } catch (e) {
        console.error(e);
    }
};

UI.beginBtn.onclick = async () => {
    try {
        const res = await fetch('/api/start', { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert(err.detail || "开局失败");
            return;
        }
        systemState.selectedCards.clear();
        fetchState();
    } catch (e) {
        console.error(e);
    }
};

async function sendAction(type, cards=[]) {
    try {
        const res = await fetch('/api/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ action: type, cards: cards })
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            const msg = err.detail || "操作失败";
            alert(msg);
            return false;
        }
        fetchState();
        return true;
    } catch (e) {
        console.error(e);
        return false;
    }
}

init();
