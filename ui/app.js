'use strict';

// ══ State ════════════════════════════════════════════════════════════════════
let currentSessionId    = genId();
let currentProvider     = 'groq';
let currentModel        = 'llama-3.3-70b-versatile';
let currentModelLabel   = 'Groq · Llama 3.3';
let jurisdictionOverride = null;
let isStreaming          = false;
let allModels            = {};   // full catalog from /api/models
let settings             = {};   // from /api/settings

// ══ Bootstrap ════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const [modelsData, settingsData] = await Promise.all([
      api('/api/models'),
      api('/api/settings')
    ]);
    allModels = modelsData;
    settings  = settingsData;

    currentProvider = settings.provider || 'groq';
    currentModel    = settings.model    || firstModel(currentProvider);
    jurisdictionOverride = settings.jurisdiction_override || null;

    syncModelLabel();
    renderModelDropdown();
    await refreshSessions();
  } catch (e) {
    console.error('Init failed:', e);
  }
});

// ══ Helpers ══════════════════════════════════════════════════════════════════
async function api(url, opts = {}) {
  const resp = await fetch(url, opts);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return resp.json();
}

function genId() {
  return 'sess_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function firstModel(provider) {
  const active = (settings.active_models || {})[provider] || [];
  if (active.length) return active[0];
  return (allModels[provider] || [])[0]?.id || null;
}

function providerLabel(p) {
  return { groq: 'Groq', openrouter: 'OpenRouter', ollama: 'Ollama' }[p] || p;
}

function shortModelName(id, name) {
  if (name) return name.split(' ').slice(0, 3).join(' ');
  return id.split('/').pop().split(':')[0];
}

function getActiveModels() {
  const active = settings.active_models || {};
  const result = {};
  for (const [p, catalog] of Object.entries(allModels)) {
    const allowed = new Set(active[p] || catalog.map(m => m.id));
    result[p] = catalog.filter(m => allowed.has(m.id));
  }
  return result;
}

function syncModelLabel() {
  const catalog = (allModels[currentProvider] || []);
  const m = catalog.find(m => m.id === currentModel);
  currentModelLabel =
    providerLabel(currentProvider) + ' · ' + shortModelName(currentModel, m?.name);
  const el = document.getElementById('model-label');
  if (el) el.textContent = currentModelLabel;
}

// ══ Model Dropdown ═══════════════════════════════════════════════════════════
function renderModelDropdown() {
  const dd = document.getElementById('model-dropdown');
  if (!dd) return;
  dd.innerHTML = '';

  const activeModels = getActiveModels();
  let hasAny = false;

  for (const [prov, models] of Object.entries(activeModels)) {
    if (!models.length) continue;
    hasAny = true;

    const lbl = document.createElement('div');
    lbl.className = 'model-group-label';
    lbl.textContent = providerLabel(prov);
    dd.appendChild(lbl);

    models.forEach(m => {
      const btn = document.createElement('button');
      btn.className = 'model-option' + (prov === currentProvider && m.id === currentModel ? ' selected' : '');
      btn.textContent = m.name;
      btn.onclick = e => { e.stopPropagation(); selectModel(prov, m.id, m.name); };
      dd.appendChild(btn);
    });

    const sep = document.createElement('hr');
    sep.className = 'model-separator';
    dd.appendChild(sep);
  }

  if (!hasAny) {
    dd.innerHTML = '<div class="model-group-label" style="padding:0.75rem">No active models. Enable some in Settings → Models.</div>';
  }
}

function selectModel(provider, modelId, modelName) {
  currentProvider   = provider;
  currentModel      = modelId;
  currentModelLabel = providerLabel(provider) + ' · ' + shortModelName(modelId, modelName);
  document.getElementById('model-label').textContent = currentModelLabel;
  renderModelDropdown();
  document.getElementById('model-dropdown').classList.add('hidden');
}

function toggleModelDropdown(e) {
  e.stopPropagation();
  document.getElementById('model-dropdown').classList.toggle('hidden');
}

document.addEventListener('click', () => {
  document.getElementById('model-dropdown')?.classList.add('hidden');
});

// ══ Sessions ═════════════════════════════════════════════════════════════════
async function refreshSessions() {
  try {
    const sessions = await api('/api/sessions');
    renderSessions(sessions);
  } catch (e) { /* silent */ }
}

function renderSessions(sessions) {
  const el = document.getElementById('session-list');
  if (!el) return;
  el.innerHTML = '';

  if (!sessions || !sessions.length) {
    el.innerHTML = '<div style="padding:.5rem .5rem;font-size:.75rem;color:var(--fg-muted)">No conversations yet</div>';
    return;
  }

  sessions.forEach(s => {
    const item = document.createElement('div');
    item.className = 'session-item' + (s.session_id === currentSessionId ? ' active' : '');
    item.dataset.id = s.session_id;

    const name = document.createElement('span');
    name.className = 'session-name';
    name.textContent = s.name || s.preview || s.session_id.slice(0, 10);
    name.title       = name.textContent;

    const del = document.createElement('button');
    del.className = 'session-del';
    del.textContent = '×';
    del.title = 'Delete';
    del.onclick = async ex => {
      ex.stopPropagation();
      await fetch(`/api/sessions/${s.session_id}`, { method: 'DELETE' });
      await refreshSessions();
      if (s.session_id === currentSessionId) newChat();
    };

    item.onclick = () => loadSession(s.session_id);
    item.appendChild(name);
    item.appendChild(del);
    el.appendChild(item);
  });
}

async function loadSession(id) {
  currentSessionId = id;
  document.querySelectorAll('.session-item').forEach(el =>
    el.classList.toggle('active', el.dataset.id === id));
  clearMessages();
  hideEmpty();

  try {
    const data = await api(`/api/sessions/${id}`);
    const msgs = data.messages || [];
    let lastUserQuestion = '';
    msgs.forEach(msg => {
      if (msg.role === 'user') {
        appendUserBubble(msg.content);
        lastUserQuestion = msg.content;
      } else {
        const wrap = appendAIBubble();
        const contentEl = wrap.querySelector('.msg-ai-content');
        // Remove cursor before setting text
        contentEl.querySelector('.cursor')?.remove();
        
        // Parse JURISDICTION tag if present in history message
        let content = msg.content || '';
        let msgJur = 'Both';
        const jurMatch = content.match(/JURISDICTION:\s*(India|UAE|Both|General)/i);
        if (jurMatch) {
          msgJur = jurMatch[1];
          content = content.replace(/JURISDICTION:\s*(India|UAE|Both|General)/i, '').trim();
        }

        if (typeof marked !== 'undefined') {
          contentEl.innerHTML = marked.parse(content);
        } else {
          contentEl.textContent = content;
        }

        const outer = wrap.querySelector('.msg-ai');
        if (outer) outer.dataset.question = lastUserQuestion;
        const sources = Array.isArray(msg.sources) ? msg.sources : [];
        const confidence = sources.length ? 'GROUNDED' : 'SYNTHESIZED';
        renderMetaBar(outer, { sources, confidence, jurisdiction: msgJur });
      }
    });
    scrollBottom();
  } catch (e) {
    console.error('Load session error:', e);
  }
}

function newChat() {
  currentSessionId = genId();
  clearMessages();
  showEmpty();
  document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
}

// ══ Messaging ════════════════════════════════════════════════════════════════
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function useSuggestion(btn) {
  document.getElementById('query-input').value = btn.textContent;
  sendMessage();
}

async function sendMessage() {
  if (isStreaming) return;

  const input    = document.getElementById('query-input');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  autoResize(input);
  hideEmpty();
  isStreaming = true;
  setDisabled(true);

  // Append user bubble
  appendUserBubble(question);

  // Append AI bubble with thinking indicator
  const aiWrap    = appendAIBubble();
  const outerEl   = aiWrap.querySelector('.msg-ai');
  if (outerEl) outerEl.dataset.question = question;
  const contentEl = aiWrap.querySelector('.msg-ai-content');
  const thinkEl   = appendThinking(contentEl);
  let thinkGone   = false;

  const removeThink = () => {
    if (!thinkGone) { thinkEl?.remove(); thinkGone = true; }
  };

  const updateStatus = (text) => {
    const s = contentEl.querySelector('.status-text');
    if (s) s.textContent = text;
  };

  // absolute safety watchdog (100s)
  const watchdog = setTimeout(() => {
    if (isStreaming) {
      removeThink();
      if (!fullAnswer) {
        contentEl.querySelector('.cursor')?.remove();
        contentEl.textContent = '⚠ Interaction timed out (100s). This could be a connection issue or high server load.';
      }
      finishStreaming();
    }
  }, 100000);

  let meta         = null;
  let fullAnswer   = '';
  let sessionName  = question.slice(0, 60);

  try {
    const resp = await fetch('/api/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        question,
        session_id:            currentSessionId,
        provider:              currentProvider,
        model:                 currentModel,
        jurisdiction_override: jurisdictionOverride
      })
    });

    if (!resp.ok) {
      clearTimeout(watchdog);
      removeThink();
      const err = await resp.text();
      contentEl.querySelector('.cursor')?.remove();
      contentEl.textContent = '⚠ Server error: ' + resp.status + ' - ' + err;
      return finishStreaming();
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // SSE lines are separated by \n or \r\n
      let lines = buf.split(/\r?\n/);
      buf = lines.pop(); // Keep partial line

      let evType = null;
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(':')) continue; // Ignore empty lines and pings

        if (trimmed.startsWith('event:')) {
          evType = trimmed.slice(6).trim();
        } else if (trimmed.startsWith('data:')) {
          const dataStr = trimmed.slice(5).trim();
          let data;
          try { data = JSON.parse(dataStr); } catch(e) { continue; }

          switch (evType) {
            case 'sources':
              meta = data;
              updateStatus('Synthesizing expert legal response...');
              break;

            case 'token':
              removeThink();
              if (data.content) {
                appendToken(contentEl, data.content);
                fullAnswer += data.content;
                scrollBottom();
              }
              break;

            case 'done':
              removeThink();
              sessionName = data.session_name || sessionName;
              if (meta) Object.assign(meta, { session_name: sessionName });
              break;

            case 'error':
              removeThink();
              if (!fullAnswer) {
                contentEl.querySelector('.cursor')?.remove();
                contentEl.textContent = '⚠ ' + (data.content || 'Unknown error');
              }
              break;
          }
          evType = null; // reset for next data block
        }
      }
    }
  } catch (e) {
    removeThink();
    if (!fullAnswer) {
      contentEl.querySelector('.cursor')?.remove();
      contentEl.textContent = '⚠ Network error: ' + e.message;
    }
  }

  // Finalize bubble
  clearTimeout(watchdog);
  contentEl.querySelector('.cursor')?.remove();
  finalizeAI(aiWrap, meta);

  await refreshSessions();
  finishStreaming();
  input.focus();
}

function finishStreaming() {
  isStreaming = false;
  setDisabled(false);
}

// ══ DOM Helpers ══════════════════════════════════════════════════════════════
function appendUserBubble(text) {
  const wrap = document.createElement('div');
  wrap.className = 'message-wrap';
  
  const outer = document.createElement('div');
  outer.className = 'msg-user';
  
  const bub = document.createElement('div');
  bub.className = 'msg-user-bubble';
  bub.textContent = text;
  outer.appendChild(bub);
  
  const actions = document.createElement('div');
  actions.className = 'msg-user-actions';
  const editBtn = document.createElement('button');
  editBtn.className = 'bubble-action-btn edit-btn';
  editBtn.title = 'Edit prompt';
  editBtn.innerHTML = `
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
      <path d="M18.5 2.5a2.121 2.121 0 1 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
    </svg>
  `;
  editBtn.onclick = () => {
    const input = document.getElementById('query-input');
    input.value = text;
    input.focus();
    autoResize(input);
  };
  actions.appendChild(editBtn);
  outer.appendChild(actions);
  
  wrap.appendChild(outer);
  document.getElementById('messages').appendChild(wrap);
  scrollBottom();
  return wrap;
}

function appendAIBubble() {
  const wrap = document.createElement('div');
  wrap.className = 'message-wrap';
  const outer = document.createElement('div');
  outer.className = 'msg-ai';
  const content = document.createElement('div');
  content.className = 'msg-ai-content';
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  content.appendChild(cursor);
  outer.appendChild(content);
  wrap.appendChild(outer);
  document.getElementById('messages').appendChild(wrap);
  return wrap;
}

function appendThinking(parent) {
  const el = document.createElement('div');
  el.className = 'thinking';
  el.innerHTML = `
    <div class="dots-row">
      <div class="thinking-dot"></div>
      <div class="thinking-dot"></div>
      <div class="thinking-dot"></div>
    </div>
    <div class="status-text">Searching legal archives...</div>
  `;
  parent.prepend(el);
  return el;
}

function appendToken(contentEl, token) {
  const cursor = contentEl.querySelector('.cursor');
  const node = document.createTextNode(token);
  cursor ? contentEl.insertBefore(node, cursor) : contentEl.appendChild(node);
}

function finalizeAI(wrap, meta) {
  const contentEl = wrap.querySelector('.msg-ai-content');
  let content = contentEl ? contentEl.textContent : '';
  let msgJur = meta?.jurisdiction || 'Both';
  
  // Parse JURISDICTION tag if present in final text
  const jurMatch = content.match(/JURISDICTION:\s*(India|UAE|Both|General)/i);
  if (jurMatch) {
    msgJur = jurMatch[1];
    content = content.replace(/JURISDICTION:\s*(India|UAE|Both|General)/i, '').trim();
  }
  
  if (contentEl) {
    if (typeof marked !== 'undefined') {
      contentEl.innerHTML = marked.parse(content);
    } else {
      contentEl.textContent = content;
    }
  }
  
  if (!meta) return;
  const outer = wrap.querySelector('.msg-ai');
  if (outer) {
    meta.jurisdiction = msgJur;
    renderMetaBar(outer, meta);
  }
}

function renderMetaBar(outer, meta) {
  const confidence  = meta.confidence  || 'GROUNDED';
  const jurisdiction = meta.jurisdiction || 'Both';
  const sources     = meta.sources      || [];
  const isGeneral   = (confidence === 'GENERAL' || confidence === 'SYNTHESIZED');

  const bar = document.createElement('div');
  bar.className = 'msg-meta';

  // Confidence badge
  const confMap = {
    GROUNDED:    ['badge-grounded', '✓ Verified Sources'],
    PARTIAL:     ['badge-partial',  '✦ Assisted Research'],
    SYNTHESIZED: ['badge-general',  '🗎 Independent Analysis'],
    GENERAL:     ['badge-general',  '🗎 Independent Analysis']
  };
  const [cls, lbl] = confMap[confidence] || confMap.PARTIAL;
  const confBadge = document.createElement('span');
  confBadge.className = `badge ${cls}`;
  confBadge.textContent = lbl;
  bar.appendChild(confBadge);

  // Jurisdiction badge
  const jurIcon = { India: '🇮🇳', UAE: '🇦🇪', Both: '🌐', General: '🌐' }[jurisdiction] || '🌐';
  const jurBadge = document.createElement('span');
  jurBadge.className = 'badge-jur';
  jurBadge.textContent = `${jurIcon} ${jurisdiction}`;
  bar.appendChild(jurBadge);

  // Copy button
  const copySvg = `
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>
  `;
  const checkSvg = `
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
  `;
  const copyBtn = document.createElement('button');
  copyBtn.className = 'meta-action-btn copy-btn';
  copyBtn.title = 'Copy response';
  copyBtn.innerHTML = copySvg;
  copyBtn.onclick = () => {
    const textEl = outer.querySelector('.msg-ai-content');
    const textToCopy = textEl ? textEl.innerText : '';
    navigator.clipboard.writeText(textToCopy).then(() => {
      copyBtn.innerHTML = checkSvg;
      copyBtn.style.color = 'var(--green)';
      setTimeout(() => {
        copyBtn.innerHTML = copySvg;
        copyBtn.style.color = '';
      }, 2000);
    });
  };
  bar.appendChild(copyBtn);

  // Retry button
  const retryBtn = document.createElement('button');
  retryBtn.className = 'meta-action-btn retry-btn';
  retryBtn.title = 'Retry question';
  retryBtn.innerHTML = `
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M23 4v6h-6"></path>
      <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
    </svg>
  `;
  retryBtn.onclick = () => {
    const question = outer.dataset.question || '';
    if (question) {
      document.getElementById('query-input').value = question;
      sendMessage();
    }
  };
  bar.appendChild(retryBtn);

  outer.appendChild(bar);

  // Disclaimer (Synthesized only)
  if (isGeneral) {
    const btn = document.createElement('button');
    btn.className = 'disclaimer-btn';
    btn.title = 'Disclaimer';
    btn.textContent = 'ⓘ';
    bar.appendChild(btn);

    const box = document.createElement('div');
    box.className = 'disclaimer-box';
    box.textContent = 'This response is synthesized from the AI model\'s general knowledge, rather than being retrieved from LexRAG\'s verified document base. It may not reflect the most current statutory provisions. Always consult a qualified legal professional before acting.';
    btn.onclick = () => box.classList.toggle('visible');
    outer.appendChild(box);
  }

  // Sources toggle
  if (sources.length) {
    const toggle = document.createElement('button');
    toggle.className = 'sources-toggle';
    toggle.textContent = `${sources.length} source${sources.length > 1 ? 's' : ''} ↓`;
    bar.appendChild(toggle);

    const panel = document.createElement('div');
    panel.className = 'sources-panel';
    sources.forEach(s => {
      const item = document.createElement('div');
      item.className = 'source-item';
      item.innerHTML = `
        <span>${s.title || s.source || 'Source'}</span>
        <span class="badge-jur" style="font-size:.55rem">${s.jurisdiction || ''}</span>
        <span class="source-score">${s.score != null ? s.score.toFixed(2) : ''}</span>
        ${s.url ? `<a href="${s.url}" target="_blank" class="source-link" onclick="event.stopPropagation()">↗</a>` : ''}
      `;
      panel.appendChild(item);
    });

    toggle.onclick = () => {
      const open = panel.classList.toggle('visible');
      toggle.textContent = `${sources.length} source${sources.length > 1 ? 's' : ''} ${open ? '↑' : '↓'}`;
    };
    outer.appendChild(panel);
  }
}

// ══ Settings ═════════════════════════════════════════════════════════════════
function openSettings() {
  document.getElementById('settings-overlay').classList.remove('hidden');
  const panel = document.getElementById('settings-panel');
  panel.classList.remove('hidden');
  requestAnimationFrame(() => panel.classList.add('visible'));
  populateSettings();
}

function closeSettings() {
  const panel = document.getElementById('settings-panel');
  panel.classList.remove('visible');
  setTimeout(() => {
    panel.classList.add('hidden');
    document.getElementById('settings-overlay').classList.add('hidden');
  }, 300);
}

function switchSettingsTab(tab, btn) {
  document.querySelectorAll('.stab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(`stab-${tab}`).classList.remove('hidden');
  btn.classList.add('active');
}

// ── General Tab ───────────────────────────────────────────────────────────
function populateSettings() {
  // Provider pills
  const pills = document.getElementById('s-provider-pills');
  if (pills) {
    pills.innerHTML = '';
    ['groq', 'openrouter', 'ollama'].forEach(p => {
      const btn = document.createElement('button');
      btn.className = 'pill-btn' + (p === (settings.provider || currentProvider) ? ' active' : '');
      btn.textContent = providerLabel(p);
      btn.onclick = () => {
        document.querySelectorAll('#s-provider-pills .pill-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        populateSettingsModelSelect(p);
      };
      pills.appendChild(btn);
    });
  }
  populateSettingsModelSelect(settings.provider || currentProvider);

  // Jurisdiction
  document.querySelectorAll('#s-jur-pills button[data-jur]').forEach(btn => {
    const v = btn.dataset.jur === 'null' ? null : btn.dataset.jur;
    btn.classList.toggle('active', v === jurisdictionOverride);
  });

  // Models tab
  populateModelsTab();

  // Custom tab
  populateCustomTab();
}

function populateSettingsModelSelect(provider) {
  const sel = document.getElementById('s-model-select');
  if (!sel) return;
  sel.innerHTML = '';
  (allModels[provider] || []).forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.name;
    opt.selected = m.id === (settings.model || currentModel);
    sel.appendChild(opt);
  });
}

function setJurSetting(btn) {
  document.querySelectorAll('#s-jur-pills button[data-jur]').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  jurisdictionOverride = btn.dataset.jur === 'null' ? null : btn.dataset.jur;
}

// ── Models Tab ────────────────────────────────────────────────────────────
function populateModelsTab() {
  const list = document.getElementById('s-models-list');
  if (!list) return;
  list.innerHTML = '';

  const activeModels = settings.active_models || {};

  for (const [prov, models] of Object.entries(allModels)) {
    if (!models.length) continue;
    const group = document.createElement('div');
    group.className = 'model-provider-group';

    const title = document.createElement('div');
    title.className = 'model-provider-title';
    title.textContent = providerLabel(prov);
    group.appendChild(title);

    const activeSet = new Set(activeModels[prov] || models.map(m => m.id));

    models.forEach(m => {
      const row = document.createElement('div');
      row.className = 'model-toggle-row';

      const nameEl = document.createElement('span');
      nameEl.className = 'model-toggle-name';
      nameEl.textContent = m.name;

      const idEl = document.createElement('span');
      idEl.className = 'model-toggle-id';
      idEl.textContent = m.id.split('/').pop().split(':')[0];

      const sw = document.createElement('label');
      sw.className = 'toggle-switch';
      sw.title = activeSet.has(m.id) ? 'Active — click to deactivate' : 'Inactive — click to activate';

      const inp = document.createElement('input');
      inp.type    = 'checkbox';
      inp.checked = activeSet.has(m.id);
      inp.dataset.provider = prov;
      inp.dataset.modelId  = m.id;

      const slider = document.createElement('span');
      slider.className = 'toggle-slider';

      sw.appendChild(inp);
      sw.appendChild(slider);
      row.appendChild(nameEl);
      row.appendChild(idEl);
      row.appendChild(sw);
      group.appendChild(row);
    });

    list.appendChild(group);
  }
}

// ── Custom Tab ────────────────────────────────────────────────────────────
function populateCustomTab() {
  const list = document.getElementById('s-custom-list');
  if (!list) return;
  list.innerHTML = '';

  const custom = settings.custom_models || {};
  let any = false;
  for (const [prov, models] of Object.entries(custom)) {
    models.forEach(m => {
      any = true;
      const item = document.createElement('div');
      item.className = 'custom-model-item';
      item.innerHTML = `<span>${m.name} <span style="color:var(--fg-muted);font-size:.75rem">(${providerLabel(prov)})</span></span>`;
      const del = document.createElement('button');
      del.className = 'custom-del-btn';
      del.textContent = '×';
      del.onclick = () => removeCustomModel(prov, m.id);
      item.appendChild(del);
      list.appendChild(item);
    });
  }
  if (!any) {
    list.innerHTML = '<div style="padding:1rem 1.5rem;font-size:.8rem;color:var(--fg-muted)">No custom models added yet.</div>';
  }
}

async function addCustomModel() {
  const provider = document.getElementById('c-provider').value;
  const id       = document.getElementById('c-model-id').value.trim();
  const name     = document.getElementById('c-model-name').value.trim();
  if (!id || !name) { alert('Please enter both Model ID and display name.'); return; }

  const custom = settings.custom_models || {};
  if (!custom[provider]) custom[provider] = [];

  // Prevent duplicates
  if (custom[provider].find(m => m.id === id)) { alert('Model already added.'); return; }
  custom[provider].push({ id, name });

  // Also add to allModels catalog client-side
  if (!allModels[provider]) allModels[provider] = [];
  if (!allModels[provider].find(m => m.id === id)) allModels[provider].push({ id, name });

  // Activate by default
  const active = settings.active_models || {};
  if (!active[provider]) active[provider] = [];
  if (!active[provider].includes(id)) active[provider].push(id);

  settings.custom_models = custom;
  settings.active_models = active;

  document.getElementById('c-model-id').value    = '';
  document.getElementById('c-model-name').value  = '';

  // Auto-save and refresh everything
  await saveSettings(true); // pass true to skip closing the panel
  populateSettings(); // Refresh all tabs
}

async function removeCustomModel(provider, modelId) {
  const custom = settings.custom_models || {};
  if (custom[provider]) {
    custom[provider] = custom[provider].filter(m => m.id !== modelId);
  }
  const active = settings.active_models || {};
  if (active[provider]) {
    active[provider] = active[provider].filter(id => id !== modelId);
  }
  if (allModels[provider]) {
    allModels[provider] = allModels[provider].filter(m => m.id !== modelId);
  }
  settings.custom_models = custom;
  settings.active_models = active;
  if (allModels[provider]) {
    allModels[provider] = allModels[provider].filter(m => m.id !== modelId);
  }
  
  await saveSettings(true);
  populateSettings();
}

async function saveSettings(skipClose = false) {
  // Read active provider from pills
  const activePill = document.querySelector('#s-provider-pills .pill-btn.active');
  const provLabels = ['groq', 'openrouter', 'ollama'];
  let selProvider = settings.provider || currentProvider;
  if (activePill) {
    const idx = [...document.querySelectorAll('#s-provider-pills .pill-btn')].indexOf(activePill);
    if (idx >= 0) selProvider = provLabels[idx];
  }

  const selModel = document.getElementById('s-model-select')?.value || currentModel;

  // Collect active models from toggles
  const activeModels = {};
  document.querySelectorAll('.toggle-switch input[type="checkbox"]').forEach(inp => {
    const prov = inp.dataset.provider;
    const id   = inp.dataset.modelId;
    if (!activeModels[prov]) activeModels[prov] = [];
    if (inp.checked) activeModels[prov].push(id);
  });

  const toSave = {
    provider:              selProvider,
    model:                 selModel,
    jurisdiction_override: jurisdictionOverride,
    active_models:         activeModels,
    custom_models:         settings.custom_models || {}
  };

  try {
    const updated = await api('/api/settings', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(toSave)
    });
    settings            = updated;
    currentProvider     = selProvider;
    currentModel        = selModel;
    syncModelLabel();
    renderModelDropdown();
    if (!skipClose) closeSettings();
  } catch (e) {
    alert('Failed to save settings: ' + e.message);
  }
}

// ══ Utils ═════════════════════════════════════════════════════════════════════
function clearMessages() { document.getElementById('messages').innerHTML = ''; }
function showEmpty()   { document.getElementById('empty-state').classList.remove('hidden'); }
function hideEmpty()   { document.getElementById('empty-state').classList.add('hidden');    }
function scrollBottom() {
  const el = document.getElementById('messages');
  el.scrollTop = el.scrollHeight;
}
function setDisabled(v) {
  document.getElementById('send-btn').disabled    = v;
  document.getElementById('query-input').disabled = v;
}
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}
