/* ========================================
   DriftLog — Frontend Application
   ======================================== */

const API_BASE = '';

// ── View Navigation ──

function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.getElementById(`view-${name}`).classList.add('active');
  document.querySelector(`.nav-link[data-view="${name}"]`).classList.add('active');
}

// ── Query ──

function fillQuery(chip) {
  const input = document.getElementById('query-input');
  input.value = chip.textContent;
  input.focus();
}

function toggleFilters() {
  const panel = document.getElementById('filters-panel');
  panel.classList.toggle('hidden');
}

async function submitQuery() {
  const question = document.getElementById('query-input').value.trim();
  if (!question) return;

  const location = document.getElementById('filter-location').value.trim();
  const country = document.getElementById('filter-country').value.trim();
  const tagsRaw = document.getElementById('filter-tags').value.trim();
  const tags = tagsRaw ? tagsRaw.split(',').map(t => t.trim()).filter(Boolean) : null;

  const filters = {};
  if (location) filters.location = location;
  if (country) filters.country = country;
  if (tags && tags.length) filters.tags = tags;
  const hasFilters = Object.keys(filters).length > 0;

  // Show loading, hide previous results and examples
  document.getElementById('example-queries').classList.add('hidden');
  document.getElementById('answer-section').classList.add('hidden');
  document.getElementById('query-loading').classList.remove('hidden');

  try {
    const res = await fetch(`${API_BASE}/api/v1/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        filters: hasFilters ? filters : null,
      }),
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    renderAnswer(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    document.getElementById('query-loading').classList.add('hidden');
  }
}

function renderAnswer(data) {
  const section = document.getElementById('answer-section');
  const body = document.getElementById('answer-body');
  const badge = document.getElementById('confidence-badge');
  const confText = document.getElementById('confidence-text');
  const metaChunks = document.getElementById('meta-chunks');
  const metaStrategy = document.getElementById('meta-strategy');
  const citationsList = document.getElementById('citations-list');
  const citationsSection = document.getElementById('citations-section');

  // Confidence
  const conf = Math.round((data.confidence || 0) * 100);
  confText.textContent = `${conf}% confident`;
  badge.className = 'confidence-badge';
  if (conf >= 70) badge.classList.add('confidence-high');
  else if (conf >= 40) badge.classList.add('confidence-medium');
  else badge.classList.add('confidence-low');

  // Meta
  metaChunks.textContent = `${data.chunks_retrieved} chunks → ${data.chunks_after_rerank} reranked`;
  metaStrategy.textContent = data.retrieval_strategy.replace(/_/g, ' ');

  // Answer body — convert citation markers [1], [2] to styled spans
  let answerHtml = escapeHtml(data.answer || '');
  answerHtml = answerHtml.replace(/\[(\d+)\]/g, (_, n) => {
    return `<a class="cite-ref" href="#citation-${n}" onclick="scrollToCitation(${n})">${n}</a>`;
  });
  // Wrap paragraphs
  answerHtml = answerHtml.split('\n\n').map(p => `<p>${p}</p>`).join('');
  answerHtml = answerHtml.replace(/\n/g, '<br>');
  body.innerHTML = answerHtml;

  // Citations
  citationsList.innerHTML = '';
  if (data.citations && data.citations.length > 0) {
    citationsSection.style.display = '';
    data.citations.forEach(c => {
      const card = document.createElement('div');
      card.className = 'citation-card';
      card.id = `citation-${c.index}`;
      card.innerHTML = `
        <div class="citation-index">${c.index}</div>
        <div class="citation-content">
          <div class="citation-source">${escapeHtml(c.source || 'Unknown source')}</div>
          ${c.location ? `<div class="citation-location">${escapeHtml(c.location)}</div>` : ''}
          <div class="citation-excerpt">${escapeHtml(c.excerpt || '')}</div>
        </div>
      `;
      citationsList.appendChild(card);
    });
  } else {
    citationsSection.style.display = 'none';
  }

  section.classList.remove('hidden');
}

function renderError(message) {
  const section = document.getElementById('answer-section');
  const body = document.getElementById('answer-body');
  document.getElementById('confidence-badge').className = 'confidence-badge confidence-low';
  document.getElementById('confidence-text').textContent = 'Error';
  document.getElementById('meta-chunks').textContent = '';
  document.getElementById('meta-strategy').textContent = '';
  body.innerHTML = `<p style="color: #c46c6c">${escapeHtml(message)}</p>`;
  document.getElementById('citations-section').style.display = 'none';
  section.classList.remove('hidden');
}

function scrollToCitation(n) {
  const el = document.getElementById(`citation-${n}`);
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    el.style.borderColor = 'var(--accent-warm)';
    setTimeout(() => { el.style.borderColor = ''; }, 1500);
  }
}

// ── Ingest ──

async function submitIngest() {
  const content = document.getElementById('ingest-content').value.trim();
  if (!content) return;

  const btn = document.getElementById('ingest-btn');
  btn.disabled = true;
  btn.textContent = 'Adding...';

  const doc = { content };
  const source = document.getElementById('ingest-source').value.trim();
  const location = document.getElementById('ingest-location').value.trim();
  const country = document.getElementById('ingest-country').value.trim();
  const entryDate = document.getElementById('ingest-date').value;
  const tagsRaw = document.getElementById('ingest-tags').value.trim();

  if (source) doc.source = source;
  if (location) doc.location = location;
  if (country) doc.country = country;
  if (entryDate) doc.entry_date = entryDate;
  if (tagsRaw) doc.tags = tagsRaw.split(',').map(t => t.trim()).filter(Boolean);

  try {
    const res = await fetch(`${API_BASE}/api/v1/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ documents: [doc] }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    showToast('ingest-result', 'success',
      `Added ${data.document_count} document(s) with ${data.chunk_count} chunk(s)`);

    // Reset form
    document.getElementById('ingest-content').value = '';
  } catch (err) {
    showToast('ingest-result', 'error', err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>
      Add to Knowledge Base
    `;
  }
}

// ── Journal ──

let selectedFiles = [];

function handleFiles(files) {
  for (const file of files) {
    if (!file.type.startsWith('image/')) continue;
    selectedFiles.push(file);
  }
  renderPreviews();
  document.getElementById('journal-btn').disabled = selectedFiles.length === 0;
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderPreviews();
  document.getElementById('journal-btn').disabled = selectedFiles.length === 0;
}

function renderPreviews() {
  const container = document.getElementById('image-previews');
  container.innerHTML = '';
  selectedFiles.forEach((file, i) => {
    const thumb = document.createElement('div');
    thumb.className = 'preview-thumb';
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.onload = () => URL.revokeObjectURL(img.src);
    const btn = document.createElement('button');
    btn.className = 'preview-remove';
    btn.textContent = '×';
    btn.onclick = (e) => { e.stopPropagation(); removeFile(i); };
    thumb.appendChild(img);
    thumb.appendChild(btn);
    container.appendChild(thumb);
  });
}

async function submitJournal() {
  if (selectedFiles.length === 0) return;

  const btn = document.getElementById('journal-btn');
  btn.disabled = true;
  btn.textContent = 'Transcribing...';

  try {
    const images = await Promise.all(selectedFiles.map(async (file) => {
      const base64 = await fileToBase64(file);
      return { data: base64, media_type: file.type };
    }));

    const body = { images };
    const source = document.getElementById('journal-source').value.trim();
    const location = document.getElementById('journal-location').value.trim();
    const country = document.getElementById('journal-country').value.trim();
    const entryDate = document.getElementById('journal-date').value;
    const tagsRaw = document.getElementById('journal-tags').value.trim();

    if (source) body.source = source;
    if (location) body.location = location;
    if (country) body.country = country;
    if (entryDate) body.entry_date = entryDate;
    if (tagsRaw) body.tags = tagsRaw.split(',').map(t => t.trim()).filter(Boolean);

    const res = await fetch(`${API_BASE}/api/v1/ingest/journal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    showToast('journal-result', 'success',
      `Transcribed ${data.document_count} entry/entries with ${data.chunk_count} chunk(s)`);

    // Reset
    selectedFiles = [];
    renderPreviews();
    document.getElementById('journal-files').value = '';
    btn.disabled = true;
  } catch (err) {
    showToast('journal-result', 'error', err.message);
  } finally {
    btn.disabled = selectedFiles.length === 0;
    btn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
      Transcribe &amp; Index
    `;
  }
}

// ── Drag & Drop ──

const dropZone = document.getElementById('drop-zone');

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});

// ── Keyboard ──

document.getElementById('query-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') submitQuery();
});

// ── Utilities ──

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // Strip data URL prefix to get raw base64
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function showToast(elementId, type, message) {
  const el = document.getElementById(elementId);
  el.className = `result-toast ${type}`;
  el.textContent = message;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 6000);
}
