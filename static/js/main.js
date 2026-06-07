/**
 * ================================================================
 * FAKEGUARD — MAIN JAVASCRIPT
 * File: static/js/main.js
 * Description:
 *   Handles the prediction UI, API calls, result display,
 *   character counter, and sample text injection.
 * ================================================================
 */

// ── DOM REFERENCES ──────────────────────────────────────────────
const newsInput      = document.getElementById('news-input');
const analyseBtn     = document.getElementById('analyse-btn');
const clearBtn       = document.getElementById('clear-btn');
const sampleFakeBtn  = document.getElementById('sample-fake-btn');
const sampleRealBtn  = document.getElementById('sample-real-btn');
const newAnalysisBtn = document.getElementById('new-analysis-btn');
const charCount      = document.getElementById('char-count');
const resultCard     = document.getElementById('result-card');
const detectorCard   = document.querySelector('.detector-card');

// ── SAMPLE TEXTS ────────────────────────────────────────────────
const SAMPLE_FAKE = `SHOCKING EXCLUSIVE: Scientists have confirmed that drinking a mixture of lemon juice and baking soda every morning cures all forms of cancer within 30 days. Big Pharma has been suppressing this information for decades because it would destroy their billion-dollar industry. A whistleblower from inside a major pharmaceutical company has leaked internal documents proving that executives knew about this cure since the 1970s. Share this before it gets taken down!`;

const SAMPLE_REAL = `The Federal Reserve raised its benchmark interest rate by 0.25 percentage points on Wednesday, marking the eleventh increase since March last year as the central bank continues its efforts to bring inflation back toward its 2 percent target. Fed Chair Jerome Powell said in a press conference that policymakers remain committed to restoring price stability, even as recent economic data showed signs of cooling in the labor market. The rate now stands in a range of 5.25 to 5.5 percent, the highest in 22 years.`;

// ── CHARACTER COUNTER ────────────────────────────────────────────
if (newsInput && charCount) {
  newsInput.addEventListener('input', () => {
    const len = newsInput.value.length;
    charCount.textContent = `${len} / 5000`;
    charCount.style.color = len > 4500 ? '#ef4444' : '';
  });
}

// ── CLEAR BUTTON ─────────────────────────────────────────────────
if (clearBtn) {
  clearBtn.addEventListener('click', () => {
    newsInput.value = '';
    charCount.textContent = '0 / 5000';
    newsInput.focus();
  });
}

// ── SAMPLE TEXT INJECTORS ─────────────────────────────────────────
if (sampleFakeBtn) {
  sampleFakeBtn.addEventListener('click', () => {
    newsInput.value = SAMPLE_FAKE;
    charCount.textContent = `${SAMPLE_FAKE.length} / 5000`;
    newsInput.focus();
  });
}

if (sampleRealBtn) {
  sampleRealBtn.addEventListener('click', () => {
    newsInput.value = SAMPLE_REAL;
    charCount.textContent = `${SAMPLE_REAL.length} / 5000`;
    newsInput.focus();
  });
}

// ── NEW ANALYSIS BUTTON ───────────────────────────────────────────
if (newAnalysisBtn) {
  newAnalysisBtn.addEventListener('click', () => {
    hideResult();
    newsInput.value = '';
    charCount.textContent = '0 / 5000';
    newsInput.focus();
  });
}

// ── MAIN PREDICTION LOGIC ─────────────────────────────────────────
if (analyseBtn) {
  analyseBtn.addEventListener('click', runAnalysis);

  // Allow Ctrl+Enter to trigger analysis
  newsInput && newsInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      runAnalysis();
    }
  });
}

/**
 * Sends the news text to the Flask /predict endpoint
 * and renders the result.
 */
async function runAnalysis() {
  const text = newsInput.value.trim();

  if (!text) {
    shakeTextarea();
    showToast('Please enter some news text first.', 'warning');
    return;
  }

  if (text.length < 10) {
    shakeTextarea();
    showToast('Text too short — enter at least 10 characters.', 'warning');
    return;
  }

  // Show loading state
  setLoading(true);

  const startTime = Date.now();

  try {
    const response = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ news_text: text })
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showToast(data.error || 'Server error. Please try again.', 'error');
      setLoading(false);
      return;
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    renderResult(data, elapsed);

  } catch (err) {
    console.error('Fetch error:', err);
    showToast('Network error. Is the server running?', 'error');
  } finally {
    setLoading(false);
  }
}

/**
 * Renders the prediction result card with animated bars.
 */
function renderResult(data, elapsed) {
  const isFake = data.label === 'Fake';

  // Update result card elements
  const icon     = document.getElementById('result-icon');
  const label    = document.getElementById('result-label');
  const sub      = document.getElementById('result-sub');
  const fakeBar  = document.getElementById('fake-bar');
  const realBar  = document.getElementById('real-bar');
  const fakePct  = document.getElementById('fake-pct');
  const realPct  = document.getElementById('real-pct');
  const metaWords = document.getElementById('meta-words');
  const metaModel = document.getElementById('meta-model');
  const metaTime  = document.getElementById('meta-time');
  const header    = document.getElementById('result-header');

  icon.textContent   = isFake ? '🚨' : '✅';
  label.textContent  = isFake ? 'FAKE NEWS' : 'REAL NEWS';
  sub.textContent    = `Confidence: ${data.confidence}%`;

  // Apply result class for coloring
  resultCard.classList.remove('fake-result', 'real-result');
  resultCard.classList.add(isFake ? 'fake-result' : 'real-result');
  header.classList.remove('fake-result', 'real-result');
  header.classList.add(isFake ? 'fake-result' : 'real-result');

  // Meta info
  metaWords.textContent = `📝 ${data.word_count} words`;
  metaModel.textContent = `🤖 ${data.model_used}`;
  metaTime.textContent  = `⏱ ${elapsed}s`;

  // Render explanation
  renderExplanation(data.explanation);

  // Show result card
  showResult();

  // Animate bars (after a tick to allow CSS transition)
  setTimeout(() => {
    fakeBar.style.width = data.fake_prob + '%';
    realBar.style.width = data.real_prob + '%';
    fakePct.textContent = data.fake_prob + '%';
    realPct.textContent = data.real_prob + '%';
  }, 80);

  // Update stats bar
  updateStats();

  // Scroll to result smoothly
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Renders the explanation section with key words.
 */
function renderExplanation(explanation) {
  const reasonEl = document.getElementById('explanation-reason');
  const keyWordsListEl = document.getElementById('key-words-list');

  if (!explanation || !explanation.reason) {
    reasonEl.textContent = 'Explanation not available.';
    keyWordsListEl.innerHTML = '';
    return;
  }

  reasonEl.textContent = explanation.reason;

  // Render key words
  if (explanation.key_words && explanation.key_words.length > 0) {
    keyWordsListEl.innerHTML = explanation.key_words.map(kw => {
      const directionClass = kw.direction === 'fake' ? 'fake-word' : 'real-word';
      const directionLabel = kw.direction === 'fake' ? 'F' : 'R';
      return `
        <span class="key-word ${directionClass}">
          <span class="key-word-direction">${directionLabel}</span>
          <span class="key-word-text">${kw.word}</span>
        </span>
      `;
    }).join('');
  } else {
    keyWordsListEl.innerHTML = '<span class="key-word">No key words identified</span>';
  }
}

// ── UI HELPERS ────────────────────────────────────────────────────

function showResult() {
  resultCard && resultCard.classList.remove('hidden');
}

function hideResult() {
  resultCard && resultCard.classList.add('hidden');
  // Reset bars
  const fakeBar = document.getElementById('fake-bar');
  const realBar = document.getElementById('real-bar');
  if (fakeBar) fakeBar.style.width = '0%';
  if (realBar) realBar.style.width = '0%';
}

function setLoading(loading) {
  if (!analyseBtn) return;
  const btnText   = analyseBtn.querySelector('.btn-text');
  const btnLoader = analyseBtn.querySelector('.btn-loader');

  analyseBtn.disabled = loading;
  if (btnText)   btnText.classList.toggle('hidden', loading);
  if (btnLoader) btnLoader.classList.toggle('hidden', !loading);
}

function shakeTextarea() {
  newsInput.style.animation = 'none';
  void newsInput.offsetWidth;
  newsInput.style.animation = 'shake .3s ease';
  setTimeout(() => { newsInput.style.animation = ''; }, 350);
}

// Inject shake keyframes dynamically
const shakeStyle = document.createElement('style');
shakeStyle.textContent = `
@keyframes shake {
  0%,100%{transform:translateX(0)}
  20%{transform:translateX(-6px)}
  40%{transform:translateX(6px)}
  60%{transform:translateX(-4px)}
  80%{transform:translateX(4px)}
}`;
document.head.appendChild(shakeStyle);

// ── TOAST NOTIFICATION ────────────────────────────────────────────
function showToast(message, type = 'info') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const colors = {
    info:    { bg: 'rgba(99,102,241,0.15)',  border: 'rgba(99,102,241,0.4)',  color: '#a5b4fc' },
    warning: { bg: 'rgba(251,191,36,0.12)',  border: 'rgba(251,191,36,0.35)', color: '#fbbf24' },
    error:   { bg: 'rgba(239,68,68,0.12)',   border: 'rgba(239,68,68,0.35)',  color: '#f87171' },
  };
  const c = colors[type] || colors.info;

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  Object.assign(toast.style, {
    position: 'fixed', bottom: '1.5rem', right: '1.5rem',
    background: c.bg, border: `1px solid ${c.border}`, color: c.color,
    padding: '.75rem 1.25rem', borderRadius: '10px',
    fontSize: '.875rem', fontFamily: 'DM Sans, sans-serif',
    zIndex: '999', backdropFilter: 'blur(8px)',
    animation: 'toastIn .3s ease',
    maxWidth: '320px', lineHeight: '1.4',
  });

  const toastAnim = document.createElement('style');
  toastAnim.textContent = `
@keyframes toastIn  { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
@keyframes toastOut { from{opacity:1;transform:translateY(0)} to{opacity:0;transform:translateY(10px)} }`;
  document.head.appendChild(toastAnim);

  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = 'toastOut .3s ease forwards';
    setTimeout(() => toast.remove(), 320);
  }, 3500);
}

// ── LIVE STATS REFRESH ────────────────────────────────────────────
async function updateStats() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();
    const el   = (id) => document.getElementById(id);
    if (el('stat-total')) el('stat-total').textContent = data.total;
    if (el('stat-fake'))  el('stat-fake').textContent  = data.fake;
    if (el('stat-real'))  el('stat-real').textContent  = data.real;
  } catch (_) {}
}

// ── INIT ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Focus textarea on load
  if (newsInput) newsInput.focus();

  // Set mini-fill widths from data attributes
  document.querySelectorAll('.mini-fill').forEach(el => {
    const width = el.getAttribute('data-width');
    if (width) {
      el.style.setProperty('--width', width);
    }
  });
});
