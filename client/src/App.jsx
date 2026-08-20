import React, { useState, useEffect } from 'react';
import axios from 'axios';

// Use relative base by default so the app works when served from the same origin
const API_BASE = import.meta.env.VITE_API_BASE || '';

function Tabs({ tabs, active, onActivate, onNewTab, onClose }) {
  return (
    <div className="tabs">
      {tabs.map((t, i) => (
        <div key={i} className={`tab ${i===active ? 'active' : ''}`}>
          <span onClick={() => onActivate(i)} className="tab-title">{t.title || 'New Tab'}</span>
          <button className="tab-close" onClick={() => onClose(i)}>×</button>
        </div>
      ))}
      <button className="new-tab" onClick={onNewTab} aria-label="New tab">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>
      </button>
    </div>
  );
}

function SearchBar({ onSearch, value, setValue }) {
  return (
    <form className="searchbar" onSubmit={(e) => { e.preventDefault(); onSearch(value); }}>
      <input value={value} onChange={(e) => setValue(e.target.value)} placeholder="Search Dingo (privacy-friendly)..." />
      <button type="submit">Search</button>
    </form>
  );
}

function Results({ results, onUse }) {
  return (
    <div className="results">
      {results.map((r, i) => (
        <div key={r.id || i} className="result">
          <a href={r.url} target="_blank" rel="noreferrer" className="result-title">{r.title}</a>
          <div className="result-link">{r.url}</div>
          <div className="result-snippet">{r.snippet}</div>
          <div className="result-actions">
            <button onClick={() => onUse(r)}>Use in AI</button>
            <span className="trust">Trust: {(r.trust||0).toFixed(2)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ChatPanel({ systemGuidelines, setSystemGuidelines, selectedEvidence, onSendMessage, chatHistory, loading, captureScreen, onGenerateSite }) {
  const [text, setText] = useState('');
  const send = async () => {
    if (!text.trim()) return;
    await onSendMessage(text);
    setText('');
  };

  const generateSite = async () => {
    const topic = window.prompt('Enter a topic for the site (e.g., "How to start composting")');
    if (!topic) return;
    const tone = window.prompt('Tone (informative / casual / authoritative)', 'informative') || 'informative';
    const length = window.prompt('Length (short / medium / long)', 'medium') || 'medium';
    try {
      const r = await axios.post(`${API_BASE}/api/build_site`, { topic, tone, length }, { responseType: 'blob' });
      const blob = new Blob([r.data], { type: 'application/zip' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dingo_site_${topic.replace(/\s+/g,'_').toLowerCase()}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      alert('Site ZIP downloaded. Extract and deploy to your hosting provider.');
    } catch (err) {
      console.error(err);
      alert('Failed to build site: ' + (err.response?.data || err.message));
    }
  };

  return (
    <div className="chatpanel">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
        <div className="chat-guidelines" style={{flex:1}}>
          <label>AI Guidelines (system):</label>
          <textarea value={systemGuidelines} onChange={(e) => setSystemGuidelines(e.target.value)} placeholder="Write rules/instructions the AI should follow..." />
        </div>
        <div style={{marginLeft:12, display:'flex', flexDirection:'column', gap:8}}>
          <button onClick={captureScreen} title="Capture visible app content">📸</button>
          <button onClick={generateSite} title="Generate SEO-optimized site">🏗️</button>
        </div>
      </div>
      <div className="chat-history">
        {chatHistory.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>{m.role === 'user' ? 'You: ' : 'AI: '}{m.content}</div>
        ))}
      </div>
      <div className="chat-evidence">
        <label>Selected evidence to include:</label>
        <ul>
          {selectedEvidence.map((s) => <li key={s.id}><a href={s.url} target="_blank" rel="noreferrer">{s.title}</a></li>)}
        </ul>
      </div>
      <div className="chat-input">
        <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask the AI..." />
        <button onClick={send} disabled={loading}>{loading ? '...' : 'Send'}</button>
      </div>
    </div>
  );
}

export default function App() {
  const [tabs, setTabs] = useState([{ title: 'Tab 1', query: '', results: [], selectedEvidence: [], chatHistory: [] }]);
  const [active, setActive] = useState(0);
  const [query, setQuery] = useState('');
  const [systemGuidelines, setSystemGuidelines] = useState("You are a helpful assistant. Follow the user's guidelines. Do not generate content depicting severe violence. Do NOT produce image-generation outputs.");
  const [loading, setLoading] = useState(false);
  const [aiOpen, setAiOpen] = useState(false);
  const [screenCaptureText, setScreenCaptureText] = useState('');
  const [dingoSummary, setDingoSummary] = useState(null);

  useEffect(() => {
    const saved = localStorage.getItem('dingo_tabs_v1');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setTabs(parsed);
      } catch {}
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('dingo_tabs_v1', JSON.stringify(tabs));
  }, [tabs]);

  const doSearch = async (q) => {
    setLoading(true);
    setDingoSummary(null);
    try {
      const r = await axios.get(`${API_BASE}/api/search`, { params: { q }});
      const newTabs = [...tabs];
      newTabs[active] = { ...newTabs[active], query: q, results: r.data.results || [] };
      setTabs(newTabs);
      // fetch dingo.ai-style summary and show above results
      try {
        const s = await axios.get(`${API_BASE}/api/summarize`, { params: { q }});
        setDingoSummary(s.data);
      } catch (err) {
        console.warn('Dingo summary failed', err);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const onUseInAI = (item) => {
    const newTabs = [...tabs];
    const t = newTabs[active];
    if (!t.selectedEvidence.some(s => s.id === item.id)) {
      t.selectedEvidence = [...t.selectedEvidence, item];
      newTabs[active] = t;
      setTabs(newTabs);
    }
  };

  const captureScreen = () => {
    const tab = tabs[active];
    const parts = [];
    parts.push(`Active Tab: ${tab.title || 'Tab ' + (active+1)}`);
    if (tab.query) parts.push(`Query: ${tab.query}`);
    const topResults = (tab.results || []).slice(0,5).map(r => `- ${r.title} (${r.url})`).join('\n');
    if (topResults) parts.push(`Top results:\n${topResults}`);
    const snapshot = parts.join('\n\n');
    setScreenCaptureText(snapshot);
    alert('Screen snapshot captured and will be included in next AI request.');
  };

  const onSendMessage = async (text) => {
    setLoading(true);
    try {
      const tab = tabs[active];
      const newHistory = [...tab.chatHistory, { role: 'user', content: text }];
      const selectedIds = tab.selectedEvidence.map(s => s.id);
      const r = await axios.post(`${API_BASE}/api/chat`, {
        messages: newHistory,
        systemGuidelines,
        selectedEvidenceIds: selectedIds,
        screen_text: screenCaptureText || ''
      });
      const assistant = r.data.answer;
      const updatedTab = { ...tab, chatHistory: [...newHistory, { role: 'assistant', content: assistant }] };
      const newTabs = [...tabs];
      newTabs[active] = updatedTab;
      setTabs(newTabs);
      setScreenCaptureText('');
    } catch (err) {
      console.error(err);
      alert('AI request failed: ' + (err.response?.data || err.message));
    } finally {
      setLoading(false);
    }
  };

  const addTab = () => {
    const newTabs = [...tabs, { title: `Tab ${tabs.length+1}`, query: '', results: [], selectedEvidence: [], chatHistory: [] }];
    setTabs(newTabs);
    setActive(newTabs.length - 1);
  };

  const closeTab = (i) => {
    if (tabs.length === 1) return;
    const newTabs = tabs.filter((_, idx) => idx !== i);
    setTabs(newTabs);
    setActive(Math.max(0, active - (i <= active ? 1 : 0)));
  };

  const activateTab = (i) => setActive(i);

  const tab = tabs[active];

  return (
    <div className="app">
      <div className="browser">
        <Tabs tabs={tabs} active={active} onActivate={activateTab} onNewTab={addTab} onClose={closeTab} />
        <div className="addressbar">
          <SearchBar onSearch={doSearch} value={query} setValue={setQuery} />
        </div>
        <div className="content">
          {loading && <div className="loader">Loading...</div>}
          {/* Dingo.ai summary card */}
          {dingoSummary && (
            <div className="dingo-card">
              <div className="title">Dingo.ai — Quick summary</div>
              <div className="summary">{dingoSummary.summary}</div>
            </div>
          )}
          <Results results={tab.results || []} onUse={onUseInAI} />
        </div>
      </div>

      {aiOpen && (
        <ChatPanel
          systemGuidelines={systemGuidelines}
          setSystemGuidelines={setSystemGuidelines}
          selectedEvidence={tab.selectedEvidence || []}
          onSendMessage={async (text) => await onSendMessage(text)}
          chatHistory={tab.chatHistory || []}
          loading={loading}
          captureScreen={captureScreen}
        />
      )}

      <button className="fab-ai" onClick={async () => {
        // check health and open panel
        try {
          const h = await axios.get(`${API_BASE}/api/health`);
          if (!h.data.llm.any) {
            if (!confirm('No LLM configured. OpenAI key not set. Continue and use search-only features?')) return;
          }
        } catch (err) {
          console.warn('Health check failed', err);
        }
        setAiOpen(!aiOpen);
      }} title="Toggle Dingo AI">AI</button>
    </div>
  );
}
