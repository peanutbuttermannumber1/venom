import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function Tabs({ tabs, active, onActivate, onNewTab, onClose }) {
  return (
    <div className="tabs">
      {tabs.map((t, i) => (
        <div key={i} className={`tab ${i===active ? 'active' : ''}`}>
          <span onClick={() => onActivate(i)} className="tab-title">{t.title || 'New Tab'}</span>
          <button className="tab-close" onClick={() => onClose(i)}>×</button>
        </div>
      ))}
      <button className="new-tab" onClick={onNewTab}>+</button>
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

function ChatPanel({ systemGuidelines, setSystemGuidelines, selectedEvidence, onSendMessage, chatHistory, loading }) {
  const [text, setText] = useState('');
  const send = async () => {
    if (!text.trim()) return;
    await onSendMessage(text);
    setText('');
  };

  return (
    <div className="chatpanel">
      <div className="chat-guidelines">
        <label>AI Guidelines (system):</label>
        <textarea value={systemGuidelines} onChange={(e) => setSystemGuidelines(e.target.value)} placeholder="Write rules/instructions the AI should follow..." />
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
  const [systemGuidelines, setSystemGuidelines] = useState("You are a helpful assistant. Follow the user's guidelines.");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // load saved tabs if any
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
    try {
      const r = await axios.get(`${API_BASE}/api/search`, { params: { q }});
      const newTabs = [...tabs];
      newTabs[active] = { ...newTabs[active], query: q, results: r.data.results || [] };
      setTabs(newTabs);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const onUseInAI = (item) => {
    const newTabs = [...tabs];
    const t = newTabs[active];
    // avoid duplicates
    if (!t.selectedEvidence.some(s => s.id === item.id)) {
      t.selectedEvidence = [...t.selectedEvidence, item];
      newTabs[active] = t;
      setTabs(newTabs);
    }
  };

  const onSendMessage = async (text) => {
    setLoading(true);
    try {
      const tab = tabs[active];
      const newHistory = [...tab.chatHistory, { role: 'user', content: text }];
      // prepare selectedEvidence ids
      const selectedIds = tab.selectedEvidence.map(s => s.id);
      const r = await axios.post(`${API_BASE}/api/chat`, {
        messages: newHistory,
        systemGuidelines,
        selectedEvidenceIds: selectedIds
      });
      const assistant = r.data.answer;
      const updatedTab = { ...tab, chatHistory: [...newHistory, { role: 'assistant', content: assistant }] };
      const newTabs = [...tabs];
      newTabs[active] = updatedTab;
      setTabs(newTabs);
    } catch (err) {
      console.error(err);
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
          <Results results={tab.results || []} onUse={onUseInAI} />
        </div>
      </div>

      <ChatPanel
        systemGuidelines={systemGuidelines}
        setSystemGuidelines={setSystemGuidelines}
        selectedEvidence={tab.selectedEvidence || []}
        onSendMessage={async (text) => await onSendMessage(text)}
        chatHistory={tab.chatHistory || []}
        loading={loading}
      />
    </div>
  );
}
