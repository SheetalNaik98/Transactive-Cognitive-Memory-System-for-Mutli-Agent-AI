#!/usr/bin/env python3
# Orchestrix — TCM + LLM Core Memory

import os, time, hashlib, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict, deque

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import streamlit as st

# ---------------- Page / Theme ----------------
st.set_page_config(page_title="Orchestrix", layout="wide")

def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300..800&display=swap');
      
      /* Global font and text visibility fixes */
      * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }
      
      /* Modern gradient background */
      [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f0f 100%);
      }
      
      [data-testid="stHeader"] { 
        background: transparent;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }

      /* FIXED: Enhanced color palette for maximum visibility */
      :root {
        --background: #0a0a0a;
        --surface: rgba(255, 255, 255, 0.06);
        --surface-hover: rgba(255, 255, 255, 0.09);
        --border: rgba(255, 255, 255, 0.15);
        --border-hover: rgba(255, 255, 255, 0.25);
        
        /* CRITICAL FIX: Brighter text colors */
        --text-primary: #ffffff;
        --text-secondary: #b8bcc4;
        --text-muted: #9ca3af;
        
        /* Vibrant accent colors */
        --accent-blue: #60a5fa;
        --accent-green: #34d399;
        --accent-purple: #a78bfa;
        --accent-amber: #fbbf24;
        --accent-red: #f87171;
        
        /* Premium gradients */
        --gradient-brand: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-success: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        --gradient-danger: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
      }

      /* Brand header with animation */
      .brandwrap { 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        padding: 32px 0 24px; 
        margin-bottom: 24px;
      }
      
      .logotype {
        font-weight: 800;
        font-size: 48px;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 25%, #f472b6 50%, #fbbf24 75%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        background-size: 200% 200%;
        animation: gradient-shift 6s ease infinite;
        filter: drop-shadow(0 10px 30px rgba(139, 92, 246, 0.3));
      }
      
      @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
      }
      
      .tag { 
        text-align: center; 
        color: var(--text-secondary);
        font-size: 16px;
        margin-top: 8px;
        letter-spacing: 0.5px;
        font-weight: 500;
      }

      /* Glass morphism cards */
      .shell {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 
          0 8px 32px rgba(0, 0, 0, 0.4),
          inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      .shell:hover {
        background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
        border-color: var(--border-hover);
        transform: translateY(-2px);
        box-shadow: 
          0 12px 48px rgba(0, 0, 0, 0.5),
          inset 0 1px 0 rgba(255, 255, 255, 0.15);
      }
      
      .shell.soft { 
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
      }

      /* Disclaimer with better visibility */
      .disclaimer {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15) 0%, rgba(167, 139, 250, 0.15) 100%);
        border: 1px solid rgba(167, 139, 250, 0.4);
        border-radius: 12px;
        padding: 18px;
        color: var(--text-primary);
        margin-bottom: 24px;
        font-size: 14px;
        line-height: 1.7;
        font-weight: 400;
      }

      /* Metrics grid */
      .grid { 
        display: grid; 
        gap: 20px; 
        grid-template-columns: 1fr;
        margin-bottom: 24px;
      }

      /* Metric chips with gradient borders */
      .chip {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        transform: translateY(0);
        opacity: 1;
      }
      
      .chip::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-brand);
        opacity: 0;
        transition: opacity 0.3s ease;
      }
      
      .chip:hover {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
        border-color: var(--border-hover);
        transform: translateY(-2px) scale(1.02);
      }
      
      .chip:hover::before {
        opacity: 1;
      }
      
      .chip .label { 
        color: var(--text-muted);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 12px;
        opacity: 0.9;
      }
      
      .chip .value { 
        color: var(--text-primary);
        font-size: 36px;
        font-weight: 700;
        letter-spacing: -1px;
        line-height: 1;
      }

      /* Status badges */
      .status { 
        display: flex; 
        gap: 10px; 
        align-items: center; 
        flex-wrap: wrap;
        margin: 20px 0;
      }
      
      .badge {
        display: inline-flex;
        align-items: center;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        color: var(--text-primary);
        background: rgba(255, 255, 255, 0.05);
      }
      
      .badge:hover {
        transform: translateY(-1px);
        background: rgba(255, 255, 255, 0.08);
      }
      
      .badge-topic { 
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15), rgba(96, 165, 250, 0.10));
        color: #93bbfc;
        border: 1px solid rgba(96, 165, 250, 0.3);
      }
      
      .badge-mem { 
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(251, 191, 36, 0.10));
        color: #fcd34d;
        border: 1px solid rgba(251, 191, 36, 0.3);
      }
      
      .delegation.ok { 
        background: var(--gradient-success);
        color: white;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
      }
      
      .delegation.bad { 
        background: var(--gradient-danger);
        color: white;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
      }

      /* Answer boxes */
      .answer {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        color: var(--text-primary);
        font-size: 15px;
        line-height: 1.8;
        margin-top: 16px;
      }
      
      .answer p {
        color: var(--text-primary) !important;
      }

      /* Agent cards */
      .agent {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
        transition: all 0.3s ease;
      }
      
      .agent:hover {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.05));
        border-color: var(--border-hover);
        transform: translateX(4px);
      }
      
      .agent .name { 
        color: var(--text-primary);
        font-weight: 600;
        font-size: 15px;
        margin-bottom: 10px;
      }
      
      /* Progress bars */
      .bar { 
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
        overflow: hidden;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
      }
      
      .bar > span {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, #ef4444 0%, #fbbf24 33%, #34d399 66%, #10b981 100%);
        border-radius: 3px;
        transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
      }

      .hint { 
        color: var(--text-muted);
        font-size: 13px;
        margin-top: 10px;
        line-height: 1.5;
      }

      /* Routing overlay */
      .overlay {
        position: fixed;
        inset: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0, 0, 0, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
      }
      
      .routebox {
        background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 36px;
        max-width: 500px;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6);
      }
      
      .routebox > div:first-child {
        color: var(--text-primary);
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 14px;
      }
      
      .small { 
        color: var(--text-secondary);
        font-size: 14px;
        line-height: 1.7;
      }

      /* Section titles */
      .sectiontitle {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 17px;
        margin-bottom: 14px;
      }
      
      /* Fix ALL Streamlit text elements */
      h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
      }
      
      p, span, div, label, .st-emotion-cache-10trblm {
        color: var(--text-secondary) !important;
      }
      
      /* Streamlit specific fixes */
      .stTextArea textarea {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
        font-family: 'Inter', sans-serif !important;
      }
      
      .stTextArea textarea:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.2) !important;
        background: rgba(255, 255, 255, 0.08) !important;
      }
      
      .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
      }
      
      .stButton button {
        background: var(--gradient-brand);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.3);
      }
      
      .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
      }
      
      /* Checkbox styling */
      .stCheckbox {
        color: var(--text-secondary) !important;
      }
      
      .stCheckbox > label {
        color: var(--text-secondary) !important;
      }
      
      /* Expander styling */
      .streamlit-expanderHeader {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px;
      }
      
      .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08) !important;
      }
      
      /* Info messages */
      .stAlert {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.1), rgba(167, 139, 250, 0.1));
        border: 1px solid rgba(96, 165, 250, 0.3);
        color: var(--text-primary);
      }
      
      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.03);
        border-right: 1px solid var(--border);
      }
      
      section[data-testid="stSidebar"] .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
      }
      
      /* Animations */
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      
      .shell, .chip, .agent {
        animation: fadeIn 0.5s ease;
      }
      
      /* Ensure all text is visible */
      .stMarkdown {
        color: var(--text-primary) !important;
      }
      
      /* Column gaps */
      .row-widget.stHorizontal {
        gap: 20px;
      }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------------- API key helper ----------------
def require_api_key():
    key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if key:
        os.environ["OPENAI_API_KEY"] = key.strip()
        return
    with st.sidebar:
        st.write("🔑 OpenAI API Key")
        val = st.text_input("Enter key", type="password", placeholder="sk-...", help="Used to call the model for answers and embeddings.")
        if val:
            os.environ["OPENAI_API_KEY"] = val.strip()
            st.success("✅ Saved for this session.")
        else:
            st.warning("⚠️ Please enter your OpenAI API key to continue.")
            st.stop()

require_api_key()

# ---------------- Engine (in-file) ----------------
@dataclass
class MemoryEntry:
    id: str
    content: str
    embedding: np.ndarray
    topic: str
    agent_id: str
    timestamp: float
    access_count: int = 0
    memory_type: str = "episodic"  # episodic | semantic | procedural
    metadata: Dict = field(default_factory=dict)

class LLMCoreMemoryLite:
    def __init__(self, client: OpenAI, embed_model: str = "text-embedding-3-small"):
        self.client = client
        self.embed_model = embed_model
        self.working_memory = deque(maxlen=10)
        self.episodic: List[MemoryEntry] = []
        self.semantic: Dict[str, List[MemoryEntry]] = {}
        self.procedural: Dict[str, MemoryEntry] = {}
        self._embed_cache: Dict[str, np.ndarray] = {}
        self.consolidation_threshold = 5

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def _embed(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._embed_cache:
            return self._embed_cache[key]
        resp = self.client.embeddings.create(model=self.embed_model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        self._embed_cache[key] = vec
        return vec

    def add_memory(self, content: str, topic: str, agent_id: str,
                   memory_type: str = "episodic") -> str:
        emb = self._embed(content)
        mem_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:10]
        entry = MemoryEntry(
            id=mem_id, content=content, embedding=emb,
            topic=topic, agent_id=agent_id, timestamp=time.time(),
            memory_type=memory_type
        )
        if memory_type == "episodic":
            self.episodic.append(entry)
            self.working_memory.append(mem_id)
        elif memory_type == "semantic":
            self.semantic.setdefault(topic, []).append(entry)
        else:
            self.procedural[topic] = entry
        return mem_id

    def _all_entries(self) -> List[MemoryEntry]:
        out = list(self.episodic)
        for arr in self.semantic.values():
            out.extend(arr)
        out.extend(self.procedural.values())
        return out

    def retrieve(self, query: str, k: int = 5) -> List[MemoryEntry]:
        entries = self._all_entries()
        if not entries:
            return []
        q = self._embed(query)
        mats = np.stack([e.embedding for e in entries], axis=0)
        sims = mats @ q / (np.linalg.norm(mats, axis=1) * np.linalg.norm(q) + 1e-9)
        idxs = np.argsort(-sims)[:k]
        results = []
        for i in idxs:
            e = entries[i]
            e.access_count += 1
            results.append(e)
        return results

    def consolidate(self) -> int:
        moved = 0
        keep: List[MemoryEntry] = []
        for e in self.episodic:
            if e.access_count >= self.consolidation_threshold:
                self.semantic.setdefault(e.topic, []).append(
                    MemoryEntry(
                        id=f"cons_{e.id}",
                        content=f"[Consolidated] {e.content}",
                        embedding=e.embedding,
                        topic=e.topic,
                        agent_id=e.agent_id,
                        timestamp=time.time(),
                        memory_type="semantic",
                        metadata={"orig": e.id, "access_count": e.access_count},
                    )
                )
                moved += 1
            else:
                keep.append(e)
        self.episodic = keep
        return moved

class TCMWithLLMMemoryLite:
    def __init__(self, agents: List[str], topics: List[str],
                 chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.chat_model = chat_model
        self.agents = agents
        self.topics = topics
        self.trust = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})
        self.mem_local = {a: LLMCoreMemoryLite(self.client) for a in agents}
        self.mem_shared = LLMCoreMemoryLite(self.client)
        self.metrics = {"delegations": 0, "total": 0, "mems_used": [], "hit_rate": [], "consolidations": 0}

    def _topic(self, text: str) -> str:
        t = text.lower()
        rules = {
            "planning": ["plan", "roadmap", "strategy", "schedule"],
            "research": ["research", "investigate", "study", "analyze"],
            "coding":   ["code", "implement", "bug", "debug", "write python"],
            "ml":       ["ml", "model", "train", "neural", "classifier"],
            "nlp":      ["nlp", "transformer", "llm", "token", "text"],
        }
        for topic, kws in rules.items():
            if any(kw in t for kw in kws): return topic
        return self.topics[0] if self.topics else "general"

    def thompson_draws(self, topic: str, seed: Optional[int] = None) -> Dict[str, float]:
        if seed is not None: np.random.seed(seed)
        draws = {}
        for a in self.agents:
            p = self.trust[f"{a}:{topic}"]
            draws[a] = float(np.random.beta(p["alpha"], p["beta"]))
        return draws

    def _expert(self, topic: str) -> str:
        scores = self.thompson_draws(topic)
        return max(scores, key=scores.get)

    def _format_mem(self, mems: List[MemoryEntry]) -> str:
        if not mems: return "No relevant memories."
        return "\n".join([f"{i}. ({m.memory_type}) {m.content[:220]}..." for i, m in enumerate(mems[:5], 1)])

    def _call_llm(self, prompt: str) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=500,
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"[LLM error] {e}"

    def _quality(self, response: str, mems: List[MemoryEntry]) -> float:
        score = 0.5
        if len(response) > 120: score += 0.2
        if mems and any(m.content[:60] in response for m in mems): score += 0.3
        return min(1.0, score)

    def _update_trust(self, agent: str, topic: str, success: bool):
        k = f"{agent}:{topic}"
        if success: self.trust[k]["alpha"] += 1
        else:        self.trust[k]["beta"]  += 1

    def trust_score(self, agent: str, topic: str) -> float:
        p = self.trust[f"{agent}:{topic}"]
        return p["alpha"] / (p["alpha"] + p["beta"])

    def process(self, query: str, requester: Optional[str] = None) -> Dict:
        self.metrics["total"] += 1
        topic = self._topic(query)
        requester = requester or random.choice(self.agents)
        expert = self._expert(topic)
        delegated = expert != requester
        if delegated: self.metrics["delegations"] += 1

        local = self.mem_local[expert].retrieve(query, k=3)
        shared = self.mem_shared.retrieve(query, k=2)
        used = local + shared

        prompt = f"""You are {expert}, an expert in {topic}.
Relevant memories:
{self._format_mem(used)}

User query:
{query}

Craft a helpful, accurate answer that uses the memories when relevant.
"""
        answer = self._call_llm(prompt)

        self.mem_local[expert].add_memory(
            content=f"Q: {query}\nA: {answer}",
            topic=topic, agent_id=expert, memory_type="episodic"
        )

        q_score = self._quality(answer, used)
        self._update_trust(expert, topic, success=(q_score > 0.7))

        self.metrics["consolidations"] += self.mem_local[expert].consolidate()
        self.metrics["consolidations"] += self.mem_shared.consolidate()

        hit = (len(local) / max(1, len(used))) if used else 0.0
        self.metrics["mems_used"].append(len(used))
        self.metrics["hit_rate"].append(hit)

        return {
            "query": query, "response": answer, "topic": topic, "requester": requester,
            "expert": expert, "delegated": delegated, "memories_used": len(used),
            "trust_score": self.trust_score(expert, topic),
            "mem_snippets": [m.content for m in used]
        }

# Baseline (no routing, no memory) — for comparison
class DirectModel:
    def __init__(self, chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.chat_model = chat_model
    def answer(self, q: str) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role":"user","content":q}],
                temperature=0.7, max_tokens=500
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"[LLM error] {e}"

# ---------------- Engine cache ----------------
@st.cache_resource(show_spinner=False)
def get_engine():
    return TCMWithLLMMemoryLite(
        agents=["researcher", "analyst", "engineer"],
        topics=["research", "planning", "coding", "ml", "nlp"]
    )

@st.cache_resource(show_spinner=False)
def get_direct():
    return DirectModel()

tcm = get_engine()
direct = get_direct()

# ---------------- Header ----------------
st.markdown('<div class="brandwrap"><h1 class="logotype">Orchestrix</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="tag">Think Better. Orchestrate Smarter.</div>', unsafe_allow_html=True)

# ---------------- Disclaimer ----------------
if "hide_disc" not in st.session_state:
    st.session_state.hide_disc = False
if not st.session_state.hide_disc:
    st.markdown(
        '<div class="disclaimer">🎯 This system learns which agent to trust for each topic over time. '
        'Early answers may route to a less-suited agent. As you interact, the trust distribution adapts '
        '(Thompson sampling over Beta priors) and routing improves.</div>',
        unsafe_allow_html=True
    )
    if st.checkbox("I understand — hide this notice from now on", value=False):
        st.session_state.hide_disc = True

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ---------------- Layout ----------------
left, right = st.columns([7, 5], gap="large")

# Left: question → routing → answers
with left:
    st.markdown('<div class="shell qbox">', unsafe_allow_html=True)

    # Clean, catchy title (no Streamlit default label)
    st.markdown('<div class="sectiontitle">💡 What should we tackle?</div>', unsafe_allow_html=True)

    # Single textarea (label collapsed so nothing extra shows above it)
    q = st.text_area(
        label="",
        placeholder="Describe the outcome you want… e.g., "Plan an MVP rollout for a chatbot."",
        height=120,
        label_visibility="collapsed"
    )

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        load_demo = st.button(
            "📚 Load sample",
            help="Adds a few generic facts (planning, NLP, cosine similarity) so you can see memory retrieval right away."
        )
    with c2:
        run = st.button("🚀 Ask", help="Route to the best expert and answer with supporting memories.")
    with c3:
        compare = st.checkbox("Compare with direct model", value=True,
                               help="Also show an answer straight from the model without routing or memory.")

    st.markdown('</div>', unsafe_allow_html=True)

    if load_demo:
        tcm.mem_shared.add_memory("Transformers weigh token-token interactions via self-attention.", "nlp", "demo", "semantic")
        tcm.mem_shared.add_memory("ML project: data → features → model → eval → iterate.", "planning", "demo", "semantic")
        tcm.mem_shared.add_memory("Cosine similarity is dot(a,b)/(|a||b|).", "coding", "demo", "semantic")
        st.info("✅ Sample knowledge loaded. Ask a question to see memory-augmented routing.")

    # Routing overlay + answer
    if run and q.strip():
        topic_preview = tcm._topic(q.strip())
        seed = int(time.time() * 1e6) & 0xffffffff
        _ = tcm.thompson_draws(topic_preview, seed=seed)  # preview draw for consistency

        # Friendly overlay (no raw numbers by default)
        overlay = st.empty()
        overlay.markdown(f"""
        <div class="overlay">
          <div class="routebox shell">
            <div style="font-weight:700; font-size:18px; margin-bottom:8px; color: #ffffff;">
              🔍 Finding the best expert for <span style="color:#a78bfa">{topic_preview}</span>
            </div>
            <div class="small">Understanding your question → matching topic & context → checking recent reliability → selecting the best fit…</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.0)

        requester = random.choice(tcm.agents)
        np.random.seed(seed)    # match preview selection
        out = tcm.process(q.strip(), requester=requester)
        overlay.empty()
        st.session_state.last = out

        # Optional baseline comparison
        if compare:
            st.session_state.baseline = direct.answer(q.strip())
        else:
            st.session_state.baseline = None

    # Render results
    if "last" in st.session_state:
        out = st.session_state.last
        delegated = bool(out["delegated"])
        cls = "delegation ok" if delegated else "delegation bad"
        txt = "✅ Delegated to trusted expert" if delegated else "⚡ Handled by initial agent"

        # Routing card (plain-English)
        st.markdown('<div class="shell" style="margin-top:14px">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="status">
          <span class="badge {cls}">{txt}</span>
          <span class="badge badge-topic">📂 Topic: {out['topic']}</span>
          <span class="badge badge-mem">🧠 Memories: {out['memories_used']}</span>
          <span class="badge">👤 Expert: {out['expert']}</span>
          <span class="badge">📊 Trust: {out['trust_score']:.3f}</span>
        </div>
        <div class="hint" style="margin-top:8px;">
          Why this differs from a normal chat: Orchestrix routes by learned trust per topic and
          pulls prior knowledge from memory before answering.
        </div>
        """, unsafe_allow_html=True)

        # Answer(s)
        if st.session_state.get("baseline") and compare:
            colA, colB = st.columns(2)
        else:
            colA = st.columns(1)[0]; colB = None

        with colA:
            st.subheader("🎯 Orchestrix answer")
            st.markdown('<div class="answer">', unsafe_allow_html=True)
            st.write(out["response"])
            st.markdown('</div>', unsafe_allow_html=True)

            # Memory reveal
            with st.expander("📝 Show the memory snippets used"):
                if out["memories_used"] == 0:
                    st.caption("No stored memories were used for this answer yet.")
                else:
                    for i, s in enumerate(out["mem_snippets"], 1):
                        st.markdown(f"**{i}.** {s}")

        if colB:
            with colB:
                st.subheader("💬 Direct model (baseline)")
                st.markdown('<div class="answer">', unsafe_allow_html=True)
                st.write(st.session_state.baseline)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("This shows what you'd typically get without Orchestrix's expert routing and memory context.")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='hint'>Ask something to see routing and answers here.</div>", unsafe_allow_html=True)

# Right: metrics + agents
with right:
    s = tcm.metrics if hasattr(tcm, "metrics") else {"total":0,"delegations":0,"mems_used":[],"hit_rate":[],"consolidations":0}
    tot = max(1, s.get("total",0))
    summary = {
        "total_queries": s.get("total",0),
        "delegation_rate": s.get("delegations",0)/tot,
        "avg_memories_used": float(np.mean(s.get("mems_used",[]))) if s.get("mems_used") else 0.0,
        "avg_memory_hit_rate": float(np.mean(s.get("hit_rate",[]))) if s.get("hit_rate") else 0.0,
        "total_consolidations": s.get("consolidations",0),
    }

    st.markdown('<div class="grid">', unsafe_allow_html=True)
    def chip(label, value, help_text=""):
        st.markdown(f"""
          <div class="chip" title="{help_text}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
          </div>
        """, unsafe_allow_html=True)

    chip("Total queries", summary["total_queries"], "How many questions you asked in this session.")
    chip("Delegation rate", f"{summary['delegation_rate']*100:.1f}%", "Share of queries routed away from the initial agent.")
    chip("Avg memories", f"{summary['avg_memories_used']:.2f}", "Average number of snippets pulled from memory per answer.")
    chip("Memory hit rate", f"{summary['avg_memory_hit_rate']:.2f}", "Fraction of used memories that came from the local expert vs shared pool.")
    chip("Consolidations", summary["total_consolidations"], "How many episodic memories hardened into semantic knowledge.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Agents trust
    st.markdown('<div class="shell soft">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #ffffff; margin-bottom: 16px;">🤖 Agents</h3>', unsafe_allow_html=True)
    trust = tcm.trust if hasattr(tcm, "trust") else {}
    agents = ["researcher","analyst","engineer"]
    colA, colB, colC = st.columns(3)
    for col, a in zip([colA,colB,colC], agents):
        vals = []
        for k,v in trust.items():
            if k.startswith(a + ":"):
                alpha, beta = v["alpha"], v["beta"]
                vals.append(alpha/(alpha+beta))
        avg = sum(vals)/len(vals) if vals else 0.5
        with col:
            st.markdown(f"""
            <div class="agent" title="Average trust across topics for {a.title()} (α/(α+β)).">
              <div class="name">{a.title()}</div>
              <div class="bar"><span style="width:{avg*100:.0f}%"></span></div>
              <div class="hint">Trust: {avg:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# How it works (short)
with st.expander("⚙️ What makes Orchestrix different?"):
    st.markdown("""
### 🎯 **Expert Routing**
Each agent builds **topic-specific trust** over time. We sample from these trust distributions and route to the highest draw (Thompson sampling).

### 🧠 **Memory-Augmented Answers**
Relevant snippets from prior interactions are retrieved and supplied to the model before answering.

### 📈 **Learning Loop**
A lightweight quality check updates trust (success increases α; otherwise β), so routing improves the more you use it.

### 🔬 **See the Difference**
Use **"Compare with direct model"** to see the difference vs a plain model call with no routing or memory.
    """)
