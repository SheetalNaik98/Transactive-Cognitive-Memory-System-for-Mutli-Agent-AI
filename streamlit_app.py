#!/usr/bin/env python3
"""
Orchestrix — TCM + LLM Core Memory
"""

import os
import time
import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict, deque

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import streamlit as st

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Orchestrix", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Apple-Inspired CSS ----------------
def inject_css():
    st.markdown("""
    <style>
      /* SF Pro Display Font Stack */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
      
      * {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }
      
      /* Dark Mode Color Palette - Apple Style */
      :root {
        --bg-primary: #000000;
        --bg-secondary: #1c1c1e;
        --bg-tertiary: #2c2c2e;
        --bg-elevated: #3a3a3c;
        
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.86);
        --text-tertiary: rgba(255, 255, 255, 0.60);
        --text-quaternary: rgba(255, 255, 255, 0.38);
        
        --separator: rgba(255, 255, 255, 0.12);
        --separator-opaque: #38383a;
        
        --accent: #0a84ff;
        --accent-hover: #409cff;
        --success: #32d74b;
        --warning: #ff9f0a;
        --danger: #ff453a;
        
        --card-bg: rgba(255, 255, 255, 0.04);
        --card-hover: rgba(255, 255, 255, 0.08);
      }
      
      /* Main App Background */
      .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
      }
      
      /* Hide Default Streamlit Elements */
      #MainMenu, footer, header {visibility: hidden;}
      
      /* Logo Animation */
      @keyframes fadeInDown {
        from {
          opacity: 0;
          transform: translateY(-20px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      
      /* Header */
      .header {
        text-align: center;
        padding: 60px 0 40px;
        animation: fadeInDown 0.8s ease;
      }
      
      .logo {
        font-size: 56px;
        font-weight: 700;
        letter-spacing: -2px;
        background: linear-gradient(180deg, #ffffff 0%, #86868b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
      }
      
      .tagline {
        font-size: 21px;
        font-weight: 400;
        color: var(--text-tertiary);
        letter-spacing: -0.5px;
      }
      
      /* Card Animations */
      @keyframes slideUp {
        from {
          opacity: 0;
          transform: translateY(30px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      
      /* Cards */
      .card {
        background: var(--bg-secondary);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        border: 1px solid var(--separator);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideUp 0.6s ease backwards;
      }
      
      .card:hover {
        background: var(--bg-tertiary);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
      }
      
      /* Input Field - CRITICAL FIX for visibility */
      .stTextArea textarea {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--separator) !important;
        color: var(--text-primary) !important;
        font-size: 17px !important;
        line-height: 1.5 !important;
        padding: 16px !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
        -webkit-appearance: none !important;
      }
      
      .stTextArea textarea:focus {
        background: var(--bg-elevated) !important;
        border-color: var(--accent) !important;
        outline: none !important;
        box-shadow: 0 0 0 4px rgba(10, 132, 255, 0.1) !important;
      }
      
      .stTextArea textarea::placeholder {
        color: var(--text-quaternary) !important;
      }
      
      /* Button Styles */
      .stButton > button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 17px;
        font-weight: 500;
        letter-spacing: -0.5px;
        transition: all 0.2s ease;
        cursor: pointer;
      }
      
      .stButton > button:hover {
        background: var(--accent-hover);
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(10, 132, 255, 0.3);
      }
      
      .stButton > button:active {
        transform: scale(0.98);
      }
      
      /* Status Pills */
      .pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
        letter-spacing: -0.3px;
        animation: fadeIn 0.3s ease;
      }
      
      @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
      }
      
      .pill-success {
        background: rgba(50, 215, 75, 0.2);
        color: var(--success);
      }
      
      .pill-warning {
        background: rgba(255, 159, 10, 0.2);
        color: var(--warning);
      }
      
      .pill-info {
        background: rgba(10, 132, 255, 0.2);
        color: var(--accent);
      }
      
      /* Metrics */
      .metric {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        border: 1px solid var(--separator);
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease backwards;
      }
      
      .metric:hover {
        background: var(--card-hover);
        transform: translateX(4px);
      }
      
      .metric-label {
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-tertiary);
        margin-bottom: 8px;
      }
      
      .metric-value {
        font-size: 34px;
        font-weight: 600;
        letter-spacing: -1px;
        color: var(--text-primary);
      }
      
      /* Trust Bar */
      .trust-container {
        margin-bottom: 20px;
      }
      
      .trust-label {
        font-size: 15px;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 8px;
      }
      
      .trust-bar {
        height: 6px;
        background: var(--separator-opaque);
        border-radius: 3px;
        overflow: hidden;
        position: relative;
      }
      
      .trust-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 3px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
      }
      
      .trust-value {
        font-size: 13px;
        color: var(--text-tertiary);
        margin-top: 4px;
      }
      
      /* Answer Box */
      .answer {
        background: var(--bg-tertiary);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        color: var(--text-primary);
        font-size: 17px;
        line-height: 1.6;
        letter-spacing: -0.3px;
        animation: slideUp 0.4s ease;
      }
      
      /* Loading State */
      @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
      }
      
      .loading {
        animation: pulse 1.5s ease infinite;
      }
      
      /* Text Colors Fix */
      h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.5px;
      }
      
      p, span, div, label {
        color: var(--text-secondary) !important;
      }
      
      /* Checkbox */
      .stCheckbox label {
        color: var(--text-secondary) !important;
        font-size: 15px;
      }
      
      /* Expander */
      .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border-radius: 12px;
        font-weight: 500;
      }
      
      /* Info Box */
      .info-box {
        background: var(--bg-secondary);
        border-left: 4px solid var(--accent);
        padding: 16px;
        margin: 20px 0;
        border-radius: 8px;
        color: var(--text-secondary);
        font-size: 15px;
        line-height: 1.5;
      }
      
      /* Smooth Scrolling */
      html {
        scroll-behavior: smooth;
      }
      
      /* Selection Color */
      ::selection {
        background: var(--accent);
        color: white;
      }
      
      /* Sidebar if needed */
      section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--separator);
      }
      
      /* Column Gap */
      .row-widget.stHorizontal {
        gap: 32px;
      }
      
      /* Animation Delays */
      .card:nth-child(1) { animation-delay: 0.1s; }
      .card:nth-child(2) { animation-delay: 0.2s; }
      .card:nth-child(3) { animation-delay: 0.3s; }
      .metric:nth-child(1) { animation-delay: 0.1s; }
      .metric:nth-child(2) { animation-delay: 0.15s; }
      .metric:nth-child(3) { animation-delay: 0.2s; }
      .metric:nth-child(4) { animation-delay: 0.25s; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------------- API Key Management ----------------
def init_openai():
    """Initialize OpenAI API key from Streamlit secrets or environment"""
    api_key = None
    
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets['OPENAI_API_KEY']
    elif 'OPENAI_API_KEY' in os.environ:
        api_key = os.environ['OPENAI_API_KEY']
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        return True
    
    with st.sidebar:
        st.markdown("### OpenAI API Key")
        user_key = st.text_input(
            "Enter your API key:",
            type="password",
            placeholder="sk-...",
            help="Required for AI agent functionality"
        )
        if user_key:
            os.environ['OPENAI_API_KEY'] = user_key
            st.success("API key configured")
            return True
        else:
            st.error("Please enter your OpenAI API key")
            st.stop()
    return False

init_openai()

# ---------------- Core Classes (Same as before) ----------------
@dataclass
class MemoryEntry:
    id: str
    content: str
    embedding: np.ndarray
    topic: str
    agent_id: str
    timestamp: float
    access_count: int = 0
    memory_type: str = "episodic"
    metadata: Dict = field(default_factory=dict)

class LLMCoreMemory:
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

    def add_memory(self, content: str, topic: str, agent_id: str, memory_type: str = "episodic") -> str:
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
        entries = list(self.episodic)
        for arr in self.semantic.values():
            entries.extend(arr)
        entries.extend(self.procedural.values())
        return entries

    def retrieve(self, query: str, k: int = 5) -> List[MemoryEntry]:
        entries = self._all_entries()
        if not entries:
            return []
        q = self._embed(query)
        mats = np.stack([e.embedding for e in entries], axis=0)
        sims = mats @ q / (np.linalg.norm(mats, axis=1) * np.linalg.norm(q) + 1e-9)
        idxs = np.argsort(-sims)[:k]
        return [entries[i] for i in idxs]

    def consolidate(self) -> int:
        moved = 0
        keep = []
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
                        memory_type="semantic"
                    )
                )
                moved += 1
            else:
                keep.append(e)
        self.episodic = keep
        return moved

class TCMWithLLMMemory:
    def __init__(self, agents: List[str], topics: List[str], chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.chat_model = chat_model
        self.agents = agents
        self.topics = topics
        self.trust = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})
        self.mem_local = {a: LLMCoreMemory(self.client) for a in agents}
        self.mem_shared = LLMCoreMemory(self.client)
        self.metrics = {
            "delegations": 0,
            "total": 0,
            "mems_used": [],
            "hit_rate": [],
            "consolidations": 0
        }

    def _topic(self, text: str) -> str:
        text_lower = text.lower()
        rules = {
            "planning": ["plan", "roadmap", "strategy", "schedule"],
            "research": ["research", "investigate", "study", "analyze"],
            "coding": ["code", "implement", "bug", "debug", "python"],
            "ml": ["ml", "model", "train", "neural", "classifier"],
            "nlp": ["nlp", "transformer", "llm", "token", "text"]
        }
        for topic, keywords in rules.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        return self.topics[0] if self.topics else "general"

    def thompson_draws(self, topic: str, seed: Optional[int] = None) -> Dict[str, float]:
        if seed is not None:
            np.random.seed(seed)
        draws = {}
        for agent in self.agents:
            params = self.trust[f"{agent}:{topic}"]
            draws[agent] = float(np.random.beta(params["alpha"], params["beta"]))
        return draws

    def _expert(self, topic: str) -> str:
        scores = self.thompson_draws(topic)
        return max(scores, key=scores.get)

    def _format_memories(self, mems: List[MemoryEntry]) -> str:
        if not mems:
            return "No relevant memories."
        return "\n".join([f"{i}. ({m.memory_type}) {m.content[:200]}..." 
                         for i, m in enumerate(mems[:5], 1)])

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def _quality_score(self, response: str, memories: List[MemoryEntry]) -> float:
        score = 0.5
        if len(response) > 120:
            score += 0.2
        if memories and any(m.content[:60] in response for m in memories):
            score += 0.3
        return min(1.0, score)

    def _update_trust(self, agent: str, topic: str, success: bool):
        key = f"{agent}:{topic}"
        if success:
            self.trust[key]["alpha"] += 1
        else:
            self.trust[key]["beta"] += 1

    def trust_score(self, agent: str, topic: str) -> float:
        params = self.trust[f"{agent}:{topic}"]
        return params["alpha"] / (params["alpha"] + params["beta"])

    def process(self, query: str, requester: Optional[str] = None) -> Dict:
        self.metrics["total"] += 1
        topic = self._topic(query)
        requester = requester or random.choice(self.agents)
        expert = self._expert(topic)
        delegated = (expert != requester)
        
        if delegated:
            self.metrics["delegations"] += 1

        local_mems = self.mem_local[expert].retrieve(query, k=3)
        shared_mems = self.mem_shared.retrieve(query, k=2)
        all_memories = local_mems + shared_mems

        prompt = f"""You are {expert}, an expert in {topic}.
        
Relevant memories:
{self._format_memories(all_memories)}

User query: {query}

Provide a helpful, accurate answer using the memories when relevant."""

        answer = self._call_llm(prompt)

        self.mem_local[expert].add_memory(
            content=f"Q: {query}\nA: {answer}",
            topic=topic,
            agent_id=expert,
            memory_type="episodic"
        )

        quality = self._quality_score(answer, all_memories)
        self._update_trust(expert, topic, success=(quality > 0.7))

        self.metrics["consolidations"] += self.mem_local[expert].consolidate()
        self.metrics["consolidations"] += self.mem_shared.consolidate()

        self.metrics["mems_used"].append(len(all_memories))
        hit_rate = (len(local_mems) / max(1, len(all_memories))) if all_memories else 0.0
        self.metrics["hit_rate"].append(hit_rate)

        return {
            "query": query,
            "response": answer,
            "topic": topic,
            "requester": requester,
            "expert": expert,
            "delegated": delegated,
            "memories_used": len(all_memories),
            "trust_score": self.trust_score(expert, topic),
            "mem_snippets": [m.content for m in all_memories]
        }

class DirectModel:
    def __init__(self, chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.chat_model = chat_model

    def answer(self, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": query}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# ---------------- Initialize Models ----------------
@st.cache_resource(show_spinner=False)
def get_tcm_engine():
    return TCMWithLLMMemory(
        agents=["researcher", "analyst", "engineer"],
        topics=["research", "planning", "coding", "ml", "nlp"]
    )

@st.cache_resource(show_spinner=False)
def get_direct_model():
    return DirectModel()

tcm = get_tcm_engine()
direct = get_direct_model()

# ---------------- UI ----------------
# Header
st.markdown("""
<div class="header">
    <h1 class="logo">Orchestrix</h1>
    <p class="tagline">Intelligent Agent Orchestration</p>
</div>
""", unsafe_allow_html=True)

# Main Layout
col_left, col_right = st.columns([7, 5], gap="large")

# Left Column - Query Interface
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown("### What can I help with?")
    
    query = st.text_area(
        label="Query",
        placeholder="Plan an MVP rollout for a chatbot application...",
        height=120,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        load_sample = st.button("Load Sample", help="Load sample knowledge")
    with col2:
        run_query = st.button("Process", help="Process your query", type="primary")
    with col3:
        compare_baseline = st.checkbox("Compare with baseline", value=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if load_sample:
        tcm.mem_shared.add_memory(
            "Transformers use self-attention mechanisms to weigh token relationships.",
            "nlp", "demo", "semantic"
        )
        tcm.mem_shared.add_memory(
            "ML pipeline: data collection → preprocessing → model training → evaluation → deployment.",
            "planning", "demo", "semantic"
        )
        tcm.mem_shared.add_memory(
            "Cosine similarity formula: cos(θ) = (A·B)/(||A||×||B||)",
            "coding", "demo", "semantic"
        )
        st.success("Sample knowledge loaded")
    
    if run_query and query.strip():
        with st.spinner("Processing..."):
            result = tcm.process(query.strip())
            st.session_state.last_result = result
            
            if compare_baseline:
                baseline_answer = direct.answer(query.strip())
                st.session_state.baseline_answer = baseline_answer
            else:
                st.session_state.baseline_answer = None
    
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        delegated = result["delegated"]
        status_class = "pill-success" if delegated else "pill-warning"
        status_text = "Delegated" if delegated else "Direct"
        
        st.markdown(f"""
        <span class="{status_class} pill">{status_text}</span>
        <span class="pill pill-info">{result['topic'].upper()}</span>
        <span class="pill pill-info">{result['expert'].upper()}</span>
        <span class="pill pill-info">{result['memories_used']} memories</span>
        <span class="pill pill-info">Trust: {result['trust_score']:.2f}</span>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if compare_baseline and "baseline_answer" in st.session_state:
            ans_col1, ans_col2 = st.columns(2)
            
            with ans_col1:
                st.markdown("#### Orchestrix")
                st.markdown(f'<div class="answer">{result["response"]}</div>', 
                          unsafe_allow_html=True)
                
                with st.expander("View memories"):
                    if result["memories_used"] > 0:
                        for i, snippet in enumerate(result["mem_snippets"], 1):
                            st.write(f"{i}. {snippet}")
                    else:
                        st.write("No memories used")
            
            with ans_col2:
                st.markdown("#### Baseline")
                st.markdown(f'<div class="answer">{st.session_state.baseline_answer}</div>', 
                          unsafe_allow_html=True)
        else:
            st.markdown("#### Response")
            st.markdown(f'<div class="answer">{result["response"]}</div>', 
                      unsafe_allow_html=True)
            
            with st.expander("View memories"):
                if result["memories_used"] > 0:
                    for i, snippet in enumerate(result["mem_snippets"], 1):
                        st.write(f"{i}. {snippet}")
                else:
                    st.write("No memories used")

# Right Column - Metrics
with col_right:
    metrics = tcm.metrics
    total = max(1, metrics["total"])
    
    st.markdown("### Metrics")
    
    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">Queries</div>
        <div class="metric-value">{metrics['total']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    delegation_rate = (metrics['delegations'] / total) * 100
    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">Delegation Rate</div>
        <div class="metric-value">{delegation_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    avg_memories = np.mean(metrics['mems_used']) if metrics['mems_used'] else 0
    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">Avg Memories</div>
        <div class="metric-value">{avg_memories:.1f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">Consolidations</div>
        <div class="metric-value">{metrics['consolidations']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Agent Trust")
    
    for agent in ["researcher", "analyst", "engineer"]:
        trust_values = []
        for key, params in tcm.trust.items():
            if key.startswith(f"{agent}:"):
                trust = params["alpha"] / (params["alpha"] + params["beta"])
                trust_values.append(trust)
        
        avg_trust = np.mean(trust_values) if trust_values else 0.5
        
        st.markdown(f"""
        <div class="trust-container">
            <div class="trust-label">{agent.capitalize()}</div>
            <div class="trust-bar">
                <div class="trust-fill" style="width: {avg_trust*100}%"></div>
            </div>
            <div class="trust-value">{avg_trust:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

# Info
with st.expander("How it works"):
    st.markdown("""
    <div class="info-box">
    <strong>Thompson Sampling:</strong> Agents build topic-specific trust through Beta-Bernoulli priors.<br><br>
    <strong>Memory System:</strong> Episodic memories consolidate into semantic knowledge over time.<br><br>
    <strong>Delegation:</strong> Routes queries to the most trusted expert for each topic.
    </div>
    """, unsafe_allow_html=True)
