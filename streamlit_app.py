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
    page_title="Orchestrix // TCM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Apple-Style CSS (Glassmorphism & Bento) ----------------
st.markdown("""
<style>
    /* Global Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-color: #000000;
        --card-bg: rgba(28, 28, 30, 0.65);
        --card-border: rgba(255, 255, 255, 0.1);
        --text-primary: #F5F5F7;
        --text-secondary: #86868B;
        --accent-blue: #2997FF;
        --accent-green: #30D158;
        --accent-orange: #FF9F0A;
    }

    .stApp {
        background-color: var(--bg-color);
        font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif;
    }

    /* Titles */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }
    
    p, label, .stMarkdown {
        color: var(--text-secondary) !important;
    }

    /* Glassmorphism Cards */
    div.css-1r6slb0, div.stExpander, div.stMetric {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }

    /* Custom Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.02);
        background: rgba(255,255,255,0.08);
    }
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
    }
    .metric-lbl {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* Input Fields */
    .stTextArea textarea {
        background-color: #1C1C1E !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(41, 151, 255, 0.3) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 30px;
        font-weight: 500;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    /* Primary Action Button */
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background: var(--accent-blue);
        color: white;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"]:hover {
        box-shadow: 0 0 15px rgba(41, 151, 255, 0.6);
    }

    /* Agent Trust Bars */
    .trust-wrapper {
        margin-bottom: 12px;
    }
    .trust-header {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 6px;
        color: var(--text-primary);
    }
    .progress-track {
        background: #333;
        height: 6px;
        border-radius: 3px;
        width: 100%;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0c0c;
        border-right: 1px solid #222;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Logic Core (Preserved & Robust) ----------------

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

class LLMCoreMemory:
    def __init__(self, client: OpenAI, embed_model: str = "text-embedding-3-small"):
        self.client = client
        self.embed_model = embed_model
        self.working_memory = deque(maxlen=10)
        self.episodic: List[MemoryEntry] = []
        self.semantic: Dict[str, List[MemoryEntry]] = {}
        self._embed_cache: Dict[str, np.ndarray] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def _embed(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._embed_cache: return self._embed_cache[key]
        resp = self.client.embeddings.create(model=self.embed_model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        self._embed_cache[key] = vec
        return vec

    def add_memory(self, content: str, topic: str, agent_id: str, memory_type: str = "episodic") -> str:
        emb = self._embed(content)
        mem_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:10]
        entry = MemoryEntry(mem_id, content, emb, topic, agent_id, time.time(), 0, memory_type)
        if memory_type == "episodic":
            self.episodic.append(entry)
            self.working_memory.append(mem_id)
        elif memory_type == "semantic":
            self.semantic.setdefault(topic, []).append(entry)
        return mem_id

    def retrieve(self, query: str, k: int = 5) -> List[MemoryEntry]:
        entries = list(self.episodic)
        for arr in self.semantic.values(): entries.extend(arr)
        if not entries: return []
        q = self._embed(query)
        mats = np.stack([e.embedding for e in entries], axis=0)
        sims = mats @ q / (np.linalg.norm(mats, axis=1) * np.linalg.norm(q) + 1e-9)
        idxs = np.argsort(-sims)[:k]
        return [entries[i] for i in idxs]

    def consolidate(self) -> int:
        moved = 0
        keep = []
        for e in self.episodic:
            if e.access_count >= 3: # Simplified threshold
                self.semantic.setdefault(e.topic, []).append(e)
                moved += 1
            else:
                keep.append(e)
        self.episodic = keep
        return moved

class TCMSystem:
    def __init__(self, agents: List[str], client: OpenAI):
        self.client = client
        self.agents = agents
        # Trust Matrix: { "agent:topic": {alpha: 1, beta: 1} }
        self.trust = defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})
        self.mem_local = {a: LLMCoreMemory(client) for a in agents}
        self.mem_shared = LLMCoreMemory(client)
        self.metrics = {"delegations": 0, "total": 0, "mems_used": [], "consolidations": 0}

    def _topic_classifier(self, text: str) -> str:
        # Simple heuristic classifier
        keywords = {
            "coding": ["code", "python", "bug", "error", "function"],
            "research": ["history", "what is", "explain", "study"],
            "planning": ["roadmap", "strategy", "steps", "how to"]
        }
        text = text.lower()
        for topic, words in keywords.items():
            if any(w in text for w in words): return topic
        return "general"

    def get_expert(self, topic: str) -> str:
        # Thompson Sampling
        draws = {}
        for agent in self.agents:
            params = self.trust[f"{agent}:{topic}"]
            draws[agent] = np.random.beta(params["alpha"], params["beta"])
        return max(draws, key=draws.get)

    def process(self, query: str) -> Dict:
        self.metrics["total"] += 1
        topic = self._topic_classifier(query)
        requester = random.choice(self.agents) # Simulation: random agent receives request
        expert = self.get_expert(topic)
        
        delegated = (requester != expert)
        if delegated: self.metrics["delegations"] += 1

        # Retrieval
        mems = self.mem_local[expert].retrieve(query, k=2) + self.mem_shared.retrieve(query, k=2)
        context = "\n".join([f"- {m.content}" for m in mems]) if mems else "No prior memory."

        # Generation
        prompt = f"Role: {expert}. Topic: {topic}.\nContext: {context}\nQuery: {query}\nAnswer concisely."
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
            answer = resp.choices[0].message.content
        except:
            answer = "I'm having trouble connecting to my brain (API Error)."

        # Feedback Loop (Simulated)
        success = len(answer) > 20 # Simple proxy for quality
        self.trust[f"{expert}:{topic}"]["alpha" if success else "beta"] += 1
        
        # Save Memory
        self.mem_local[expert].add_memory(f"Q: {query} A: {answer}", topic, expert)
        self.metrics["mems_used"].append(len(mems))
        
        return {
            "response": answer, "expert": expert, "topic": topic, 
            "delegated": delegated, "mems": len(mems),
            "trust": self.trust[f"{expert}:{topic}"]["alpha"] / (self.trust[f"{expert}:{topic}"]["alpha"] + self.trust[f"{expert}:{topic}"]["beta"])
        }

# ---------------- Initialization & Sidebar ----------------

with st.sidebar:
    st.title("⚙️ System Config")
    
    # Secure API Key Handling
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your SK key to activate agents.")
    if not api_key and "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
    
    st.divider()
    
    # Reset Button
    if st.button("Reset Memory System"):
        st.session_state.tcm = None
        st.experimental_rerun()
        
    st.info("System Status: " + ("🟢 Online" if api_key else "🔴 Offline"))
    st.markdown("---")
    st.markdown("**Agents Active:**\n- Planner\n- Researcher\n- Coder")

# Initialize System
if api_key:
    if "tcm" not in st.session_state or st.session_state.tcm is None:
        client = OpenAI(api_key=api_key)
        st.session_state.tcm = TCMSystem(["planner", "researcher", "coder"], client)
        # Seed some data
        st.session_state.tcm.mem_shared.add_memory("Project Orchestrix uses a multi-agent star topology.", "research", "system", "semantic")
else:
    st.warning("Please provide an OpenAI API Key in the sidebar to initialize the Neural Network.")
    st.stop()

tcm = st.session_state.tcm

# ---------------- Main UI ----------------

# Hero Header
st.markdown("""
<div style="text-align: center; margin-bottom: 40px; animation: fadeIn 1s;">
    <h1 style="font-size: 60px; margin-bottom: 0;">Orchestrix</h1>
    <p style="font-size: 20px; color: #86868B;">Transactive Cognitive Memory System</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([7, 5], gap="large")

with col1:
    st.markdown("### 💬 Input Interface")
    user_query = st.text_area("Task input", height=100, placeholder="E.g., Research the history of transformers or Write a Python script for merge sort...", label_visibility="collapsed")
    
    c1, c2 = st.columns([1, 4])
    with c1:
        process_btn = st.button("Process", type="primary", use_container_width=True)
    
    if process_btn and user_query:
        with st.spinner("Agents negotiating delegation..."):
            result = tcm.process(user_query)
            st.session_state.last_result = result
            time.sleep(0.5) # UX pause for effect

    # Result Display
    if "last_result" in st.session_state:
        res = st.session_state.last_result
        
        # Meta-Data Badge
        st.markdown(f"""
        <div style="display: flex; gap: 10px; margin-top: 20px; margin-bottom: 10px;">
            <span style="background: rgba(41, 151, 255, 0.2); color: #2997FF; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;">{res['expert'].upper()}</span>
            <span style="background: rgba(48, 209, 88, 0.2); color: #30D158; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600;">{res['topic'].upper()}</span>
            <span style="background: rgba(255, 255, 255, 0.1); color: #ccc; padding: 4px 12px; border-radius: 12px; font-size: 12px;">Trust: {res['trust']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # The Response Card
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border-radius: 16px; padding: 24px; border: 1px solid rgba(255,255,255,0.1); line-height: 1.6;">
            {res['response']}
        </div>
        """, unsafe_allow_html=True)
        
        if res['delegated']:
            st.caption("⚡ Task was dynamically delegated to the domain expert.")

with col2:
    st.markdown("### 📊 System Vitality")
    
    # Bento Grid for Metrics
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{tcm.metrics['total']}</div>
            <div class="metric-lbl">Total Queries</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        del_rate = (tcm.metrics['delegations'] / max(1, tcm.metrics['total'])) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{del_rate:.0f}%</div>
            <div class="metric-lbl">Delegation Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Trust Matrix Visualization
    st.markdown("#### 🧠 Trust Matrix")
    st.markdown("<div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    for agent in ["planner", "researcher", "coder"]:
        # Calculate average trust for this agent across topics
        t_vals = [params["alpha"]/(params["alpha"]+params["beta"]) for k, params in tcm.trust.items() if k.startswith(agent)]
        avg_t = np.mean(t_vals) if t_vals else 0.5
        
        # Color logic
        bar_color = "#2997FF" # Blue
        if avg_t > 0.8: bar_color = "#30D158" # Green
        elif avg_t < 0.4: bar_color = "#FF453A" # Red
        
        st.markdown(f"""
        <div class="trust-wrapper">
            <div class="trust-header">
                <span>{agent.capitalize()}</span>
                <span>{avg_t:.2f}</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width: {avg_t*100}%; background: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("Show Memory Logs"):
        if st.session_state.last_result:
            st.json(tcm.metrics)
        else:
            st.write("No data yet.")
