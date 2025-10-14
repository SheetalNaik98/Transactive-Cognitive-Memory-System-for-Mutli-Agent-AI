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
    page_title="Orchestrix - AI Agent Orchestration", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CSS Styling ----------------
def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
      
      /* Global Reset and Font */
      * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }
      
      /* Main App Container */
      .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1f 100%);
        color: #ffffff;
      }
      
      /* Hide Streamlit Branding */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      
      /* Custom Variables */
      :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #141418;
        --bg-card: rgba(255, 255, 255, 0.05);
        --border: rgba(255, 255, 255, 0.1);
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --text-muted: #707070;
        --accent-purple: #8b5cf6;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
      }
      
      /* Logo and Header */
      .brand-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
      }
      
      .logo {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
      }
      
      .tagline {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 500;
      }
      
      /* Card Component */
      .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
      }
      
      .card:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
      }
      
      /* Disclaimer Box */
      .disclaimer {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        color: var(--text-primary);
        font-size: 0.9rem;
        line-height: 1.6;
      }
      
      /* Metrics */
      .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
      }
      
      .metric-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
      }
      
      .metric-value {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
      }
      
      /* Status Badges */
      .badge {
        display: inline-block;
        padding: 0.375rem 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
      }
      
      .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
      }
      
      .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
      }
      
      .badge-info {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
      }
      
      .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
      }
      
      /* Agent Cards */
      .agent-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
      }
      
      .agent-name {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 0.5rem;
      }
      
      .trust-bar {
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        overflow: hidden;
        margin: 0.5rem 0;
      }
      
      .trust-fill {
        height: 100%;
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
        transition: width 0.5s ease;
      }
      
      /* Answer Box */
      .answer-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        color: var(--text-primary);
        line-height: 1.6;
        margin-top: 1rem;
      }
      
      /* Input Area */
      .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        font-size: 14px !important;
      }
      
      .stTextArea textarea:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
      }
      
      /* Buttons */
      .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
      }
      
      .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(139, 92, 246, 0.3);
      }
      
      /* Fix Text Colors */
      h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
      }
      
      p, span, div, label {
        color: var(--text-secondary) !important;
      }
      
      .st-emotion-cache-16idsys p {
        color: var(--text-primary) !important;
      }
      
      /* Expander */
      .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
      }
      
      /* Checkbox */
      .stCheckbox label {
        color: var(--text-secondary) !important;
      }
      
      /* Loading Overlay */
      .overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(5px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
      }
      
      .overlay-content {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        max-width: 400px;
      }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------------- API Key Management ----------------
def init_openai():
    """Initialize OpenAI API key from Streamlit secrets or environment"""
    api_key = None
    
    # Try Streamlit secrets first
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets['OPENAI_API_KEY']
    # Then environment variable
    elif 'OPENAI_API_KEY' in os.environ:
        api_key = os.environ['OPENAI_API_KEY']
    
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        return True
    
    # Show sidebar input if no key found
    with st.sidebar:
        st.markdown("### 🔑 OpenAI API Key Required")
        user_key = st.text_input(
            "Enter your OpenAI API key:",
            type="password",
            placeholder="sk-...",
            help="Your API key is needed to power the AI agents"
        )
        if user_key:
            os.environ['OPENAI_API_KEY'] = user_key
            st.success("✅ API key saved for this session")
            return True
        else:
            st.error("⚠️ Please enter your OpenAI API key to continue")
            st.stop()
    return False

# Initialize API
init_openai()

# ---------------- Core Classes ----------------
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

        # Retrieve memories
        local_mems = self.mem_local[expert].retrieve(query, k=3)
        shared_mems = self.mem_shared.retrieve(query, k=2)
        all_memories = local_mems + shared_mems

        # Generate response
        prompt = f"""You are {expert}, an expert in {topic}.
        
Relevant memories:
{self._format_memories(all_memories)}

User query: {query}

Provide a helpful, accurate answer using the memories when relevant."""

        answer = self._call_llm(prompt)

        # Store new memory
        self.mem_local[expert].add_memory(
            content=f"Q: {query}\nA: {answer}",
            topic=topic,
            agent_id=expert,
            memory_type="episodic"
        )

        # Update trust based on quality
        quality = self._quality_score(answer, all_memories)
        self._update_trust(expert, topic, success=(quality > 0.7))

        # Consolidate memories
        self.metrics["consolidations"] += self.mem_local[expert].consolidate()
        self.metrics["consolidations"] += self.mem_shared.consolidate()

        # Update metrics
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
    """Baseline model without routing or memory"""
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

# ---------------- UI Components ----------------
# Header
st.markdown("""
<div class="brand-container">
    <h1 class="logo">Orchestrix</h1>
    <p class="tagline">Think Better. Orchestrate Smarter.</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
if "hide_disclaimer" not in st.session_state:
    st.session_state.hide_disclaimer = False

if not st.session_state.hide_disclaimer:
    st.markdown("""
    <div class="disclaimer">
        <strong>How it works:</strong> This system learns which agent to trust for each topic over time. 
        Early answers may route to less-optimal agents, but the system improves through Thompson Sampling 
        with Beta-Bernoulli trust models.
    </div>
    """, unsafe_allow_html=True)
    
    if st.checkbox("Hide this notice"):
        st.session_state.hide_disclaimer = True

# Main Layout
col_left, col_right = st.columns([7, 5], gap="large")

# Left Column - Query Interface
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown("### 💬 What can I help you with?")
    
    query = st.text_area(
        label="Query",
        placeholder="Example: Plan an MVP rollout for a chatbot application",
        height=120,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        load_sample = st.button("📚 Load Sample", help="Load sample knowledge into memory")
    with col2:
        run_query = st.button("🚀 Ask", help="Process your query", type="primary")
    with col3:
        compare_baseline = st.checkbox("Compare with baseline", value=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load sample memories
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
        st.success("✅ Sample knowledge loaded successfully!")
    
    # Process query
    if run_query and query.strip():
        with st.spinner("Processing..."):
            # Process with TCM
            result = tcm.process(query.strip())
            st.session_state.last_result = result
            
            # Process with baseline if requested
            if compare_baseline:
                baseline_answer = direct.answer(query.strip())
                st.session_state.baseline_answer = baseline_answer
            else:
                st.session_state.baseline_answer = None
    
    # Display results
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        
        # Status badges
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        delegated = result["delegated"]
        badge_class = "badge-success" if delegated else "badge-warning"
        badge_text = "✅ Delegated to Expert" if delegated else "⚡ Direct Processing"
        
        st.markdown(f"""
        <span class="{badge_class} badge">{badge_text}</span>
        <span class="badge badge-info">Topic: {result['topic']}</span>
        <span class="badge badge-info">Expert: {result['expert']}</span>
        <span class="badge badge-warning">Memories: {result['memories_used']}</span>
        <span class="badge badge-info">Trust: {result['trust_score']:.2f}</span>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Answers
        if compare_baseline and "baseline_answer" in st.session_state:
            ans_col1, ans_col2 = st.columns(2)
            
            with ans_col1:
                st.markdown("#### 🎯 Orchestrix Answer")
                st.markdown(f'<div class="answer-box">{result["response"]}</div>', 
                          unsafe_allow_html=True)
                
                with st.expander("View memories used"):
                    if result["memories_used"] > 0:
                        for i, snippet in enumerate(result["mem_snippets"], 1):
                            st.write(f"{i}. {snippet}")
                    else:
                        st.write("No memories used yet")
            
            with ans_col2:
                st.markdown("#### 💬 Baseline Answer")
                st.markdown(f'<div class="answer-box">{st.session_state.baseline_answer}</div>', 
                          unsafe_allow_html=True)
        else:
            st.markdown("#### 🎯 Orchestrix Answer")
            st.markdown(f'<div class="answer-box">{result["response"]}</div>', 
                      unsafe_allow_html=True)
            
            with st.expander("View memories used"):
                if result["memories_used"] > 0:
                    for i, snippet in enumerate(result["mem_snippets"], 1):
                        st.write(f"{i}. {snippet}")
                else:
                    st.write("No memories used yet")

# Right Column - Metrics
with col_right:
    metrics = tcm.metrics
    total = max(1, metrics["total"])
    
    st.markdown("### 📊 System Metrics")
    
    # Metric cards
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Queries</div>
        <div class="metric-value">{metrics['total']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    delegation_rate = (metrics['delegations'] / total) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Delegation Rate</div>
        <div class="metric-value">{delegation_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    avg_memories = np.mean(metrics['mems_used']) if metrics['mems_used'] else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Memories Used</div>
        <div class="metric-value">{avg_memories:.1f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Consolidations</div>
        <div class="metric-value">{metrics['consolidations']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent trust levels
    st.markdown("### 🤖 Agent Trust Levels")
    
    for agent in ["researcher", "analyst", "engineer"]:
        trust_values = []
        for key, params in tcm.trust.items():
            if key.startswith(f"{agent}:"):
                trust = params["alpha"] / (params["alpha"] + params["beta"])
                trust_values.append(trust)
        
        avg_trust = np.mean(trust_values) if trust_values else 0.5
        
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-name">{agent.capitalize()}</div>
            <div class="trust-bar">
                <div class="trust-fill" style="width: {avg_trust*100}%"></div>
            </div>
            <small>Trust Score: {avg_trust:.3f}</small>
        </div>
        """, unsafe_allow_html=True)

# Footer
with st.expander("ℹ️ How Orchestrix Works"):
    st.markdown("""
    ### Key Features
    
    **🎯 Expert Routing:** Agents build topic-specific trust over time through Thompson Sampling
    
    **🧠 Memory System:** Episodic memories consolidate into semantic knowledge
    
    **📈 Learning Loop:** Quality checks update trust scores (α/β parameters)
    
    **🔬 Comparison Mode:** See the difference vs. baseline model without orchestration
    
    ### Trust Model
    - Uses Beta-Bernoulli conjugate priors
    - Thompson Sampling for optimal exploration/exploitation
    - Trust scores evolve based on answer quality
    """)
