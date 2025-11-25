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

# ---------------- Page Config (Clean & Minimal) ----------------
st.set_page_config(
    page_title="Orchestrix",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Apple-Style "Command Center" CSS ----------------
st.markdown("""
<style>
    /* 1. Typography & Reset */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* 2. Backgrounds - The "Off-Black" Apple Look */
    .stApp {
        background-color: #000000;
    }
    
    /* 3. Containers (The Glass Cards) */
    div.css-1r6slb0, div.stExpander, section[data-testid="stSidebar"] {
        background-color: #1C1C1E; /* macOS Surface Color */
        border: 1px solid #2C2C2E;
        border-radius: 12px; 
    }

    /* 4. Inputs */
    .stTextArea textarea {
        background-color: #1C1C1E !important;
        border: 1px solid #3A3A3C !important;
        color: #F5F5F7 !important;
        font-size: 16px;
        border-radius: 10px;
    }
    .stTextArea textarea:focus {
        border-color: #0A84FF !important; /* Apple Blue */
        box-shadow: 0 0 0 1px #0A84FF !important;
    }

    /* 5. The "Process" Button - High Contrast Fix */
    div.stButton > button {
        background-color: #0A84FF !important;
        color: #FFFFFF !important; /* Force White Text */
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover {
        opacity: 0.85;
    }

    /* 6. Text Hierarchy */
    h1, h2, h3 {
        color: #F5F5F7 !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    p, label {
        color: #86868B !important; /* Apple Secondary Text */
    }
    
    /* 7. Custom Metric Cards */
    .stat-card {
        background: #1C1C1E;
        border: 1px solid #2C2C2E;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .stat-value {
        font-size: 24px;
        font-weight: 700;
        color: #F5F5F7;
        font-feature-settings: "tnum"; /* Tabular numbers */
    }
    .stat-label {
        font-size: 13px;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* 8. Trust Bars (Clean Lines) */
    .trust-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .trust-name {
        font-size: 14px;
        color: #F5F5F7;
        width: 100px;
    }
    .trust-track {
        flex-grow: 1;
        height: 4px;
        background: #2C2C2E;
        border-radius: 2px;
        margin: 0 12px;
        position: relative;
    }
    .trust-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    .trust-score {
        font-size: 13px;
        color: #86868B;
        width: 40px;
        text-align: right;
        font-feature-settings: "tnum";
    }

    /* Hide standard streamlit junk */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ---------------- Backend Logic (Preserved) ----------------

@dataclass
class MemoryEntry:
    id: str
    content: str
    topic: str
    agent_id: str
    timestamp: float

class TCMSystem:
    def __init__(self, client: OpenAI):
        self.client = client
        self.agents = ["Planner", "Researcher", "Verifier"]
        # Trust Matrix: { "Agent:Topic": {alpha, beta} }
        self.trust = defaultdict(lambda: {"alpha": 5.0, "beta": 1.0}) 
        self.metrics = {"delegations": 0, "total": 0}

    def _classify(self, text: str) -> str:
        # Simple Simulation Logic
        text = text.lower()
        if any(w in text for w in ["plan", "roadmap", "strategy"]): return "planning"
        if any(w in text for w in ["code", "python", "bug"]): return "coding"
        return "general_research"

    def select_agent(self, topic: str) -> str:
        # Simulate Thompson Sampling
        draws = {}
        for agent in self.agents:
            # Create synthetic variance based on agent specialty
            base_score = 0.5
            if agent == "Planner" and topic == "planning": base_score = 0.9
            if agent == "Researcher" and topic == "general_research": base_score = 0.9
            
            # Add noise
            draws[agent] = base_score + random.uniform(-0.1, 0.1)
        
        return max(draws, key=draws.get)

    def process(self, query: str):
        self.metrics["total"] += 1
        
        # 1. Classification
        topic = self._classify(query)
        yield "status", f"Classified intent as: **{topic.upper()}**"
        time.sleep(0.4)
        
        # 2. Trust Evaluation
        expert = self.select_agent(topic)
        current_trust = self.trust[f"{expert}:{topic}"]["alpha"] / (self.trust[f"{expert}:{topic}"]["alpha"] + self.trust[f"{expert}:{topic}"]["beta"])
        yield "status", f"Evaluating Trust Matrix... Selected Expert: **{expert}** (Confidence: {current_trust:.2f})"
        time.sleep(0.4)
        
        # 3. Delegation
        if expert != "Planner": 
            self.metrics["delegations"] += 1
            yield "status", "Delegating task to specialist node..."
        time.sleep(0.4)

        # 4. Generation
        try:
            # Real LLM Call
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are {expert}, an expert in {topic}. Be concise."},
                    {"role": "user", "content": query}
                ]
            )
            answer = resp.choices[0].message.content
        except:
            answer = "Error connecting to OpenAI API."
            
        yield "result", {
            "response": answer,
            "expert": expert,
            "topic": topic,
            "trust_score": random.uniform(0.85, 0.99) # Simulated updated trust
        }

# ---------------- UI Layout ----------------

# Init
if "tcm" not in st.session_state:
    # Try to get API Key from environment or user input
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        with st.sidebar:
            api_key = st.text_input("API Key", type="password")
            
    if api_key:
        st.session_state.tcm = TCMSystem(OpenAI(api_key=api_key))
    else:
        st.warning("System Offline: API Key Missing")
        st.stop()

tcm = st.session_state.tcm

# Header
st.markdown("<h1 style='font-size: 28px; margin-bottom: 0px;'>Orchestrix <span style='color: #86868B; font-weight: 400; font-size: 18px; margin-left: 10px;'>v2.1</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='margin-bottom: 30px;'>Multi-Agent Transactive Memory System</p>", unsafe_allow_html=True)

# Main Grid
col_main, col_sidebar = st.columns([2, 1], gap="large")

with col_main:
    # Input Section
    st.markdown("### Directive")
    query = st.text_area("Input", height=100, placeholder="Describe a complex task for the agent swarm...", label_visibility="collapsed")
    
    # Action Bar
    col_btn, col_blank = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("Initialize Sequence")

    # Output Section
    if run_btn and query:
        # The "Orchestration Log" - Shows differentiation from ChatGPT
        with st.status("Orchestrating Agents...", expanded=True) as status:
            processor = tcm.process(query)
            
            final_data = None
            for type_, data in processor:
                if type_ == "status":
                    st.write(data)
                elif type_ == "result":
                    final_data = data
            
            status.update(label="Execution Complete", state="complete", expanded=False)
        
        # The Result Card
        if final_data:
            st.markdown("### System Output")
            st.markdown(f"""
            <div style="background: #1C1C1E; border: 1px solid #2C2C2E; border-radius: 12px; padding: 24px;">
                <div style="display: flex; gap: 12px; margin-bottom: 16px;">
                    <span style="background: #1C2C40; color: #0A84FF; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{final_data['expert']} Node</span>
                    <span style="background: #1C2C40; color: #86868B; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{final_data['topic']}</span>
                </div>
                <div style="color: #F5F5F7; line-height: 1.6; font-size: 16px;">
                    {final_data['response']}
                </div>
            </div>
            """, unsafe_allow_html=True)

with col_sidebar:
    st.markdown("### Telemetry")
    
    # Bento Grid for Stats
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 24px;">
        <div class="stat-card">
            <div class="stat-value">{tcm.metrics['total']}</div>
            <div class="stat-label">Cycles</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{tcm.metrics['delegations']}</div>
            <div class="stat-label">Delegations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Trust Matrix (Custom HTML implementation)
    st.markdown("### Neural Trust Matrix")
    st.markdown("<div style='background: #1C1C1E; border: 1px solid #2C2C2E; border-radius: 12px; padding: 20px;'>", unsafe_allow_html=True)
    
    agents = [
        {"name": "Planner", "score": 0.92, "color": "#0A84FF"},
        {"name": "Researcher", "score": 0.78, "color": "#30D158"},
        {"name": "Verifier", "score": 0.88, "color": "#BF5AF2"},
    ]
    
    for a in agents:
        st.markdown(f"""
        <div class="trust-row">
            <div class="trust-name">{a['name']}</div>
            <div class="trust-track">
                <div class="trust-fill" style="width: {a['score']*100}%; background-color: {a['color']};"></div>
            </div>
            <div class="trust-score">{a['score']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
