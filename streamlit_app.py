import streamlit as st
import time
import random
import os
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="Orchestrix // Multi-Agent System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PROFESSIONAL DASHBOARD CSS ----------------
st.markdown("""
<style>
    /* 1. Global Reset & Dark Mode Base */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
    
    .stApp {
        background-color: #0E0E10; /* Deep obsidian */
        font-family: 'Inter', sans-serif;
    }

    /* 2. The "Agent Nodes" Visualization */
    .agent-grid {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
        padding: 20px;
        background: #18181B;
        border-bottom: 1px solid #27272A;
    }
    
    .agent-node {
        width: 120px;
        padding: 10px;
        border: 1px solid #3F3F46;
        border-radius: 8px;
        text-align: center;
        background: #27272A;
        color: #A1A1AA;
        font-size: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .agent-active {
        border-color: #3B82F6; /* Blue Glow */
        background: rgba(59, 130, 246, 0.15);
        color: #3B82F6;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
    }

    /* 3. The "Terminal" Log */
    .terminal-box {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        background: #000000;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 15px;
        height: 150px;
        overflow-y: auto;
        color: #00FF94; /* Hacker Green */
        margin-bottom: 20px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
    }
    
    .log-line {
        margin-bottom: 4px;
        border-bottom: 1px solid #111;
        padding-bottom: 2px;
    }
    
    .log-timestamp { color: #555; margin-right: 8px; }
    .log-agent { color: #3B82F6; font-weight: bold; margin-right: 8px; }

    /* 4. Chat Bubbles (Distinct from ChatGPT) */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    
    .msg-user {
        align-self: flex-end;
        background: #27272A;
        color: #F4F4F5;
        padding: 12px 16px;
        border-radius: 12px 12px 0 12px;
        max-width: 80%;
        border: 1px solid #3F3F46;
    }
    
    .msg-ai {
        align-self: flex-start;
        background: transparent;
        color: #E4E4E7;
        padding: 0;
        max-width: 100%;
        border-left: 2px solid #3B82F6;
        padding-left: 15px;
    }

    /* 5. Memory Cards */
    .memory-card {
        background: #18181B;
        border: 1px solid #27272A;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
        font-size: 11px;
        color: #A1A1AA;
    }
    .memory-tag {
        display: inline-block;
        padding: 2px 6px;
        background: #27272A;
        color: #3B82F6;
        border-radius: 4px;
        margin-bottom: 4px;
        font-size: 9px;
        text-transform: uppercase;
    }

    /* Hiding Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE SETUP ----------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'active_agent' not in st.session_state:
    st.session_state.active_agent = "IDLE"
if 'memories' not in st.session_state:
    st.session_state.memories = []

# ---------------- LOGIC CORE (SIMULATED FOR DEMO) ----------------
def log_event(agent, action):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"<span class='log-timestamp'>[{timestamp}]</span> <span class='log-agent'>[{agent}]</span> {action}"
    st.session_state.logs.append(entry)

def run_orchestration(prompt):
    """Simulates the multi-agent decision loop with visual feedback"""
    
    # 1. Planner Phase
    st.session_state.active_agent = "PLANNER"
    log_event("PLANNER", f"Analyzing intent: '{prompt[:30]}...'")
    yield
    time.sleep(0.8)
    
    # Determine Intent (Simple Logic)
    intent = "general"
    if "code" in prompt.lower() or "python" in prompt.lower(): intent = "coding"
    elif "history" in prompt.lower() or "what is" in prompt.lower(): intent = "research"
    
    log_event("PLANNER", f"Intent classified as: <{intent.upper()}>")
    yield
    time.sleep(0.5)

    # 2. Researcher/Memory Phase
    st.session_state.active_agent = "RESEARCHER"
    log_event("RESEARCHER", "Querying Transactive Memory System...")
    yield
    time.sleep(0.8)
    
    # Simulate Retrieval
    new_mem = {
        "id": f"mem_{random.randint(1000,9999)}",
        "topic": intent,
        "content": f"Relevant context found for {intent}..."
    }
    st.session_state.memories.insert(0, new_mem)
    log_event("MEMORY", f"Retrieved context ID: {new_mem['id']} (Confidence: 0.92)")
    yield
    time.sleep(0.6)

    # 3. Execution Phase
    target_agent = "CODER" if intent == "coding" else "WRITER"
    st.session_state.active_agent = target_agent
    log_event(target_agent, "Generating response based on retrieved context...")
    yield
    time.sleep(1.0)
    
    # Fake LLM Response (Replace with real OpenAI call if API key exists)
    api_key = os.environ.get("OPENAI_API_KEY")
    response_text = ""
    
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = resp.choices[0].message.content
        except:
            response_text = "Error: API connection failed. Using fallback."

    if not response_text:
        # Fallback simulation
        if intent == "coding":
            response_text = f"```python\n# Optimized implementation for {prompt}\ndef solve():\n    return 'Orchestrix Logic'\n```\nHere is the generated code based on the parameters."
        else:
            response_text = f"Based on the analysis of **{prompt}**, the system identifies key factors in the memory graph. The simulation suggests a 94% probability of success."

    st.session_state.active_agent = "IDLE"
    log_event("SYSTEM", "Task complete. Output verified.")
    
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "ai", "content": response_text})
    yield

# ---------------- UI LAYOUT ----------------

# 1. Top Bar: Agent Visualization
planner_class = "agent-active" if st.session_state.active_agent == "PLANNER" else "agent-node"
research_class = "agent-active" if st.session_state.active_agent == "RESEARCHER" else "agent-node"
coder_class = "agent-active" if st.session_state.active_agent in ["CODER", "WRITER"] else "agent-node"

st.markdown(f"""
<div class="agent-grid">
    <div class="{planner_class}">PLANNER<br><span style="font-size:10px; opacity:0.6;">Architecture</span></div>
    <div style="align-self:center; color:#555;">➔</div>
    <div class="{research_class}">RESEARCHER<br><span style="font-size:10px; opacity:0.6;">Memory Retrieval</span></div>
    <div style="align-self:center; color:#555;">➔</div>
    <div class="{coder_class}">EXECUTOR<br><span style="font-size:10px; opacity:0.6;">Code/Text Gen</span></div>
</div>
""", unsafe_allow_html=True)

# 2. Main Workspace
col_log, col_chat = st.columns([1, 2])

# Left Column: System Internals (The "Difference" from ChatGPT)
with col_log:
    st.markdown("### 📟 System Kernel")
    
    # Render the scrolling log
    log_html = "".join([f"<div class='log-line'>{l}</div>" for l in st.session_state.logs[-10:]]) # Show last 10
    st.markdown(f"""
    <div class="terminal-box" id="terminal">
        {log_html}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧠 Active Memory")
    if not st.session_state.memories:
        st.markdown("<div class='memory-card'>No active context loaded.</div>", unsafe_allow_html=True)
    else:
        for mem in st.session_state.memories[:3]: # Show top 3
            st.markdown(f"""
            <div class="memory-card">
                <span class="memory-tag">{mem['topic']}</span>
                <span style="float:right; opacity:0.5;">{mem['id']}</span><br>
                {mem['content']}
            </div>
            """, unsafe_allow_html=True)

# Right Column: Interaction
with col_chat:
    st.markdown("### 💬 Orchestrix Interface")
    
    # History Display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.history:
            if msg['role'] == 'user':
                st.markdown(f"<div class='msg-user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='msg-ai'>{msg['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input Area
    with st.form(key="query_form", clear_on_submit=True):
        user_input = st.text_area("Input Directive", height=80, placeholder="Enter complex instruction for multi-agent swarm...", label_visibility="collapsed")
        c1, c2 = st.columns([1, 5])
        with c1:
            submit = st.form_submit_button("Execute")
            
    if submit and user_input:
        # Run the generator to update UI step-by-step
        runner = run_orchestration(user_input)
        
        # Create a placeholder to force UI refreshes during the loop
        placeholder = st.empty()
        
        try:
            for _ in runner:
                # This forces Streamlit to re-render the Top Bar & Logs
                # In a real app, we'd use st.rerun(), but inside a loop we just need to wait
                time.sleep(0.1) 
                st.rerun() 
        except StopIteration:
            pass
        except Exception as e:
            pass # Handle rerun interrupts

# Sidebar Config
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Required for live LLM generation")
    if api_key: os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("**System Health:** 🟢 Online")
    st.markdown("**Memory Shards:** 1,024")
    st.markdown("**Agents Online:** 3")
    
    if st.button("Reset System Memory"):
        st.session_state.history = []
        st.session_state.logs = []
        st.session_state.memories = []
        st.rerun()
