#!/usr/bin/env python3
import os, json
import streamlit as st
from tcm_core.study2_lite import TCMWithLLMMemoryLite

st.set_page_config(page_title="TCM + LLM Core Memory", page_icon="🧠", layout="centered")

# --- Secrets / API key ---
# Prefer Streamlit Secrets; fallback to env
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- Build / restore system in session ---
def build_system():
    agents = ["researcher", "analyst", "engineer"]
    topics = ["research", "planning", "coding", "ml", "nlp"]
    sys = TCMWithLLMMemoryLite(agents=agents, topics=topics)
    # restore from session, if present
    if "tcm_state" in st.session_state:
        try:
            sys.import_state(st.session_state["tcm_state"])
        except Exception:
            pass
    return sys

if "tcm" not in st.session_state:
    try:
        st.session_state.tcm = build_system()
    except Exception as e:
        st.error(f"Startup error: {e}")
        st.stop()

tcm = st.session_state.tcm

# --- UI ---
st.title("🧠 TCM + LLM Core Memory")
st.caption("Trust-based delegation + memory-augmented reasoning")

with st.sidebar:
    st.subheader("⚙️ Controls")
    if st.button("Seed shared knowledge"):
        seeds = [
            ("Transformers use self-attention to weigh token-token interactions.", "nlp", "seed"),
            ("Basic ML project plan: data → features → model → eval → iterate.", "planning", "seed"),
            ("Cosine similarity is dot(a,b)/(|a||b|).", "coding", "seed"),
        ]
        for text, topic, who in seeds:
            tcm.mem_shared.add_memory(content=text, topic=topic, agent_id=who, memory_type="semantic")
        st.success("Seeded 3 entries into shared memory.")
        st.session_state["tcm_state"] = tcm.export_state()

    if st.button("Reset session state"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.subheader("Trust (mean of Beta)")
    trust = tcm.summary().get("trust", {})
    if not trust:
        st.caption("no trust yet")
    else:
        for k,v in trust.items():
            st.write(f"{k}: **{v:.3f}**")

st.markdown("### Ask something")
q = st.text_area("Your question (natural language)", height=120, placeholder="e.g., Plan an MVP rollout for a chatbot")

col1, col2 = st.columns([1,1])
with col1:
    ask = st.button("Ask", type="primary")
with col2:
    show_summary = st.button("Show metrics")

if ask and q.strip():
    out = tcm.process(q.strip())
    st.markdown("#### Response")
    st.write(out["response"])
    st.info(f"Expert: {out['expert']}  •  Topic: {out['topic']}  •  Delegated: {out['delegated']}  •  Memories used: {out['memories_used']}  •  Trust: {out['trust_score']:.3f}")
    st.session_state["tcm_state"] = tcm.export_state()

if show_summary:
    st.markdown("#### Metrics")
    s = tcm.summary()
    st.json(s)
