#!/usr/bin/env python3
import os, json, importlib.util
from pathlib import Path
import streamlit as st

# -----------------------------
# Secrets / API key
# -----------------------------
if not os.getenv("OPENAI_API_KEY"):
    # Prefer Streamlit Secrets; fall back to env if present
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# -----------------------------
# Dynamic import of file with spaces & '+'
# -----------------------------
BASE_DIR = Path(__file__).parent
MOD_PATH = BASE_DIR / "tcm_core" / "TCM + LLM Core Memory.py"
if not MOD_PATH.exists():
    st.error(f"Backend file not found at: {MOD_PATH}")
    st.stop()

spec = importlib.util.spec_from_file_location("tcm_llm_core", str(MOD_PATH))
tcm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcm_mod)

TCM = tcm_mod.TCMWithLLMMemoryLite

# -----------------------------
# App state
# -----------------------------
if "tcm" not in st.session_state:
    st.session_state.tcm = TCM(
        agents=["researcher","analyst","engineer"],
        topics=["research","planning","coding","ml","nlp"],
        chat_model="gpt-4o-mini",
        embed_model="text-embedding-3-small"
    )

tcm = st.session_state.tcm

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="TCM + LLM Core Memory", page_icon="🧠", layout="centered")

st.title("🧠 TCM + LLM Core Memory")
st.caption("Trust-based delegation + memory-augmented answering (Study 2)")

with st.expander("Seed shared knowledge (safe to re-run)", expanded=False):
    if st.button("Seed now", type="primary"):
        tcm.seed_shared()
        st.success("Seeded 3 semantic memories into the shared store.")

st.markdown("### Ask a question")
q = st.text_area("Your query", placeholder="e.g., Explain self-attention in transformers")
cols = st.columns(2)
with cols[0]:
    requester = st.selectbox("Requesting agent", ["auto","researcher","analyst","engineer"], index=0)
with cols[1]:
    run = st.button("Ask", type="primary")

if run and q.strip():
    req = None if requester == "auto" else requester
    out = tcm.process(q.strip(), requester=req)
    st.markdown("#### Answer")
    st.write(out["response"])
    st.caption(f"Expert: **{out['expert']}** · Delegated: **{out['delegated']}** · Topic: **{out['topic']}** · Memories used: **{out['memories_used']}** · Trust now: **{out['trust_score']:.3f}**")

st.markdown("### Metrics")
m = tcm.summary()
met1, met2, met3 = st.columns(3)
met1.metric("Delegation rate", f"{m['delegation_rate']*100:.1f}%")
met2.metric("Avg memories used", f"{m['avg_memories_used']:.2f}")
met3.metric("Consolidations", f"{m['total_consolidations']}")

st.markdown("#### Trust (by agent:topic)")
if m["trust"]:
    import pandas as pd
    trust_rows = [{"agent_topic": k, "trust": v} for k, v in m["trust"].items()]
    st.dataframe(pd.DataFrame(trust_rows).sort_values("trust", ascending=False), hide_index=True, use_container_width=True)
else:
    st.info("No trust observations yet. Ask a question to start learning.")
