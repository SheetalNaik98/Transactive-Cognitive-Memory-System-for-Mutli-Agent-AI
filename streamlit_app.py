#!/usr/bin/env python3
import os, json, time
import streamlit as st

# Make sure our package is importable when running on Streamlit Cloud
from tcm_core.study2_lite import TCMWithLLMMemoryLite

st.set_page_config(page_title="TCM-core", page_icon="🧠", layout="wide")

# -----------------------------
# Secrets / API key
# -----------------------------
# Prefer Streamlit secrets → fallback to env var
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_KEY:
    st.error("Set your OpenAI key in Streamlit → App → Settings → Secrets as `OPENAI_API_KEY`.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_KEY  # used by study2_lite internals

# -----------------------------
# Build / cache the TCM system
# -----------------------------
@st.cache_resource(show_spinner=False)
def init_system():
    agents = ["researcher", "analyst", "engineer"]
    topics = ["research", "planning", "coding", "ml", "nlp"]
    return TCMWithLLMMemoryLite(agents=agents, topics=topics)

tcm = init_system()
if "history" not in st.session_state:
    st.session_state.history = []  # chat history (list of dicts)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Seed default knowledge", use_container_width=True):
        n = tcm.seed_defaults()
        st.success(f"Seeded {n} shared semantic memories")

    st.divider()
    st.subheader("➕ Add memory")
    new_topic = st.selectbox("Topic", ["planning","research","coding","ml","nlp","general"], index=0)
    new_text = st.text_area("Content", placeholder="Paste a fact/snippet to store in shared memory…")
    if st.button("Save to shared memory", use_container_width=True, disabled=not new_text.strip()):
        tcm.add_shared_memory(new_text.strip(), new_topic, who="user")
        st.toast("Saved to shared semantic memory", icon="✅")

    st.divider()
    st.subheader("📊 Live metrics")
    m = tcm.summary()
    st.metric("Total queries", m["total_queries"])
    st.metric("Delegation rate", f"{m['delegation_rate']*100:.1f}%")
    st.metric("Avg memories used", f"{m['avg_memories_used']:.2f}")
    st.metric("Memory hit rate", f"{m['avg_memory_hit_rate']:.2f}")
    st.metric("Consolidations", m["total_consolidations"])

# -----------------------------
# Main area
# -----------------------------
st.title("🧠 TCM-core — Study 2 (TCM + LLM Memory)")
st.caption("Trust-based delegation (Thompson sampling) + hierarchical memory (episodic/semantic).")

# Chat-like input
user_query = st.chat_input("Ask anything (e.g., 'Explain self-attention' or 'Plan an MVP rollout…')")

if user_query:
    with st.spinner("Thinking with TCM…"):
        out = tcm.process(user_query)
    st.session_state.history.append(out)

# Show conversation
for item in st.session_state.history[-10:]:  # show last 10
    with st.chat_message("user"):
        st.write(item["query"])
    with st.chat_message("assistant"):
        st.write(item["response"])
        st.caption(f"Agent **{item['expert']}** | Topic **{item['topic']}** | Delegated: **{item['delegated']}** | Trust now: **{item['trust_score']:.3f}**")

st.divider()

# Trust table
st.subheader("Trust by (agent:topic)")
trust = tcm.summary().get("trust", {})
if not trust:
    st.info("No trust observations yet.")
else:
    # Pretty print in a small grid
    rows = []
    for k, v in sorted(trust.items()):
        agent, topic = k.split(":")
        rows.append({"agent": agent, "topic": topic, "trust": round(v, 3)})
    st.dataframe(rows, hide_index=True, use_container_width=True)
