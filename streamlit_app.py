#!/usr/bin/env python3
import os, json, time, hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict, deque

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import streamlit as st

# ---------------- Page / Theme ----------------
st.set_page_config(page_title="TCM + LLM Core Memory", layout="wide")

def inject_css():
    st.markdown("""
    <style>
      :root{
        --bg:#0b0b0c; --panel:#131316; --muted:#9ca3af;
        --border:#1f1f23; --text:#e7e7ea; --accent:#5ea3ff;
        --ok:#1fbf75; --bad:#e24c4b; --chip:#1a1a1f;
      }
      /* app background */
      [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
      [data-testid="stHeader"] { background: transparent; }
      .hero {
        width:100%; border-radius:16px; overflow:hidden;
        background: linear-gradient(135deg, #111 0%, #181b20 100%);
        border:1px solid var(--border); margin-bottom:16px;
      }
      .hero img { display:block; width:100%; height:auto; }
      .wrap { padding:18px; }
      .disclaimer {
        background: #121217; border:1px solid var(--border);
        border-radius:12px; padding:14px 16px; color: var(--muted);
      }
      .grid { display:grid; gap:14px; grid-template-columns: repeat(4, minmax(0,1fr)); }
      @media (max-width:1100px){ .grid{ grid-template-columns: repeat(2,1fr);} }
      @media (max-width:640px){ .grid{ grid-template-columns: 1fr;} }

      .card {
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 14px; padding: 16px; box-shadow: 0 6px 24px rgba(0,0,0,.25);
        transform: translateY(6px); opacity:.0; animation: rise .45s ease forwards;
      }
      @keyframes rise { to { transform: translateY(0); opacity:1; } }

      .metric {
        display:flex; flex-direction:column; gap:4px;
      }
      .metric .label { color: var(--muted); font-size:12px; letter-spacing:.4px; text-transform:uppercase; }
      .metric .value { font-size:28px; font-weight:600; line-height:1.1; }

      .status-line { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:8px; }
      .badge {
        padding:6px 10px; border-radius:999px; font-weight:600; font-size:12px;
        border:1px solid var(--border); background: var(--chip); color: var(--text);
      }
      .badge-topic { color:#a0c7ff; border-color:#264361; background:#0f1823; }
      .badge-mem   { color:#ffc37b; border-color:#5a3b17; background:#1e1309; }

      .delegation {
        padding:8px 12px; border-radius:999px; font-weight:700; letter-spacing:.2px;
        border:1px solid var(--border); color:#fff;
      }
      .delegation.ok   { background: linear-gradient(90deg, #0f7a46, #12a35b); animation: greenPulse 1.2s ease-in-out 3; }
      .delegation.bad  { background: linear-gradient(90deg, #a32725, #d8423f); animation: redPulse 1.2s ease-in-out 3; }
      @keyframes greenPulse { 0%{filter:brightness(.9)} 50%{filter:brightness(1.15)} 100%{filter:brightness(1)} }
      @keyframes redPulse   { 0%{filter:brightness(.9)} 50%{filter:brightness(1.15)} 100%{filter:brightness(1)} }

      .answer {
        background:#101014; border:1px solid var(--border);
        border-radius:12px; padding:16px; margin-top:10px; color: var(--text);
      }
      .controls { display:flex; gap:10px; }
      .hint { color: var(--muted); font-size: 13px; margin-top:6px; }
      .muted { color: var(--muted); }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------------- API key ----------------
def require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if key:
        os.environ["OPENAI_API_KEY"] = key.strip()
        return key.strip()
    with st.sidebar:
        st.write("**OpenAI API Key**")
        key_in = st.text_input("Provide your key", type="password", placeholder="sk-...")
        if key_in:
            os.environ["OPENAI_API_KEY"] = key_in.strip()
            st.success("API key saved for this session.")
            return key_in.strip()
        st.stop()

_ = require_api_key()

# ---------------- Engine ----------------
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
        out = list(self.episodic)
        for arr in self.semantic.values(): out.extend(arr)
        out.extend(self.procedural.values())
        return out

    def retrieve(self, query: str, k: int = 5) -> List[MemoryEntry]:
        entries = self._all_entries()
        if not entries: return []
        q = self._embed(query)
        mats = np.stack([e.embedding for e in entries], axis=0)
        denom = (np.linalg.norm(mats, axis=1) * np.linalg.norm(q) + 1e-9)
        sims = mats.dot(q) / denom
        idxs = np.argsort(-sims)[:k]
        out = []
        for i in idxs:
            e = entries[i]; e.access_count += 1; out.append(e)
        return out

    def consolidate(self) -> int:
        moved = 0; keep: List[MemoryEntry] = []
        for e in self.episodic:
            if e.access_count >= self.consolidation_threshold:
                self.semantic.setdefault(e.topic, []).append(
                    MemoryEntry(
                        id=f"cons_{e.id}", content=f"[Consolidated] {e.content}",
                        embedding=e.embedding, topic=e.topic, agent_id=e.agent_id,
                        timestamp=time.time(), memory_type="semantic",
                        metadata={"orig": e.id, "access_count": e.access_count},
                    )
                ); moved += 1
            else: keep.append(e)
        self.episodic = keep; return moved

class TCMWithLLMMemoryLite:
    def __init__(self, agents: List[str], topics: List[str], chat_model: str = "gpt-4o-mini"):
        self.client = OpenAI()  # reads from env
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
            "planning": ["plan","roadmap","strategy","schedule"],
            "research": ["research","investigate","study","analyze"],
            "coding":   ["code","implement","bug","debug","write python"],
            "ml":       ["ml","model","train","neural","classifier"],
            "nlp":      ["nlp","transformer","llm","token","text"],
        }
        for topic, kws in rules.items():
            if any(kw in t for kw in kws): return topic
        return self.topics[0] if self.topics else "general"

    def _expert(self, topic: str) -> str:
        scores = {}
        for a in self.agents:
            p = self.trust[f"{a}:{topic}"]; scores[a] = np.random.beta(p["alpha"], p["beta"])
        return max(scores, key=scores.get)

    def _format_mem(self, mems: List[MemoryEntry]) -> str:
        if not mems: return "No relevant memories."
        return "\n".join([f"{i}. ({m.memory_type}) {m.content[:220]}..." for i,m in enumerate(mems[:5],1)])

    def _call_llm(self, prompt: str) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.7, max_tokens=450,
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"[LLM error] {e}"

    def _quality(self, response: str, mems: List[MemoryEntry]) -> float:
        s = 0.5
        if len(response) > 120: s += 0.2
        if mems and any(m.content[:60] in response for m in mems): s += 0.3
        return min(1.0, s)

    def _update_trust(self, agent: str, topic: str, success: bool):
        k = f"{agent}:{topic}"
        if success: self.trust[k]["alpha"] += 1
        else:       self.trust[k]["beta"]  += 1

    def trust_score(self, agent: str, topic: str) -> float:
        p = self.trust[f"{agent}:{topic}"]; return p["alpha"] / (p["alpha"] + p["beta"])

    def process(self, query: str, requester: Optional[str] = None) -> Dict:
        self.metrics["total"] += 1
        topic = self._topic(query)
        requester = requester or np.random.choice(self.agents)
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
            content=f"Q: {query}\nA: {answer}", topic=topic, agent_id=expert, memory_type="episodic"
        )
        q = self._quality(answer, used); self._update_trust(expert, topic, success=(q > 0.7))

        cons_local = self.mem_local[expert].consolidate()
        cons_shared = self.mem_shared.consolidate()
        self.metrics["consolidations"] += (cons_local + cons_shared)

        hit = (len(local) / max(1, len(used))) if used else 0.0
        self.metrics["mems_used"].append(len(used)); self.metrics["hit_rate"].append(hit)

        return {
            "query": query, "response": answer, "topic": topic, "requester": requester,
            "expert": expert, "delegated": delegated, "memories_used": len(used),
            "trust_score": self.trust_score(expert, topic),
        }

    def summary(self) -> Dict:
        tot = max(1, self.metrics["total"])
        return {
            "total_queries": self.metrics["total"],
            "delegation_rate": self.metrics["delegations"] / tot,
            "avg_memories_used": float(np.mean(self.metrics["mems_used"])) if self.metrics["mems_used"] else 0.0,
            "avg_memory_hit_rate": float(np.mean(self.metrics["hit_rate"])) if self.metrics["hit_rate"] else 0.0,
            "total_consolidations": self.metrics["consolidations"],
            "trust": {k: v["alpha"]/(v["alpha"]+v["beta"]) for k,v in self.trust.items()},
        }

# ---------------- Engine cache ----------------
@st.cache_resource(show_spinner=False)
def get_engine():
    agents = ["researcher", "analyst", "engineer"]
    topics = ["research", "planning", "coding", "ml", "nlp"]
    return TCMWithLLMMemoryLite(agents=agents, topics=topics)

tcm = get_engine()

# ---------------- Header / Optional hero image ----------------
with st.container():
    hero_url = st.sidebar.text_input("Header image URL (optional)", value="")
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    if hero_url.strip():
        st.markdown(f'<img src="{hero_url.strip()}" alt="header"/>', unsafe_allow_html=True)
    st.markdown('<div class="wrap">', unsafe_allow_html=True)
    st.markdown("<h2 style='margin:0'>TCM + LLM Core Memory</h2>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Trust-based expert routing with memory-augmented answers</div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------- Disclaimer (trust building) ----------------
if "hide_disclaimer" not in st.session_state:
    st.session_state.hide_disclaimer = False

if not st.session_state.hide_disclaimer:
    with st.container():
        st.markdown("""
        <div class="disclaimer">
          <strong>Heads up:</strong> This system learns which agent to trust for each topic over time.
          Early answers may route to a less-suited agent. As you ask more questions and we observe quality,
          the trust distribution adapts (via Thompson sampling over Beta priors) and routing improves.
        </div>
        """, unsafe_allow_html=True)
        if st.checkbox("I understand — hide this notice from now on", value=False):
            st.session_state.hide_disclaimer = True

# ---------------- Main columns ----------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Ask")
    q = st.text_area("Your question", placeholder="Plan an MVP rollout for a chatbot", height=120, label_visibility="collapsed")
    colA, colB = st.columns(2)
    if colA.button("Seed demo memories"):
        tcm.mem_shared.add_memory("Transformers use self-attention to weigh token-token interactions.", "nlp", "seed", "semantic")
        tcm.mem_shared.add_memory("Basic ML project plan: data → features → model → eval → iterate.", "planning", "seed", "semantic")
        tcm.mem_shared.add_memory("Cosine similarity is dot(a,b)/(|a||b|).", "coding", "seed", "semantic")
        st.success("Seeded shared semantic memory.")

    run = colB.button("Ask")
    st.markdown('</div>', unsafe_allow_html=True)

    if run and q.strip():
        out = tcm.process(q.strip())
        st.session_state.last_out = out

    if "last_out" in st.session_state:
        out = st.session_state.last_out
        delegated = bool(out["delegated"])
        badge_cls = "delegation ok" if delegated else "delegation bad"
        badge_txt = "Delegated (trusted expert)" if delegated else "Not delegated (local)"
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="status-line">
                  <span class="{badge_cls}">{badge_txt}</span>
                  <span class="badge badge-topic">Topic: {out['topic']}</span>
                  <span class="badge badge-mem">Memories: {out['memories_used']}</span>
                  <span class="badge">Trust: {out['trust_score']:.3f}</span>
                  <span class="badge">Expert: {out['expert']}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="answer">', unsafe_allow_html=True)
            st.write(out["response"])
            st.markdown('</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hint">Ask something to see routing and responses here.</div>', unsafe_allow_html=True)

with right:
    s = tcm.summary()
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    def metric_card(label: str, value: str):
        st.markdown(f"""
          <div class="card">
            <div class="metric">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
            </div>
          </div>
        """, unsafe_allow_html=True)

    metric_card("Total queries", str(s["total_queries"]))
    metric_card("Delegation rate", f"{s['delegation_rate']*100:.1f}%")
    metric_card("Avg memories used", f"{s['avg_memories_used']:.2f}")
    metric_card("Avg memory hit rate", f"{s['avg_memory_hit_rate']:.2f}")

    metric_card("Total consolidations", str(s["total_consolidations"]))
    # Trust cards (top 3 entries)
    trust_items = sorted(s["trust"].items(), key=lambda kv: -kv[1])[:3]
    for k, v in trust_items:
        metric_card(f"Trust • {k}", f"{v:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="hint">You can customize the header image via the sidebar.</div>', unsafe_allow_html=True)
