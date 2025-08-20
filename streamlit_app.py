#!/usr/bin/env python3
# Premium dark UI for TCM + LLM Core Memory (no extra deps)
import os, time, hashlib
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
      /* --------- Fonts / Base --------- */
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300..800&display=swap');
      html, body, [data-testid="stAppViewContainer"] * { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
      [data-testid="stAppViewContainer"]{
        background:
          radial-gradient(60rem 40rem at -10% -10%, #1b1c20 0%, rgba(0,0,0,0) 60%),
          radial-gradient(50rem 30rem at 110% -10%, #15202e 0%, rgba(0,0,0,0) 60%),
          radial-gradient(70rem 40rem at 50% 120%, #13151a 0%, rgba(0,0,0,0) 70%),
          #0b0c0f;
      }
      [data-testid="stHeader"] { background: transparent; }

      :root{
        --glass: rgba(255,255,255,0.04);
        --glass-strong: rgba(255,255,255,0.06);
        --stroke: rgba(255,255,255,0.10);
        --muted:#9aa3ad; --text:#e7e9ee; --accent:#7fb4ff; --ok:#1fbf75; --bad:#e24c4b;
      }

      /* --------- Glass cards / shells --------- */
      .shell{
        border: 1px solid var(--stroke);
        background: linear-gradient(180deg, var(--glass-strong), rgba(255,255,255,0.02));
        border-radius: 16px; padding: 16px;
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 35px rgba(0,0,0,.35);
      }
      .shell.soft{ background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)); }

      .halo{
        position: relative;
        border-radius: 16px;
        padding: 1px;
        background: linear-gradient(135deg, rgba(127,180,255,.35), rgba(31,191,117,.35));
      }
      .halo > .inner{
        border-radius: 15px; background: rgba(14,15,19,.9); padding: 16px;
        border: 1px solid var(--stroke);
        backdrop-filter: blur(8px);
      }

      /* --------- Headline --------- */
      .headline{
        display:flex; align-items:end; gap:16px; margin-bottom:8px;
      }
      .headline h1{ margin:0; font-size:28px; letter-spacing:.2px; color:var(--text); }
      .headline .sub{ color:var(--muted); margin-top:4px; }

      /* --------- Disclaimer --------- */
      .disclaimer{
        border:1px solid var(--stroke); color:var(--muted);
        background: linear-gradient(180deg, rgba(127,180,255,.05), rgba(31,191,117,.05));
        padding:14px 16px; border-radius:12px;
      }

      /* --------- Grid --------- */
      .grid{
        display:grid; gap:14px; grid-template-columns: repeat(4, minmax(0,1fr));
      }
      @media (max-width:1100px){ .grid{ grid-template-columns: repeat(2,1fr);} }
      @media (max-width:680px){ .grid{ grid-template-columns: 1fr;} }

      /* --------- Metric chips --------- */
      .chip{
        padding: 14px; border-radius:14px;
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border:1px solid var(--stroke);
        box-shadow: 0 6px 20px rgba(0,0,0,.25);
        transform: translateY(6px); opacity:0; animation: rise .45s ease forwards;
      }
      .chip .label{ color:var(--muted); font-size:12px; letter-spacing:.4px; text-transform:uppercase; }
      .chip .value{ font-size:26px; font-weight:700; padding-top:2px; }

      @keyframes rise { to { transform: translateY(0); opacity:1; } }

      /* --------- Status line --------- */
      .status{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:8px; }
      .badge{
        padding:7px 12px; border-radius:999px; font-weight:700; font-size:12px;
        border:1px solid var(--stroke); background: rgba(255,255,255,.04); color:var(--text);
      }
      .badge-topic { color:#a0c7ff; border-color:rgba(127,180,255,.35); background: rgba(127,180,255,.08); }
      .badge-mem   { color:#ffd39c; border-color:rgba(255,196,127,.32); background: rgba(255,196,127,.08); }

      .delegation{ color:#fff; border:1px solid var(--stroke); }
      .delegation.ok{
        background: linear-gradient(90deg, #0f7a46, #12a35b);
        animation: pulseG 1.1s ease-in-out 3;
      }
      .delegation.bad{
        background: linear-gradient(90deg, #a32725, #d8423f);
        animation: pulseR 1.1s ease-in-out 3;
      }
      @keyframes pulseG{ 0%{filter:brightness(.9)} 50%{filter:brightness(1.18)} 100%{filter:brightness(1)} }
      @keyframes pulseR{ 0%{filter:brightness(.9)} 50%{filter:brightness(1.18)} 100%{filter:brightness(1)} }

      /* --------- Answer panel --------- */
      .answer{ background: rgba(255,255,255,.03); border:1px solid var(--stroke); border-radius:12px; padding:16px; color:var(--text); }

      /* --------- Agent cards --------- */
      .agent{
        border:1px solid var(--stroke); border-radius:14px; padding:14px;
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
      }
      .agent .name{ font-weight:700; }
      .bar{
        width:100%; height:8px; background:#13161b; border-radius:999px; border:1px solid var(--stroke); overflow:hidden;
      }
      .bar > span{
        display:block; height:100%;
        background: linear-gradient(90deg, #7fb4ff, #1fbf75);
        width:0%;
        animation: fill .7s ease forwards;
      }
      @keyframes fill{ from{ width:0%; } }

      /* --------- Buttons --------- */
      .btnrow{ display:flex; gap:10px; }
      .hint{ color:var(--muted); font-size:13px; }

      /* Hide default Streamlit block labels spacing */
      .st-eb{ gap: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------- API key ----------
def require_api_key():
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if key:
        os.environ["OPENAI_API_KEY"] = key.strip()
        return
    with st.sidebar:
        st.write("OpenAI API Key")
        val = st.text_input("Enter key", type="password", placeholder="sk-...")
        if val:
            os.environ["OPENAI_API_KEY"] = val.strip()
            st.success("Saved for this session.")
        else:
            st.stop()

require_api_key()

# ---------- Engine ----------
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
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self._embed_cache:
            return self._embed_cache[h]
        r = self.client.embeddings.create(model=self.embed_model, input=text)
        vec = np.array(r.data[0].embedding, dtype=np.float32)
        self._embed_cache[h] = vec
        return vec

    def add_memory(self, content: str, topic: str, agent_id: str, memory_type: str = "episodic") -> str:
        emb = self._embed(content)
        mem_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:10]
        e = MemoryEntry(mem_id, content, emb, topic, agent_id, time.time(), 0, memory_type)
        if memory_type == "episodic":
            self.episodic.append(e); self.working_memory.append(mem_id)
        elif memory_type == "semantic":
            self.semantic.setdefault(topic, []).append(e)
        else:
            self.procedural[topic] = e
        return mem_id

    def _all(self) -> List[MemoryEntry]:
        out = list(self.episodic)
        for v in self.semantic.values(): out.extend(v)
        out.extend(self.procedural.values())
        return out

    def retrieve(self, query: str, k: int = 5) -> List[MemoryEntry]:
        entries = self._all()
        if not entries: return []
        q = self._embed(query)
        mats = np.stack([e.embedding for e in entries], axis=0)
        sims = mats @ q / (np.linalg.norm(mats, axis=1) * np.linalg.norm(q) + 1e-9)
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            ent = entries[i]; ent.access_count += 1; out.append(ent)
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
            "planning":["plan","roadmap","strategy","schedule"],
            "research":["research","investigate","study","analyze"],
            "coding":["code","implement","bug","debug","write python"],
            "ml":["ml","model","train","neural","classifier"],
            "nlp":["nlp","transformer","llm","token","text"],
        }
        for k, kws in rules.items():
            if any(kw in t for kw in kws): return k
        return self.topics[0] if self.topics else "general"

    def _expert(self, topic: str) -> str:
        s = {}
        for a in self.agents:
            p = self.trust[f"{a}:{topic}"]
            s[a] = np.random.beta(p["alpha"], p["beta"])
        return max(s, key=s.get)

    def _format_mem(self, mems: List[MemoryEntry]) -> str:
        if not mems: return "No relevant memories."
        return "\n".join([f"{i}. ({m.memory_type}) {m.content[:220]}..." for i, m in enumerate(mems[:5], 1)])

    def _call_llm(self, prompt: str) -> str:
        try:
            r = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.7, max_tokens=500
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"[LLM error] {e}"

    def _quality(self, response: str, mems: List[MemoryEntry]) -> float:
        s = 0.5
        if len(response) > 120: s += 0.2
        if mems and any(m.content[:60] in response for m in mems): s += 0.3
        return min(1.0, s)

    def _update_trust(self, agent: str, topic: str, ok: bool):
        k = f"{agent}:{topic}"
        if ok: self.trust[k]["alpha"] += 1
        else:  self.trust[k]["beta"]  += 1

    def trust_score(self, agent: str, topic: str) -> float:
        p = self.trust[f"{agent}:{topic}"]; return p["alpha"]/(p["alpha"]+p["beta"])

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
        ans = self._call_llm(prompt)
        self.mem_local[expert].add_memory(f"Q: {query}\nA: {ans}", topic, expert, "episodic")

        q = self._quality(ans, used); self._update_trust(expert, topic, q > 0.7)
        self.metrics["consolidations"] += self.mem_local[expert].consolidate() + self.mem_shared.consolidate()

        hit = (len(local)/max(1,len(used))) if used else 0.0
        self.metrics["mems_used"].append(len(used)); self.metrics["hit_rate"].append(hit)

        return {"query":query,"response":ans,"topic":topic,"requester":requester,
                "expert":expert,"delegated":delegated,"memories_used":len(used),
                "trust_score": self.trust_score(expert, topic)}

    def summary(self) -> Dict:
        tot = max(1, self.metrics["total"])
        return {
            "total_queries": self.metrics["total"],
            "delegation_rate": self.metrics["delegations"]/tot,
            "avg_memories_used": float(np.mean(self.metrics["mems_used"])) if self.metrics["mems_used"] else 0.0,
            "avg_memory_hit_rate": float(np.mean(self.metrics["hit_rate"])) if self.metrics["hit_rate"] else 0.0,
            "total_consolidations": self.metrics["consolidations"],
            "trust": {k: v["alpha"]/(v["alpha"]+v["beta"]) for k,v in self.trust.items()},
        }

# ---------- Engine cache ----------
@st.cache_resource(show_spinner=False)
def get_engine():
    return TCMWithLLMMemoryLite(agents=["researcher","analyst","engineer"],
                                topics=["research","planning","coding","ml","nlp"])
tcm = get_engine()

# ---------- Header ----------
st.markdown("""
<div class="headline">
  <h1>TCM + LLM Core Memory</h1>
</div>
<div class="sub">Trust-based expert routing with memory-augmented answers</div>
""", unsafe_allow_html=True)

# ---------- Disclaimer ----------
if "hide_disc" not in st.session_state:
    st.session_state.hide_disc = False
if not st.session_state.hide_disc:
    st.markdown('<div class="disclaimer">This system learns which agent to trust for each topic over time. Early answers may route to a less-suited agent. As you interact, the trust distribution adapts (Thompson sampling over Beta priors) and routing improves.</div>', unsafe_allow_html=True)
    if st.checkbox("I understand — hide this notice from now on", value=False):
        st.session_state.hide_disc = True

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ---------- Layout ----------
left, right = st.columns([7,5], gap="large")

# Left: ask + result
with left:
    st.markdown('<div class="halo"><div class="inner">', unsafe_allow_html=True)
    st.write("Ask")
    q = st.text_area("Your question", label_visibility="collapsed",
                     placeholder="Plan an MVP rollout for a chatbot", height=120)
    c1, c2 = st.columns(2)
    if c1.button("Seed demo memories"):
        tcm.mem_shared.add_memory("Transformers weigh token-token interactions via self-attention.", "nlp", "seed", "semantic")
        tcm.mem_shared.add_memory("ML project: data → features → model → eval → iterate.", "planning", "seed", "semantic")
        tcm.mem_shared.add_memory("Cosine similarity = dot(a,b)/(|a||b|).", "coding", "seed", "semantic")
        st.success("Seeded.")
    run = c2.button("Ask")

    st.markdown('</div></div>', unsafe_allow_html=True)

    if run and q.strip():
        with st.spinner("Generating answer..."):
            out = tcm.process(q.strip())
            st.session_state.last = out

    if "last" in st.session_state:
        out = st.session_state.last
        delegated = bool(out["delegated"])
        cls = "delegation ok" if delegated else "delegation bad"
        txt = "Delegated to trusted expert" if delegated else "Not delegated (handled locally)"

        st.markdown('<div class="shell" style="margin-top:14px">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="status">
          <span class="badge {cls}">{txt}</span>
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
        st.markdown("<div class='hint'>Ask something to see routing and answers here.</div>", unsafe_allow_html=True)

# Right: metrics + agent trust
with right:
    s = tcm.summary()
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    def chip(label, value):
        st.markdown(f"""
          <div class="chip">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
          </div>
        """, unsafe_allow_html=True)
    chip("Total queries", s["total_queries"])
    chip("Delegation rate", f"{s['delegation_rate']*100:.1f}%")
    chip("Avg memories used", f"{s['avg_memories_used']:.2f}")
    chip("Avg memory hit rate", f"{s['avg_memory_hit_rate']:.2f}")
    chip("Total consolidations", s["total_consolidations"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Agent trust cards (aggregate across topics)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="shell soft">', unsafe_allow_html=True)
    st.write("Agents")
    trust = s["trust"]
    agents = ["researcher","analyst","engineer"]
    colA, colB, colC = st.columns(3)
    cols = [colA, colB, colC]
    for i, a in enumerate(agents):
        vals = [v for k,v in trust.items() if k.startswith(a + ":")]
        avg = sum(vals)/len(vals) if vals else 0.5
        with cols[i]:
            st.markdown(f"""
            <div class="agent">
              <div class="name">{a.title()}</div>
              <div class="bar"><span style="width:{avg*100:.0f}%"></span></div>
              <div class="hint">Avg trust: {avg:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.caption("Tip: trust improves as you interact more. Seed memories to give agents context quickly.")
