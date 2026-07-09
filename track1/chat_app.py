"""TerraceRoute Chat — light, Fireworks-only demo UI for Track 1.

The scored *batch* agent (agent/, track1/Dockerfile) is local-first and spends zero
API tokens whenever it can prove an answer. This chatbot is the human-facing *demo*:
it reuses the same token-efficient routing brain — free deterministic short-circuits,
terse per-category prompts, and per-category max_tokens caps — but routes straight to
Fireworks so the live UX stays fast (a 3B model on CPU is too slow to chat with).

Secrets (the Fireworks API key/base URL) are read at RUNTIME from st.secrets / env.
They are NEVER baked into the image and NEVER displayed in the UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import streamlit as st

st.set_page_config(page_title="TerraceRoute Chat", page_icon="🪜", layout="centered")


# --- Runtime secrets → environment, BEFORE importing the agent (its config singleton
#     reads the environment at import time). Only the API key is truly required; the
#     base URL and model have public defaults so "just the Fireworks key" is enough. ---
_SECRET_KEYS = ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS", "REMOTE_MAX_TOKENS")
_DEFAULTS = {
    "FIREWORKS_BASE_URL": "https://api.fireworks.ai/inference/v1",
    "ALLOWED_MODELS": "accounts/fireworks/models/llama-v3p1-8b-instruct",
}


def _load_secrets_into_env() -> None:
    for key in _SECRET_KEYS:
        val = None
        try:
            val = st.secrets[key]  # type: ignore[index]
        except Exception:
            val = None
        if val in (None, ""):
            val = os.environ.get(key)
        if val in (None, ""):
            val = _DEFAULTS.get(key)
        if val not in (None, ""):
            os.environ[key] = str(val)


_load_secrets_into_env()

from agent import classify as C          # noqa: E402
from agent import verify as V            # noqa: E402
from agent.config import config          # noqa: E402
from agent.fireworks import Fireworks    # noqa: E402
from agent.solve import (                # noqa: E402
    _SYS_REMOTE,
    _REMOTE_MAXTOK,
    _coerce,
    _DEFAULT_SYS,
)


def _sync_config() -> None:
    """Keep the config singleton in step with the (possibly updated) environment."""
    config.fireworks_api_key = os.environ.get("FIREWORKS_API_KEY", "")
    config.fireworks_base_url = os.environ.get("FIREWORKS_BASE_URL", config.fireworks_base_url)
    config.allowed_models_raw = os.environ.get("ALLOWED_MODELS", config.allowed_models_raw)


_sync_config()

_CODE_CATS = {C.CODE_GEN, C.CODE_DEBUG}


@dataclass
class ChatResult:
    answer: str
    category: str
    route: str            # deterministic | remote | error
    tokens: int = 0
    model: str = ""


def route_message(prompt: str) -> ChatResult:
    """Cheapest tier that works: free deterministic check → terse capped Fireworks call."""
    category = C.classify(prompt)

    # FREE + instant: letter counting is a universal LLM blind spot we answer in Python.
    cnt = V.try_count_letter(prompt)
    if cnt is not None:
        return ChatResult(str(cnt), category, "deterministic")

    if not config.can_remote:
        return ChatResult(
            "⚠️ Fireworks belum dikonfigurasi. Set `FIREWORKS_API_KEY` di Streamlit "
            "secrets (base URL & model punya default).",
            category, "error",
        )

    fw = Fireworks()
    prefer = "code" if category in _CODE_CATS else None
    model = fw.pick_model(prefer=prefer)
    res = fw.generate(
        prompt,
        system=_SYS_REMOTE.get(category, _DEFAULT_SYS),
        model=model,
        temperature=0.0,
        max_tokens=_REMOTE_MAXTOK.get(category, config.remote_max_tokens),
    )
    if res is None or not res.text:
        return ChatResult(
            "⚠️ Panggilan Fireworks gagal — cek API key, nama model (`ALLOWED_MODELS`), "
            "atau kuota.",
            category, "error",
        )
    answer = _coerce(category, res.text) or res.text
    return ChatResult(answer, category, "remote", tokens=res.total_tokens, model=res.model)


def _receipt(r: ChatResult) -> str:
    cat = r.category or "general"
    if r.route == "deterministic":
        return f"🟢 Local · deterministic · **0 tokens** · {cat}"
    if r.route == "remote":
        short = r.model.split("/")[-1] if r.model else "fireworks"
        return f"🟡 Fireworks · {short} · **{r.tokens} tokens** · {cat}"
    return f"⚪ {cat} · no tokens spent"


# ----------------------------- session state -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, receipt}]
    st.session_state.tokens = 0
    st.session_state.n_free = 0
    st.session_state.n_remote = 0


# ------------------------------- sidebar --------------------------------
with st.sidebar:
    st.markdown("## 🪜 TerraceRoute")
    st.caption(
        "Hybrid **token-efficient** routing. Free deterministic answers cost **0 tokens**; "
        "everything else goes to Fireworks with a terse, per-category token cap."
    )
    if config.can_remote:
        model_short = (config.allowed_models[0].split("/")[-1] if config.allowed_models else "?")
        st.success(f"Fireworks: connected · `{model_short}`")
    else:
        st.error("Fireworks: not configured — add `FIREWORKS_API_KEY` in secrets.")

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Fireworks tokens", st.session_state.tokens)
    c2.metric("Free (0-tok)", st.session_state.n_free)
    st.caption(f"{st.session_state.n_remote} via Fireworks · {st.session_state.n_free} local-free")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tokens = 0
        st.session_state.n_free = 0
        st.session_state.n_remote = 0
        st.rerun()


# ------------------------------- main chat -------------------------------
st.title("TerraceRoute Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("receipt"):
            st.caption(msg["receipt"])

if prompt := st.chat_input("Ask anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Routing…"):
            r = route_message(prompt)
        st.markdown(r.answer)
        receipt = _receipt(r)
        st.caption(receipt)

    st.session_state.messages.append(
        {"role": "assistant", "content": r.answer, "receipt": receipt}
    )
    if r.route == "remote":
        st.session_state.tokens += r.tokens
        st.session_state.n_remote += 1
    elif r.route == "deterministic":
        st.session_state.n_free += 1
    st.rerun()
