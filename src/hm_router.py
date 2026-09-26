"""hm-router: factored routing over (model, harness) routes.

    S(q, m, h) = w' [LN(u_m * q) ; LN(u_h * q) ; LN(u_m * u_h * q)]

q is a linear projection of the query embedding; u_m and u_h are model and
harness representations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import CACHE, encode

D_Q = 32   # query projection width
D_E = 16   # model/harness embedding width
EPOCHS = 250
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 1024
NOISE_STD = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def route_factors(routes: pd.DataFrame):
    """Model and harness codes for each route, plus the harness names."""
    cm = pd.Categorical(routes["model"])
    ch = pd.Categorical(routes["scaffold"])
    return cm.codes.astype(int), ch.codes.astype(int), list(ch.categories)


def summary_text_matrix(names: list[str]) -> torch.Tensor:
    """Embeddings of the harness summaries, one row per harness."""
    cache = CACHE / "harness_summaries.npz"
    if cache.exists():
        with np.load(cache) as cached:
            table = dict(zip(cached["scaffold"], cached["embeddings"]))
    else:
        slugs = sorted(HARNESS_SUMMARIES)
        emb = encode([HARNESS_SUMMARIES[s] for s in slugs])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, scaffold=np.array(slugs), embeddings=emb)
        table = dict(zip(slugs, emb))
    mat = np.stack([table[name] for name in names]).astype(np.float32)
    return torch.as_tensor(mat, device=DEVICE)


def behavioral_features(
    train_acc: np.ndarray, train_emb: np.ndarray, codes: np.ndarray
) -> torch.Tensor:
    """Per component: mean train-query embedding weighted by (2*success - 1),
    plus its success rate and log1p(number of routes)."""
    hot = np.eye(codes.max() + 1)[codes]
    pooled = train_acc @ hot / hot.sum(axis=0)
    signed = 2.0 * pooled - 1.0
    feats = signed.T @ train_emb / len(train_emb)
    rate = pooled.mean(axis=0)[:, None]
    support = np.log1p(hot.sum(axis=0))[:, None]
    out = np.hstack([feats, rate, support])
    return torch.as_tensor(out.astype(np.float32), device=DEVICE)


class HMRouter(nn.Module):
    """Model rep = [id embedding ; behavioral features]; harness rep adds the
    summary text embedding."""

    def __init__(self, q_dim: int, n_models: int, n_harnesses: int,
                 b_m: torch.Tensor, b_h: torch.Tensor,
                 t_h: torch.Tensor):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, D_Q)
        self.emb_m = nn.Embedding(n_models, D_E)
        self.emb_h = nn.Embedding(n_harnesses, D_E)
        self.register_buffer("b_m", b_m)
        self.register_buffer("b_h", b_h)
        self.bproj_m = nn.Linear(q_dim + 2, D_E)
        self.bproj_h = nn.Linear(q_dim + 2, D_E)
        self.register_buffer("t_h", t_h)
        self.tproj_h = nn.Linear(t_h.shape[1], D_E)
        self.uproj_m = nn.Linear(2 * D_E, D_Q)
        self.uproj_h = nn.Linear(3 * D_E, D_Q)
        self.ln1 = nn.LayerNorm(D_Q)
        self.ln2 = nn.LayerNorm(D_Q)
        self.lng = nn.LayerNorm(D_Q)
        self.out = nn.Linear(3 * D_Q, 1)

    def forward(self, q_vec, m_ids, h_ids):
        q = self.q_proj(q_vec)
        em = torch.cat([
            self.emb_m(m_ids),
            self.bproj_m(self.b_m[m_ids]),
        ], -1)
        eh = torch.cat([
            self.emb_h(h_ids),
            self.bproj_h(self.b_h[h_ids]),
            self.tproj_h(self.t_h[h_ids]),
        ], -1)
        u_m = self.uproj_m(em)
        u_h = self.uproj_h(eh)
        h1 = self.ln1(u_m * q)
        h2 = self.ln2(u_h * q)
        hg = self.lng(u_m * u_h * q)
        return self.out(torch.cat([h1, h2, hg], -1)).squeeze(-1)


def _train(model, opt, loss_fn, q_all, q_idx, r_idx, targets, m_ids, h_ids):
    order = np.arange(len(q_idx))
    model.train()
    for _ in range(EPOCHS):
        np.random.shuffle(order)
        for start in range(0, len(order), BATCH):
            sel = torch.as_tensor(order[start : start + BATCH], device=DEVICE)
            q_in = q_all[q_idx[sel]]
            q_in = q_in + NOISE_STD * torch.randn_like(q_in)
            logits = model(q_in, m_ids[r_idx[sel]], h_ids[r_idx[sel]])
            loss = loss_fn(logits, targets[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()


def hm_router_select(
    features: np.ndarray,
    accuracy: np.ndarray,
    routes: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train hm-router on the train rows; return route picks for test rows."""
    m_codes, h_codes, h_names = route_factors(routes)
    t_h = summary_text_matrix(h_names)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m_ids = torch.as_tensor(m_codes, device=DEVICE)
    h_ids = torch.as_tensor(h_codes, device=DEVICE)
    q_all = torch.as_tensor(np.asarray(features, dtype=np.float32), device=DEVICE)

    emb64 = np.asarray(features, dtype=np.float64)
    b_m = behavioral_features(accuracy[train], emb64[train], m_codes)
    b_h = behavioral_features(accuracy[train], emb64[train], h_codes)

    # one training example per (train query, route) cell
    n, r = len(train), len(routes)
    rows = np.repeat(np.arange(n), r)
    cols = np.tile(np.arange(r), n)
    targets = torch.as_tensor(
        accuracy[train][rows, cols].astype(np.float32), device=DEVICE
    )
    q_idx = torch.as_tensor(train[rows], device=DEVICE)
    cols_t = torch.as_tensor(cols, device=DEVICE)

    torch.manual_seed(seed)
    model = HMRouter(
        q_all.shape[1], int(m_codes.max()) + 1, int(h_codes.max()) + 1,
        b_m, b_h, t_h
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    _train(model, opt, nn.BCEWithLogitsLoss(), q_all, q_idx, cols_t, targets,
           m_ids, h_ids)

    model.eval()
    with torch.no_grad():
        q_test = q_all[torch.as_tensor(test, device=DEVICE)]
        scores = model(
            q_test[:, None, :].expand(-1, r, -1).reshape(-1, q_all.shape[1]),
            m_ids.repeat(len(test)),
            h_ids.repeat(len(test)),
        ).reshape(len(test), r)
    return scores.cpu().numpy().argmax(axis=1)


# Harness text summaries, keyed by the `scaffold` column of the benchmark.


def h(kind, interface, tools, loop, autonomy, special):
    return (f"{kind}. Interface: {interface}. Tools: {tools}. "
            f"Control loop: {loop}. Autonomy: {autonomy}. "
            f"Specialization: {special}.")


HARNESS_SUMMARIES = {
    "claudecode": h(
        "Claude Code, Anthropic's terminal coding agent",
        "command-line agentic session in a real shell workspace",
        "bash execution, file read/search/edit, automatic repository context gathering",
        "iterative propose-execute-observe with long multi-step sessions",
        "high - runs autonomously until task completion",
        "software engineering and general computer tasks from the terminal"),
    "codex": h(
        "Codex CLI, OpenAI's open-source terminal coding agent",
        "command-line agentic session in a sandboxed local workspace",
        "shell commands and file edits",
        "iterative propose-and-execute until the task is done",
        "high - autonomous multi-step execution",
        "software engineering from the terminal"),
    "geminicli": h(
        "Gemini CLI, Google's open-source terminal agent",
        "command-line ReAct-style session",
        "built-in shell, file-system and web tools",
        "reason-act alternation",
        "high - autonomous multi-step execution",
        "coding and automation tasks from the command line"),
    "terminus2": h(
        "Terminus 2, a minimal terminal-native agent harness",
        "raw tmux terminal driven by emitted keystrokes and rendered screen reading",
        "nothing beyond the raw shell - no structured tools",
        "keystroke-observe loop on terminal state",
        "high but unassisted - no specialized affordances",
        "generic terminal task execution"),
    "hal_generalist": h(
        "HAL Generalist Agent, the default scaffold of the HAL evaluation harness",
        "plain ReAct tool-use loop",
        "bash, Python execution, file inspection",
        "reason-act alternation without task-specific prompting",
        "moderate - simple generic loop, one agent for every task type",
        "none - deliberately generic"),
    "taubench_toolcalling": h(
        "Native function-calling agent loop",
        "the model's structured tool-calling API over environment actions",
        "domain API functions exposed as callable tools",
        "alternate model calls and tool executions until a final answer is committed",
        "moderate - conversational tool use",
        "API-mediated interactive tasks"),
    "taubench_fewshot": h(
        "Few-shot prompted agent loop",
        "structured tool-calling with worked example trajectories in the prompt",
        "domain API functions exposed as callable tools",
        "alternate model calls and tool executions, guided by demonstrations",
        "moderate - conversational tool use",
        "API-mediated interactive tasks"),
    "swe": h(
        "SWE-agent, an agent-computer interface for repository-level software editing",
        "specialized ACI commands rather than a raw shell",
        "file viewer, search, in-place editing commands with lint feedback after each edit",
        "iterative edit-lint-observe",
        "high - autonomous repository navigation and editing",
        "software engineering on existing codebases"),
    "openhands": h(
        "OpenHands (formerly OpenDevin), an event-stream coding agent",
        "sandboxed runtime with multiple modalities",
        "bash shell, interactive Python/Jupyter, code editor, optional browser",
        "event-stream action-observation cycle",
        "high - designed for long autonomous sessions",
        "software engineering with rich runtime feedback"),
    "tools": h(
        "A minimal reference scaffold",
        "plain agentic loop with two tools only",
        "one bash tool and one file-edit tool",
        "iterative tool calls until done",
        "high but minimal affordances",
        "none - deliberately bare-bones"),
    "core": h(
        "CORE-Agent, an AutoGPT-derived task agent",
        "plan-and-execute agent over a provided file workspace",
        "shell and Python execution",
        "explicit planning then stepwise execution",
        "high - autonomous plan refinement",
        "computational and data-analysis tasks in a workspace"),
    "sabselfdebug": h(
        "Self-debug program-synthesis scaffold",
        "single-program generation with execution feedback",
        "program execution and error traces only",
        "write complete program, run it, revise from errors until it runs",
        "low - no free-form tool use or exploration",
        "one-shot program synthesis with iterative repair"),
    "hfopendeepresearch": h(
        "Hugging Face Open Deep Research, a smolagents-based web-research agent",
        "code-as-plan agent - plans are written as executable code",
        "web search, page browsing, file inspection",
        "plan-in-code then execute, multi-hop",
        "high - autonomous multi-hop information gathering",
        "web research and information synthesis"),
    "browseruse": h(
        "Browser-Use, a web-browsing agent",
        "real browser via structured DOM extraction",
        "click, type, scroll, navigate actions with page-state observation",
        "act-observe on DOM state each step",
        "high - autonomous web navigation",
        "web tasks on live sites via DOM"),
    "seeact": h(
        "SeeAct, a vision-grounded web agent",
        "rendered page screenshots as observations",
        "browser actions grounded by visual element selection",
        "see-reason-act on screenshots",
        "high - autonomous web navigation",
        "web tasks on live sites via vision"),
    "hermes": h(
        "Hermes, a lightweight autonomous agent harness",
        "shell-centric local tool loop",
        "shell, file operations, messaging",
        "iterative tool calls",
        "high - general computer use",
        "general local computer tasks"),
    "moltislocal": h(
        "Moltis (local deployment), a self-hosted personal agent harness",
        "local tool loop on the user's machine",
        "shell, files, service integrations",
        "iterative tool calls",
        "high - personal assistant style",
        "general local computer tasks"),
    "nanobot": h(
        "Nanobot, a minimal MCP-based agent framework",
        "Model Context Protocol servers wiring model to tools",
        "shell, file and service tools via MCP",
        "thin orchestration over MCP tool calls",
        "high - general computer use",
        "general local computer tasks via MCP"),
    "nullclaw": h(
        "NullClaw, a minimal dependency-light OpenClaw reimplementation",
        "small local runtime",
        "shell and messaging tools",
        "iterative tool calls",
        "high - personal assistant style",
        "general local computer tasks"),
    "zeroclawlocal": h(
        "ZeroClaw (local deployment), a lightweight Rust OpenClaw reimplementation",
        "local runtime with full tool loop",
        "shell, files, messaging",
        "iterative tool calls",
        "high - personal assistant style",
        "general local computer tasks"),
    "openclawlocal": h(
        "OpenClaw (local deployment), the open-source personal AI assistant agent",
        "local sandbox",
        "standard shell, file and automation tools",
        "iterative tool calls with persistent session",
        "high - personal assistant style",
        "general local computer tasks"),
    "openclaw": h(
        "OpenClaw, the open-source personal AI assistant agent",
        "persistent session on the user's machine",
        "shell access, file tools, messaging integrations",
        "iterative tool calls with persistent session",
        "high - personal assistant style",
        "acting on a user's machine and accounts"),
    "opencode": h(
        "OpenCode, an open-source terminal coding agent with a TUI",
        "terminal UI agentic session in the repository",
        "shell commands and file edits",
        "iterative propose-execute",
        "high - autonomous multi-step execution",
        "software engineering from the terminal"),
    "scicodezeroshot": h(
        "Zero-shot direct prompting scaffold",
        "single completion, no environment",
        "none - no tool use or execution feedback",
        "one-shot generation",
        "none - single response",
        "pure generation without iteration"),
    "deep": h(
        "DeepAgent, an autonomous multi-step agent harness",
        "planning agent with sub-task decomposition",
        "shell and file tools",
        "plan, decompose, execute sub-tasks end to end",
        "high - explicit planning and decomposition",
        "workspace tasks end to end"),
}
