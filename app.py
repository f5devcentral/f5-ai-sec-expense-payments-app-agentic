from __future__ import annotations

import html
import json
import os
import re

import streamlit as st
from dotenv import load_dotenv

from agents import (
    COMPLIANCE_SYSTEM_PROMPT,
    CREATE_PAYMENT_TOOL,
    INTAKE_SYSTEM_PROMPT,
    LOOKUP_EXPENSE_POLICY_TOOL,
    PAYMENT_SYSTEM_PROMPT,
)
from pipeline import PipelineEvent, PipelineRun, run_pipeline
from scenarios import SCENARIOS, scenario_by_key

load_dotenv()

DEFAULT_F5_AI_SECURITY_BASE_URL = "https://us1.calypsoai.app/"

st.set_page_config(
    page_title="Multi-Agent Prompt Injection Demo",
    page_icon="💸",
    layout="wide",
)

LIGHT_MODE = bool(st.session_state.get("light_mode", False))


STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;650;750;850&display=swap');
:root {
  --ll-bg: #100f0d;
  --ll-bg-soft: #17150f;
  --ll-panel: #1d1a14;
  --ll-panel-2: #242015;
  --ll-border: #3a3529;
  --ll-border-bright: #2dd4bf;
  --ll-text: #f2efe7;
  --ll-muted: #b3aa96;
  --ll-cyan: #2dd4bf;
  --ll-blue: #34d399;
  --ll-green: #65d48f;
  --ll-amber: #e0a84f;
  --ll-red: #ff7b72;
  --ll-shadow: 0 18px 45px rgba(0, 0, 0, .28);
}
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 18% 0%, rgba(45, 212, 191, .10), transparent 30%),
    radial-gradient(circle at 82% 8%, rgba(224, 168, 79, .10), transparent 28%),
    linear-gradient(180deg, #100f0d 0%, #15130f 55%, #0f0e0c 100%);
  color: var(--ll-text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1480px; padding-top: 1.25rem; padding-bottom: 3rem; }
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: var(--ll-text); letter-spacing: 0; }
.stMarkdown, .stText, p, label, .stCaptionContainer { color: var(--ll-muted); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #19170f 0%, #100f0d 100%);
  border-right: 1px solid var(--ll-border);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label { color: var(--ll-text); }
[data-testid="stSidebar"] .stCaptionContainer { color: var(--ll-muted); }
.stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea {
  background: #17150f;
  border-color: #413a2d;
  color: var(--ll-text);
}
.stButton > button {
  border-radius: 8px;
  border: 1px solid #413a2d;
  background: #242015;
  color: var(--ll-text);
  font-weight: 750;
  min-height: 2.75rem;
  box-shadow: none;
}
.stButton > button:hover {
  border-color: var(--ll-blue);
  background: #2b2619;
  color: #ffffff;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0f766e, #d49436);
  border-color: #36c7ae;
  color: #fffaf0;
}
.brand-shell {
  border: 1px solid var(--ll-border);
  border-radius: 10px;
  background: linear-gradient(180deg, rgba(29, 26, 20, .96), rgba(19, 17, 13, .96));
  box-shadow: var(--ll-shadow);
  margin-bottom: 1rem;
  overflow: hidden;
}
.brand-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: .85rem 1rem;
  border-bottom: 1px solid var(--ll-border);
}
.brand-left { display: flex; align-items: center; gap: .8rem; min-width: 0; }
.brand-mark {
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(145deg, rgba(45, 212, 191, .18), rgba(224, 168, 79, .10));
  border: 1px solid rgba(45, 212, 191, .46);
  color: var(--ll-blue);
}
.brand-name { color: #fffaf0; font-weight: 850; font-size: 1.22rem; line-height: 1; }
.brand-tagline { color: var(--ll-muted); font-size: .78rem; margin-top: .18rem; }
.brand-status {
  display: flex;
  align-items: center;
  gap: .5rem;
  color: #bdebd5;
  font-weight: 750;
  font-size: .82rem;
  white-space: nowrap;
}
.status-dot { width: .55rem; height: .55rem; border-radius: 50%; background: var(--ll-green); box-shadow: 0 0 0 4px rgba(66, 211, 146, .12); }
.hero-band {
  display: grid;
  grid-template-columns: 1.35fr .9fr;
  gap: 1rem;
  padding: 1.35rem 1rem 1rem;
}
.top-title { font-size: 2.3rem; font-weight: 850; letter-spacing: 0; margin: 0 0 .35rem; color: var(--ll-text); line-height: 1.04; }
.subtitle { color: #c9c0ad; font-size: 1rem; margin-bottom: 0; line-height: 1.45; }
.metric-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .7rem; }
.metric-card {
  border: 1px solid var(--ll-border);
  border-radius: 8px;
  background: rgba(16, 15, 13, .56);
  padding: .8rem;
}
.metric-label { color: var(--ll-muted); font-size: .72rem; font-weight: 750; text-transform: uppercase; letter-spacing: .03em; }
.metric-value { color: var(--ll-text); font-size: 1.25rem; font-weight: 850; margin-top: .2rem; }
.metric-good { color: var(--ll-green); }
.metric-watch { color: var(--ll-amber); }
.metric-risk { color: var(--ll-red); }
.timeline-card {
  border: 1px solid var(--ll-border);
  border-left: 6px solid var(--ll-blue);
  border-radius: 8px;
  padding: 1rem 1.1rem;
  margin: 0 0 1rem 0;
  background: rgba(29, 26, 20, .92);
  box-shadow: 0 10px 30px rgba(0, 0, 0, .18);
}
.telemetry-wrap {
  border: 1px solid var(--ll-border);
  border-radius: 8px;
  background: rgba(29, 26, 20, .92);
  padding: .75rem;
  margin: 0 0 1.25rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, .16);
}
.telemetry-row {
  display: grid;
  grid-template-columns: 138px 112px 1fr;
  gap: .75rem;
  align-items: start;
  border-bottom: 1px solid rgba(216, 222, 232, .10);
  padding: .68rem .35rem;
}
.telemetry-row:last-child { border-bottom: 0; }
.telemetry-actor {
  font-size: .78rem;
  font-weight: 820;
  color: var(--ll-text);
}
.telemetry-type {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: .7rem;
  color: var(--ll-muted);
}
.telemetry-message {
  font-size: .86rem;
  line-height: 1.35;
  color: #ebe5d8;
}
.telemetry-detail {
  margin-top: .2rem;
  font-size: .75rem;
  line-height: 1.3;
  color: var(--ll-muted);
  overflow-wrap: anywhere;
}
.severity-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 72px;
  border-radius: 999px;
  padding: .16rem .45rem;
  font-size: .68rem;
  font-weight: 820;
  text-transform: uppercase;
  letter-spacing: .02em;
}
.sev-info { background: rgba(45, 212, 191, .14); color: #9af5e8; border: 1px solid rgba(45, 212, 191, .28); }
.sev-warning { background: rgba(245, 184, 75, .16); color: #ffd990; border: 1px solid rgba(245, 184, 75, .32); }
.sev-danger { background: rgba(255, 107, 107, .18); color: #ffb0b0; border: 1px solid rgba(255, 107, 107, .35); }
.sev-blocked { background: rgba(255, 107, 107, .24); color: #ffd0d0; border: 1px solid rgba(255, 107, 107, .45); }
.agent-title { font-size: 1.15rem; font-weight: 800; color: var(--ll-text); margin-bottom: .35rem; }
.field-label { font-size: .72rem; text-transform: uppercase; font-weight: 800; color: var(--ll-muted); margin-top: .75rem; letter-spacing: .03em; }
.mono-box {
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: .86rem;
  background: #11100d;
  border: 1px solid #3a3529;
  border-radius: 6px;
  padding: .75rem;
  color: #f0eadc;
  overflow-wrap: anywhere;
}
.reasoning { color: #e6decd; font-size: .95rem; line-height: 1.45; }
.inj { color: #b91c1c; background: #fee2e2; font-weight: 760; padding: 0 .15rem; border-radius: 3px; }
.badge {
  display: inline-block;
  padding: .22rem .5rem;
  border-radius: 999px;
  font-size: .78rem;
  font-weight: 760;
  margin-right: .35rem;
}
.badge-allowed { background: rgba(66, 211, 146, .16); color: #9ff0c9; border: 1px solid rgba(66, 211, 146, .35); }
.badge-blocked { background: rgba(255, 107, 107, .18); color: #ffb0b0; border: 1px solid rgba(255, 107, 107, .35); }
.decision { background: rgba(45, 212, 191, .12); color: #a6f4e8; border: 1px solid rgba(45, 212, 191, .30); }
.banner {
  border-radius: 8px;
  padding: 1rem 1.1rem;
  font-size: 1.25rem;
  font-weight: 800;
  margin: 1rem 0 1.25rem;
}
.paid { background: rgba(66, 211, 146, .16); color: #b6f4d2; border: 1px solid rgba(66, 211, 146, .55); }
.blocked { background: rgba(255, 107, 107, .18); color: #ffd0d0; border: 1px solid rgba(255, 107, 107, .45); }
.approved { background: rgba(45, 212, 191, .14); color: #b8f7ed; border: 1px solid rgba(45, 212, 191, .50); }
.flow-wrap {
  border: 1px solid var(--ll-border);
  border-radius: 8px;
  background: linear-gradient(180deg, rgba(29, 26, 20, .96) 0%, rgba(16, 15, 13, .96) 100%);
  padding: 1rem 1rem .85rem;
  margin: .5rem 0 1.25rem;
  box-shadow: var(--ll-shadow);
}
.flow-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: .85rem;
}
.flow-heading {
  font-size: .95rem;
  font-weight: 830;
  color: var(--ll-text);
}
.flow-caption {
  margin-top: .16rem;
  color: var(--ll-muted);
  font-size: .82rem;
  line-height: 1.35;
}
.flow-legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: .35rem;
  min-width: 250px;
}
.legend-pill {
  border: 1px solid var(--ll-border);
  border-radius: 999px;
  padding: .2rem .46rem;
  background: #17150f;
  color: #d6ccb8;
  font-size: .72rem;
  font-weight: 740;
  white-space: nowrap;
}
.flow-rail {
  display: flex;
  align-items: stretch;
  gap: 0;
  overflow-x: auto;
  padding: .15rem .1rem .7rem;
  scrollbar-color: #cbd5e1 transparent;
}
.flow-node {
  position: relative;
  flex: 1 0 132px;
  min-width: 132px;
  max-width: 170px;
  min-height: 142px;
  border: 1px solid #413a2d;
  border-radius: 8px;
  padding: .72rem .72rem .68rem;
  background: #1b1812;
  transition: border-color .18s ease, background .18s ease, box-shadow .18s ease, transform .18s ease;
}
.flow-node.agent { border-top: 4px solid var(--ll-blue); }
.flow-node.boundary { border-top: 4px solid var(--ll-amber); background: rgba(245, 184, 75, .08); border-style: dashed; }
.flow-node.outcome { border-top: 4px solid var(--ll-green); }
.flow-node.running {
  border-color: var(--ll-cyan);
  background: rgba(45, 212, 191, .09);
  box-shadow: 0 0 0 3px rgba(45, 212, 191, .13), 0 12px 22px rgba(13, 148, 136, .12);
  transform: translateY(-1px);
}
.flow-node.complete { border-color: rgba(66, 211, 146, .58); background: rgba(66, 211, 146, .08); }
.flow-node.blocked { border-color: rgba(255, 107, 107, .65); background: rgba(255, 107, 107, .10); box-shadow: 0 0 0 3px rgba(255, 107, 107, .12); }
.flow-node.skipped { border-color: #4a4335; background: #17150f; opacity: .72; }
.flow-node.waiting { color: var(--ll-muted); }
.flow-step {
  width: 1.55rem;
  height: 1.55rem;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #115e59;
  color: #fffaf0;
  font-size: .72rem;
  font-weight: 820;
  margin-bottom: .55rem;
}
.flow-node.boundary .flow-step { background: #8a5a0a; }
.flow-node.outcome .flow-step { background: #23724b; }
.flow-title { font-size: .88rem; font-weight: 850; color: var(--ll-text); margin-bottom: .12rem; line-height: 1.18; }
.flow-subtitle { font-size: .73rem; color: var(--ll-muted); line-height: 1.24; min-height: 2.2em; }
.flow-status {
  margin-top: .6rem;
  display: inline-flex;
  align-items: center;
  gap: .32rem;
  font-size: .67rem;
  font-weight: 820;
  text-transform: uppercase;
  color: #d4cbb7;
  letter-spacing: .02em;
}
.flow-dot {
  width: .48rem;
  height: .48rem;
  border-radius: 999px;
  background: #8b806c;
}
.flow-node.running .flow-dot { background: var(--ll-cyan); }
.flow-node.complete .flow-dot { background: var(--ll-green); }
.flow-node.blocked .flow-dot { background: var(--ll-red); }
.flow-node.skipped .flow-dot { background: #9a907d; }
.flow-detail {
  margin-top: .28rem;
  font-size: .72rem;
  color: #bdb39f;
  line-height: 1.26;
  overflow-wrap: anywhere;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.flow-connector {
  flex: 0 0 34px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 142px;
}
.flow-connector::before {
  content: "";
  height: 2px;
  flex: 1;
  background: #5a5140;
}
.flow-connector::after {
  content: "";
  width: 0;
  height: 0;
  border-top: 5px solid transparent;
  border-bottom: 5px solid transparent;
  border-left: 7px solid #8b806c;
}
.flow-note {
  border-top: 1px solid var(--ll-border);
  padding-top: .7rem;
  color: var(--ll-muted);
  font-size: .84rem;
  line-height: 1.35;
}
.agent-inspector {
  border: 1px solid var(--ll-border);
  border-radius: 8px;
  background: rgba(29, 26, 20, .94);
  padding: 1rem;
  margin: -.45rem 0 1.25rem;
  box-shadow: var(--ll-shadow);
}
.inspector-head { display: flex; align-items: center; gap: .8rem; margin-bottom: .8rem; }
.inspector-icon {
  width: 2.75rem;
  height: 2.75rem;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--ll-blue);
  background: rgba(45, 212, 191, .10);
  border: 1px solid rgba(45, 212, 191, .32);
}
.inspector-title {
  font-size: 1rem;
  font-weight: 850;
  color: var(--ll-text);
  margin-bottom: .25rem;
}
.inspector-subtitle {
  color: var(--ll-muted);
  font-size: .84rem;
  line-height: 1.35;
}
.inspector-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .65rem;
  margin-bottom: .85rem;
}
.inspector-tile {
  border: 1px solid var(--ll-border);
  border-radius: 8px;
  background: #17150f;
  padding: .65rem;
  min-height: 76px;
}
.inspector-label {
  font-size: .68rem;
  font-weight: 820;
  color: var(--ll-muted);
  text-transform: uppercase;
  letter-spacing: .02em;
  margin-bottom: .18rem;
}
.inspector-value {
  font-size: .82rem;
  line-height: 1.3;
  color: #e8dfcf;
  overflow-wrap: anywhere;
}
.detail-panel {
  border: 1px solid var(--ll-border);
  border-radius: 8px;
  background: rgba(16, 15, 13, .76);
  padding: .85rem;
  min-height: 100%;
}
.detail-panel-title {
  display: flex;
  align-items: center;
  gap: .5rem;
  color: var(--ll-text);
  font-weight: 850;
  font-size: .9rem;
  margin-bottom: .5rem;
}
.code-panel {
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: .76rem;
  line-height: 1.45;
  color: #e2dacb;
  background: #0c0b09;
  border: 1px solid #352f24;
  border-radius: 8px;
  padding: .8rem;
  max-height: 360px;
  overflow: auto;
}
.tool-row {
  display: grid;
  grid-template-columns: 2rem 1fr;
  gap: .55rem;
  align-items: start;
  padding: .65rem 0;
  border-bottom: 1px solid rgba(232, 224, 208, .10);
}
.tool-row:last-child { border-bottom: 0; }
.tool-glyph {
  width: 2rem;
  height: 2rem;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: rgba(45, 212, 191, .10);
  color: var(--ll-cyan);
  border: 1px solid rgba(45, 212, 191, .26);
}
.tool-name { color: var(--ll-text); font-weight: 820; font-size: .86rem; }
.tool-desc { color: var(--ll-muted); font-size: .78rem; line-height: 1.35; margin-top: .12rem; }
.boundary-callout {
  position: relative;
  border: 1px solid rgba(224, 168, 79, .46);
  border-left: 5px solid var(--ll-amber);
  background: linear-gradient(180deg, rgba(90, 65, 26, .72), rgba(45, 36, 20, .90));
  color: #fff1c9;
  border-radius: 8px;
  padding: .95rem 1rem;
  line-height: 1.45;
  font-size: .92rem;
  font-weight: 650;
  box-shadow: inset 0 1px 0 rgba(255, 232, 185, .08), 0 12px 28px rgba(0, 0, 0, .18);
}
.boundary-callout::before {
  content: "Boundary risk";
  display: block;
  color: #ffd27a;
  font-size: .68rem;
  font-weight: 850;
  letter-spacing: .04em;
  text-transform: uppercase;
  margin-bottom: .3rem;
}
.section-title {
  display: flex;
  align-items: center;
  gap: .5rem;
  color: var(--ll-text);
  font-weight: 850;
  margin: 1.15rem 0 .55rem;
}
.sidebar-brand {
  border: 1px solid var(--ll-border);
  border-radius: 10px;
  background: rgba(29, 26, 20, .86);
  padding: .85rem;
  margin: .35rem 0 1rem;
}
.sidebar-brand-row { display: flex; align-items: center; gap: .65rem; }
.sidebar-brand-mark {
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--ll-blue);
  border: 1px solid rgba(45, 212, 191, .46);
  background: rgba(45, 212, 191, .12);
}
.sidebar-brand-name { color: var(--ll-text); font-size: 1.05rem; font-weight: 850; line-height: 1; }
.sidebar-brand-sub { color: var(--ll-muted); font-size: .72rem; margin-top: .18rem; }
.sidebar-status {
  display: flex;
  align-items: center;
  gap: .45rem;
  color: var(--ll-green);
  font-size: .76rem;
  font-weight: 780;
  margin-top: .7rem;
  padding-top: .65rem;
  border-top: 1px solid rgba(232, 224, 208, .12);
}
.theme-row {
  display: flex;
  justify-content: flex-end;
  margin: 0 0 .45rem;
}
.ops-panel {
  border: 1px solid var(--ll-border);
  border-radius: 10px;
  background: rgba(29, 26, 20, .94);
  box-shadow: var(--ll-shadow);
  padding: 1rem;
  margin: .5rem 0 1.25rem;
}
.ops-panel .telemetry-wrap {
  background: transparent;
  border: 0;
  box-shadow: none;
  padding: 0;
  margin: 0;
}
.ops-panel .telemetry-row {
  display: grid;
  grid-template-columns: 1fr;
  gap: .45rem;
  border: 1px solid rgba(232, 224, 208, .12);
  border-radius: 8px;
  padding: .8rem;
  margin-bottom: .65rem;
  background: rgba(16, 15, 13, .48);
}
.ops-panel .telemetry-row:last-child { margin-bottom: 0; border-bottom: 1px solid rgba(232, 224, 208, .12); }
.ops-panel .telemetry-actor { font-size: .84rem; }
.ops-panel .telemetry-type { margin-top: .15rem; }
.ops-panel .telemetry-message {
  font-size: .9rem;
  line-height: 1.42;
  overflow-wrap: normal;
  word-break: normal;
}
.ops-panel .telemetry-detail {
  font-size: .78rem;
  line-height: 1.4;
  overflow-wrap: anywhere;
}
.ops-panel .severity-pill {
  justify-content: flex-start;
  width: fit-content;
  min-width: 0;
  padding: .2rem .65rem;
}
.ops-title {
  display: flex;
  align-items: center;
  gap: .5rem;
  color: var(--ll-text);
  font-size: 1rem;
  font-weight: 850;
  margin-bottom: .2rem;
}
.ops-subtitle {
  color: var(--ll-muted);
  font-size: .8rem;
  line-height: 1.35;
  margin-bottom: .85rem;
}
.ops-section {
  border-top: 1px solid rgba(232, 224, 208, .12);
  padding-top: .85rem;
  margin-top: .85rem;
}
.ops-section:first-of-type {
  border-top: 0;
  padding-top: 0;
  margin-top: 0;
}
.ops-section-title {
  display: flex;
  align-items: center;
  gap: .45rem;
  color: var(--ll-text);
  font-size: .82rem;
  font-weight: 850;
  text-transform: uppercase;
  letter-spacing: .03em;
  margin-bottom: .5rem;
}
@media (max-width: 900px) {
  .flow-head { flex-direction: column; }
  .flow-legend { justify-content: flex-start; min-width: 0; }
  .flow-node { flex-basis: 150px; }
  .telemetry-row { grid-template-columns: 1fr; gap: .35rem; }
  .inspector-grid { grid-template-columns: 1fr; }
  .hero-band { grid-template-columns: 1fr; }
  .metric-strip { grid-template-columns: 1fr; }
}
</style>
"""

LIGHT_THEME_STYLE = """
<style>
:root {
  --ll-bg: #f4f8fb;
  --ll-bg-soft: #eef5fa;
  --ll-panel: #ffffff;
  --ll-panel-2: #f7fbff;
  --ll-border: #cfdbe7;
  --ll-border-bright: #0b7fd3;
  --ll-text: #172535;
  --ll-muted: #5b6f82;
  --ll-cyan: #0891b2;
  --ll-blue: #0b7fd3;
  --ll-green: #16845b;
  --ll-amber: #b7791f;
  --ll-red: #c24141;
  --ll-shadow: 0 18px 38px rgba(24, 43, 66, .10);
}
.stApp {
  background:
    radial-gradient(circle at 18% 0%, rgba(8, 145, 178, .10), transparent 30%),
    radial-gradient(circle at 82% 8%, rgba(11, 127, 211, .10), transparent 28%),
    linear-gradient(180deg, #f7fbff 0%, #edf5fb 55%, #f7fbff 100%);
  color: var(--ll-text);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #edf5fb 100%);
  border-right: 1px solid var(--ll-border);
}
.brand-shell, .flow-wrap, .agent-inspector, .timeline-card, .telemetry-wrap, .ops-panel {
  background: rgba(255, 255, 255, .94);
}
.brand-name, .top-title, .flow-heading, .flow-title, .agent-title, .inspector-title,
.detail-panel-title, .section-title, .tool-name, .telemetry-actor, .sidebar-brand-name {
  color: var(--ll-text);
}
.subtitle, .flow-caption, .flow-subtitle, .flow-note, .inspector-subtitle, .inspector-label,
.tool-desc, .telemetry-type, .telemetry-detail, .brand-tagline, .sidebar-brand-sub {
  color: var(--ll-muted);
}
.metric-card, .flow-node, .inspector-tile, .detail-panel, .legend-pill, .sidebar-brand {
  background: #f7fbff;
}
.flow-node.boundary { background: rgba(183, 121, 31, .08); }
.flow-node.complete { background: rgba(22, 132, 91, .08); }
.flow-node.blocked { background: rgba(194, 65, 65, .09); }
.mono-box, .code-panel {
  background: #f3f8fc;
  border-color: #c9d8e6;
  color: #172535;
}
.telemetry-message, .reasoning, .inspector-value { color: #213449; }
.boundary-callout {
  background: linear-gradient(180deg, #fff4d6, #ffe8ad);
  border-color: #d49b37;
  color: #402b0b;
  box-shadow: 0 12px 24px rgba(95, 63, 18, .10);
}
.boundary-callout::before {
  color: #8a5a0a;
}
.stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea {
  background: #ffffff;
  border-color: #c9d8e6;
  color: var(--ll-text);
}
.stButton > button {
  background: #f7fbff;
  border-color: #c9d8e6;
  color: var(--ll-text);
}
.stButton > button:hover {
  background: #eef6fc;
  color: var(--ll-text);
}
.ops-panel .telemetry-row {
  background: #f7fbff;
  border-color: #d8e4ef;
}
</style>
"""

st.markdown(STYLE + (LIGHT_THEME_STYLE if LIGHT_MODE else ""), unsafe_allow_html=True)


def highlight(text: str, fragments: tuple[str, ...]) -> str:
    escaped = html.escape(text or "")
    for fragment in sorted(fragments, key=len, reverse=True):
        if not fragment:
            continue
        escaped_fragment = html.escape(fragment)
        escaped = re.sub(
            re.escape(escaped_fragment),
            f'<span class="inj">{escaped_fragment}</span>',
            escaped,
            flags=re.IGNORECASE,
        )
    return escaped


def amount_label(amount: float | None) -> str:
    return "unknown amount" if amount is None else f"${amount:,.0f}"


def f5_ready() -> bool:
    base_url = (os.getenv("F5_AI_SECURITY_BASE_URL") or DEFAULT_F5_AI_SECURITY_BASE_URL).strip()
    token = (os.getenv("F5_AI_SECURITY_PROJECT_TOKEN") or os.getenv("CALYPSOAI_TOKEN") or "").strip()
    connection_name = (os.getenv("F5_AI_SECURITY_CONNECTION_NAME") or "").strip()
    has_openai_path = "/openai/" in base_url.rstrip("/") or base_url.rstrip("/").endswith("/openai")
    return bool(base_url and token and "CONNECTION-NAME" not in base_url and (has_openai_path or connection_name))


ICONS = {
    "shield": '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3z"/><path d="M9 12l2 2 4-5"/></svg>',
    "doc": '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 3h7l4 4v14H7z"/><path d="M14 3v5h5"/><path d="M9 13h6"/><path d="M9 17h6"/></svg>',
    "scale": '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v18"/><path d="M5 7h14"/><path d="M7 7l-4 7h8z"/><path d="M17 7l-4 7h8z"/></svg>',
    "card": '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18"/><path d="M7 15h4"/></svg>',
    "arrow": '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14"/><path d="M13 6l6 6-6 6"/></svg>',
    "tool": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a4 4 0 0 0-5 5L4 17v3h3l5.7-5.7a4 4 0 0 0 5-5l-2.4 2.4-3-3z"/></svg>',
    "code": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 9l-4 3 4 3"/><path d="M16 9l4 3-4 3"/><path d="M13 5l-2 14"/></svg>',
    "monitor": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8"/><path d="M12 17v4"/></svg>',
    "boundary": '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l7 4v5c0 4-3 7-7 9-4-2-7-5-7-9V7z"/><path d="M12 8v5"/><path d="M12 17h.01"/></svg>',
}


AGENT_ICONS = {
    "Intake Agent": ICONS["doc"],
    "Compliance Agent": ICONS["scale"],
    "Payment Agent": ICONS["card"],
}


def render_brand_header() -> None:
    guardrail_status = "F5 ready" if f5_ready() else "Local only"
    guardrail_class = "metric-good" if f5_ready() else "metric-watch"
    st.markdown(
        f"""
<div class="brand-shell">
  <div class="brand-topbar">
    <div class="brand-left">
      <div class="brand-mark">{ICONS["shield"]}</div>
      <div>
        <div class="brand-name">LedgerLoop</div>
        <div class="brand-tagline">Secure payments. Greater trust.</div>
      </div>
    </div>
    <div class="brand-status"><span class="status-dot"></span>Demo environment online</div>
  </div>
  <div class="hero-band">
    <div>
      <div class="top-title">Agentic Payments Control Room</div>
      <div class="subtitle">A live security demo showing how prompt injection can cross agent boundaries, trigger tools, and reach a simulated payment action.</div>
    </div>
    <div class="metric-strip">
      <div class="metric-card"><div class="metric-label">Pipeline</div><div class="metric-value">3 agents</div></div>
      <div class="metric-card"><div class="metric-label">Guardrails</div><div class="metric-value {guardrail_class}">{guardrail_status}</div></div>
      <div class="metric-card"><div class="metric-label">Tool surface</div><div class="metric-value metric-watch">2 actions</div></div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.markdown(
        f"""
<div class="sidebar-brand">
  <div class="sidebar-brand-row">
    <div class="sidebar-brand-mark">{ICONS["shield"]}</div>
    <div>
      <div class="sidebar-brand-name">LedgerLoop</div>
      <div class="sidebar-brand-sub">Agentic payment security</div>
    </div>
  </div>
  <div class="sidebar-status"><span class="status-dot"></span>Demo environment online</div>
</div>
""",
        unsafe_allow_html=True,
    )


FLOW_STAGES = (
    ("input", "Raw request", "Employee expense data"),
    ("intake", "Intake Agent", "Parses and structures"),
    ("guardrail_1", "Trust boundary", "Intake -> Compliance"),
    ("compliance", "Compliance Agent", "Policy decision"),
    ("guardrail_2", "Trust boundary", "Compliance -> Payment"),
    ("payment", "Payment Agent", "Trusts compliance handoff"),
    ("outcome", "Outcome", "Paid, blocked, denied, or escalated"),
)


FLOW_ICONS = {
    "input": ICONS["doc"],
    "intake": ICONS["doc"],
    "guardrail_1": ICONS["boundary"],
    "compliance": ICONS["scale"],
    "guardrail_2": ICONS["boundary"],
    "payment": ICONS["card"],
    "outcome": ICONS["shield"],
}


AGENT_DETAILS = {
    "Intake Agent": {
        "role": "Transforms raw employee expense text into structured expense data.",
        "received": "Raw request from the employee.",
        "passes": "Structured expense handoff to Compliance.",
        "tools": [],
        "system_prompt": INTAKE_SYSTEM_PROMPT,
        "trust_boundary": "Entry point for untrusted user text. It preserves descriptions and notes for downstream context.",
        "f5_view": "Appears as an agent run with system prompt, raw user request, and structured JSON response.",
    },
    "Compliance Agent": {
        "role": "Applies expense policy and decides APPROVE, DENY, or ESCALATE.",
        "received": "Structured message from Intake.",
        "passes": "Compliance decision and payment handoff.",
        "tools": [LOOKUP_EXPENSE_POLICY_TOOL],
        "system_prompt": COMPLIANCE_SYSTEM_PROMPT,
        "trust_boundary": "Treats Intake output as structured business data, even if malicious text survived inside fields.",
        "f5_view": "Shows the policy prompt, Intake handoff, lookup_expense_policy tool call, and final compliance response.",
    },
    "Payment Agent": {
        "role": "Trusts Compliance and simulates payout.",
        "received": "Compliance decision handoff.",
        "passes": "Final payment status.",
        "tools": [CREATE_PAYMENT_TOOL],
        "system_prompt": PAYMENT_SYSTEM_PROMPT,
        "trust_boundary": "Does not re-check policy. This is where a poisoned approval can become an action.",
        "f5_view": "Shows the payment prompt, Compliance handoff, create_payment tool call, and final payment response.",
    },
}


def empty_flow_state() -> dict[str, dict[str, str]]:
    return {
        stage: {"status": "waiting", "label": "Waiting", "detail": subtitle}
        for stage, _title, subtitle in FLOW_STAGES
    }


def render_flow_chart(state: dict[str, dict[str, str]], *, guardrails_enabled: bool) -> str:
    pieces: list[str] = [
        """
<div class="flow-wrap">
  <div class="flow-head">
    <div>
      <div class="flow-heading">Sequential agent workflow</div>
      <div class="flow-caption">Watch the handoff boundary where untrusted request text becomes trusted downstream context.</div>
    </div>
    <div class="flow-legend">
      <span class="legend-pill">Blue: agent</span>
      <span class="legend-pill">Amber: trust boundary</span>
      <span class="legend-pill">Red: blocked</span>
    </div>
  </div>
  <div class="flow-rail">
"""
    ]
    for index, (stage, title, subtitle) in enumerate(FLOW_STAGES, start=1):
        data = state.get(stage, {"status": "waiting", "label": "Waiting", "detail": subtitle})
        status = html.escape(data.get("status", "waiting"))
        label = html.escape(data.get("label", "Waiting"))
        detail = html.escape(compact_flow_detail(data.get("detail", subtitle)))
        kind = "boundary" if stage.startswith("guardrail") else "outcome" if stage == "outcome" else "agent"
        icon = FLOW_ICONS.get(stage, ICONS["doc"])
        pieces.append(
            f"""
<div class="flow-node {kind} {status}">
  <div class="flow-step">{icon}</div>
  <div class="flow-title">{html.escape(title)}</div>
  <div class="flow-subtitle">{html.escape(subtitle)}</div>
  <div class="flow-status"><span class="flow-dot"></span>{label}</div>
  <div class="flow-detail">{detail}</div>
</div>
"""
        )
        if index < len(FLOW_STAGES):
            pieces.append('<div class="flow-connector"></div>')
    guardrail_mode = "enabled at both trust boundaries" if guardrails_enabled else "off; messages cross boundaries uninspected"
    pieces.append(
        f'</div><div class="flow-note"><strong>Architecture:</strong> sequential agent handoffs with guardrails {guardrail_mode}. '
        "The attack path is visible when untrusted user text survives one agent and becomes trusted input to the next.</div></div>"
    )
    return "".join(pieces)


def compact_flow_detail(detail: str) -> str:
    if "F5_AI_SECURITY_BASE_URL" in detail or "F5_AI_SECURITY_PROJECT_TOKEN" in detail:
        return "F5 is not configured. Set F5 env vars or turn guardrails off for the vulnerable run."
    if len(detail) > 130:
        return f"{detail[:127]}..."
    return detail


def render_agent_popovers() -> None:
    st.markdown(
        f'<div class="section-title">{ICONS["monitor"]} Inspect agent configuration</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    for column, agent_name in zip(columns, AGENT_DETAILS, strict=True):
        details = AGENT_DETAILS[agent_name]
        with column:
            with st.popover(agent_name, use_container_width=True):
                render_agent_detail(agent_name, details)


def render_agent_detail(selected: str, details: dict) -> None:
    selected_icon = AGENT_ICONS[selected]
    tool_names = [
        tool.get("function", {}).get("name", "unnamed_tool")
        for tool in details["tools"]
    ]
    tool_value = ", ".join(tool_names) if tool_names else "No tools"

    st.markdown(
        f"""
<div class="agent-inspector">
  <div class="inspector-head">
    <div class="inspector-icon">{selected_icon}</div>
    <div>
      <div class="inspector-title">{html.escape(selected)} under the hood</div>
      <div class="inspector-subtitle">{html.escape(details["role"])}</div>
    </div>
  </div>
  <div class="inspector-grid">
    <div class="inspector-tile">
      <div class="inspector-label">Receives</div>
      <div class="inspector-value">{html.escape(details["received"])}</div>
    </div>
    <div class="inspector-tile">
      <div class="inspector-label">Passes onward</div>
      <div class="inspector-value">{html.escape(details["passes"])}</div>
    </div>
    <div class="inspector-tile">
      <div class="inspector-label">Tool access</div>
      <div class="inspector-value">{html.escape(tool_value)}</div>
    </div>
    <div class="inspector-tile">
      <div class="inspector-label">F5 session view</div>
      <div class="inspector-value">{html.escape(details["f5_view"])}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, .9])
    with left:
        st.markdown(
            f"""
<div class="detail-panel">
  <div class="detail-panel-title">{ICONS["code"]} System prompt</div>
  <div class="code-panel">{html.escape(details["system_prompt"].strip())}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        tool_rows = []
        if details["tools"]:
            for tool in details["tools"]:
                function = tool.get("function", {})
                tool_rows.append(
                    f"""
<div class="tool-row">
  <div class="tool-glyph">{ICONS["tool"]}</div>
  <div>
    <div class="tool-name">{html.escape(function.get("name", "Tool"))}</div>
    <div class="tool-desc">{html.escape(function.get("description", ""))}</div>
  </div>
</div>
"""
                )
            tool_rows.append(
                f'<div class="code-panel">{html.escape(json.dumps(details["tools"], indent=2))}</div>'
            )
        else:
            tool_rows.append(
                f"""
<div class="tool-row">
  <div class="tool-glyph">{ICONS["monitor"]}</div>
  <div>
    <div class="tool-name">No tool access</div>
    <div class="tool-desc">This agent only transforms text and emits the next handoff.</div>
  </div>
</div>
"""
            )
        st.markdown(
            f"""
<div class="detail-panel">
  <div class="detail-panel-title">{ICONS["tool"]} Tool access</div>
  {''.join(tool_rows)}
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="section-title">{ICONS["boundary"]} Trust boundary</div>
<div class="boundary-callout">{html.escape(details["trust_boundary"])}</div>
""",
            unsafe_allow_html=True,
        )


def apply_progress_event(state: dict[str, dict[str, str]], event: PipelineEvent) -> None:
    state[event.stage] = {
        "status": event.status,
        "label": event.label,
        "detail": event.detail,
    }


def render_banner(result: PipelineRun) -> None:
    if result.final_outcome == "BLOCKED_BY_GUARDRAIL":
        st.markdown(
            f'<div class="banner blocked">🛑 BLOCKED BY GUARDRAIL: {html.escape(result.blocked_reason or "")}</div>',
            unsafe_allow_html=True,
        )
    elif result.final_outcome == "PAID":
        st.markdown(
            f'<div class="banner paid">💸 PAID: {amount_label(result.final_amount_usd)}</div>',
            unsafe_allow_html=True,
        )
    elif result.final_outcome in {"APPROVED", "APPROVE"}:
        st.markdown(
            f'<div class="banner approved">✅ APPROVED: {amount_label(result.final_amount_usd)}</div>',
            unsafe_allow_html=True,
        )
    elif result.final_outcome == "ESCALATE":
        st.markdown(
            f'<div class="banner blocked">🛑 ESCALATED FOR MANAGER APPROVAL: {amount_label(result.final_amount_usd)}</div>',
            unsafe_allow_html=True,
        )
    elif result.final_outcome == "DENY":
        st.markdown(
            f'<div class="banner blocked">🛑 DENIED: {amount_label(result.final_amount_usd)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner blocked">🛑 {html.escape(result.final_outcome)}: {amount_label(result.final_amount_usd)}</div>',
            unsafe_allow_html=True,
        )


def render_step(step, fragments: tuple[str, ...]) -> None:
    verdict_html = ""
    if step.guardrail_after:
        if step.guardrail_after.allowed:
            verdict_html = (
                '<span class="badge badge-allowed">✅ ALLOWED</span>'
                f"{html.escape(step.guardrail_after.reason)}"
            )
        else:
            verdict_html = (
                '<span class="badge badge-blocked">🛑 BLOCKED</span>'
                f"{html.escape(step.guardrail_after.reason)}"
            )

    st.markdown(
        f"""
<div class="timeline-card">
  <div class="agent-title">{html.escape(step.agent)}</div>
  <span class="badge decision">{html.escape(step.decision)}</span>
  <div class="field-label">Message received</div>
  <div class="mono-box">{highlight(step.received, fragments)}</div>
  <div class="field-label">Reasoning / decision</div>
  <div class="reasoning">{html.escape(step.reasoning)}</div>
  <div class="field-label">Message passed onward</div>
  <div class="mono-box">{highlight(step.passed_message, fragments)}</div>
  {f'<div class="field-label">Guardrail verdict</div><div>{verdict_html}</div>' if verdict_html else ''}
</div>
""",
        unsafe_allow_html=True,
    )


def render_telemetry(result: PipelineRun) -> None:
    events = result.telemetry or []
    if not events:
        st.info("No telemetry was recorded for this run.")
        return

    rows = ['<div class="telemetry-wrap">']
    for event in events:
        severity = html.escape(event.severity)
        rows.append(
            f"""
<div class="telemetry-row">
  <div>
    <div class="telemetry-actor">{html.escape(event.actor)}</div>
    <div class="telemetry-type">{html.escape(event.event_type)}</div>
  </div>
  <div><span class="severity-pill sev-{severity}">{severity}</span></div>
  <div>
    <div class="telemetry-message">{html.escape(event.message)}</div>
    {f'<div class="telemetry-detail">{html.escape(event.detail)}</div>' if event.detail else ''}
  </div>
</div>
"""
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


_theme_left, _theme_right = st.columns([1, .18])
with _theme_right:
    st.toggle("Light mode", key="light_mode")

render_brand_header()

with st.sidebar:
    render_sidebar_brand()
    st.header("Run setup")
    selected_key = st.selectbox("Scenario", [s.key for s in SCENARIOS], format_func=lambda key: scenario_by_key(key).label)
    guardrails_enabled = st.toggle("GUARDRAILS ON/OFF", value=True)
    use_custom = st.checkbox("Use custom request")
    custom_request = st.text_area("Custom request", height=220, disabled=not use_custom)
    run_clicked = st.button("Run pipeline", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Preflight")
    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
    f5_configured = f5_ready()
    f5_agent_proxy = guardrails_enabled and f5_configured and os.getenv("F5_AI_SECURITY_PROXY_AGENT_CALLS", "true").strip().lower() not in {"0", "false", "no", "off"}
    st.caption(f"OpenAI API key: {'configured' if openai_ready else 'missing'}")
    st.caption(f"LLM model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    st.caption(f"F5 guardrail config: {'configured' if f5_configured else 'missing'}")
    st.caption(f"Agent calls through F5: {'enabled' if f5_agent_proxy else 'disabled'}")
    if guardrails_enabled and not f5_configured:
        st.warning("Guardrails are ON but F5 is not configured. The run will block at the first inter-agent boundary.")
    if not openai_ready and not f5_agent_proxy:
        st.warning("OPENAI_API_KEY is missing. Live agent calls will fail until it is set.")

scenario = scenario_by_key(selected_key)
raw_request = custom_request if use_custom and custom_request.strip() else scenario.request
fragments = () if use_custom else scenario.injected_fragments

main_col, ops_col = st.columns([1.55, .85], gap="large")

with main_col:
    st.subheader("Architecture and live flow")
    flow_state = empty_flow_state()
    flow_chart_slot = st.empty()
    flow_chart_slot.markdown(render_flow_chart(flow_state, guardrails_enabled=guardrails_enabled), unsafe_allow_html=True)
    render_agent_popovers()
    activity_slot = st.empty()
    with activity_slot.container():
        st.subheader("Logs and activity")
        st.info("Run the pipeline to stream agent work, tool calls, guardrail verdicts, and out-of-bounds alerts here.")

with ops_col:
    st.markdown(
        f"""
<div class="ops-panel">
  <div class="ops-title">{ICONS["monitor"]} Operations panel</div>
  <div class="ops-subtitle">Live request context and run telemetry for the audience.</div>
  <div class="ops-section">
    <div class="ops-section-title">{ICONS["doc"]} Input request</div>
    <div class="mono-box">{highlight(raw_request, fragments)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

if run_clicked:
    with st.spinner("Running live agents..."):
        try:
            def update_flow(event: PipelineEvent) -> None:
                apply_progress_event(flow_state, event)
                flow_chart_slot.markdown(render_flow_chart(flow_state, guardrails_enabled=guardrails_enabled), unsafe_allow_html=True)

            scenario_name = "Custom request" if use_custom else scenario.label
            result = run_pipeline(
                raw_request,
                guardrails_enabled=guardrails_enabled,
                scenario_name=scenario_name,
                on_event=update_flow,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
        else:
            with ops_col:
                st.markdown(
                    f'<div class="ops-section-title">{ICONS["shield"]} Final outcome</div>',
                    unsafe_allow_html=True,
                )
                render_banner(result)
                st.caption(f"Trace ID: {result.trace_id}. Full run logged under runs/.")
            with main_col:
                activity_slot.empty()
                with activity_slot.container():
                    st.subheader("Logs and activity")
                    render_telemetry(result)
                    st.subheader("Pipeline timeline")
                    for step in result.steps:
                        render_step(step, fragments)
else:
    with ops_col:
        st.info("Choose a scenario and run the pipeline when you are ready.")
