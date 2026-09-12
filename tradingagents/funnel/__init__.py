"""The screening funnel: turn a large universe into a ranked shortlist cheaply.

Stages, each ~10-20x cheaper per name than the next:
    Stage 0  dataflows.screen.screen_universe   quant screen, no LLM
    Stage 1  funnel.triage.triage_candidates    batched cheap-LLM triage   <-- here
    Stage 2  (the TradingAgentsGraph)            full multi-agent analysis
    Stage 3  funnel.report                       ranked report

Universe sourcing (dataflows.universe) and the Stage 0 screen live in
``dataflows`` as reusable data primitives; everything that spends LLM tokens or
orchestrates the run lives here.
"""

from .batch import AnalysisResult, run_deep_analysis
from .pipeline import FunnelOutput, run_funnel, run_theme_funnel
from .report import build_report, classify_decision
from .theme import expand_theme, theme_slug
from .triage import TriageResult, triage_candidates

__all__ = [
    "TriageResult",
    "triage_candidates",
    "AnalysisResult",
    "run_deep_analysis",
    "build_report",
    "classify_decision",
    "FunnelOutput",
    "run_funnel",
    "run_theme_funnel",
    "expand_theme",
    "theme_slug",
]
