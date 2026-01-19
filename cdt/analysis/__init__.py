# cdt/analysis/__init__.py
"""Statistical analysis module for causal inference."""

from .statistical_analysis import (
    TreatmentEffectEstimate,
    estimate_att_matched,
    estimate_ate_ipw,
    estimate_ate_stratified,
    mcnemars_test,
    paired_t_test,
    sensitivity_analysis_rosenbaum,
    summarize_analysis
)

__all__ = [
    'TreatmentEffectEstimate',
    'estimate_att_matched',
    'estimate_ate_ipw',
    'estimate_ate_stratified',
    'mcnemars_test',
    'paired_t_test',
    'sensitivity_analysis_rosenbaum',
    'summarize_analysis'
]
