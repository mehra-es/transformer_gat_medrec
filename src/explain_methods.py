"""
Layered explainability methods (Module 6).

Implemented: DeepSHAP (see src/explain.py)
Deferred to companion study: Integrated Gradients, LRP, GNNExplainer, counterfactuals.
"""

from __future__ import annotations

from typing import Any, Dict

NOT_IMPLEMENTED_MSG = (
    "This explainability method is specified in the architecture but deferred "
    "to the companion study. Only DeepSHAP is experimentally validated in the present work."
)


def integrated_gradients(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError(NOT_IMPLEMENTED_MSG)


def layerwise_relevance_propagation(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError(NOT_IMPLEMENTED_MSG)


def gnn_explainer(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError(NOT_IMPLEMENTED_MSG)


def counterfactual_explanation(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError(NOT_IMPLEMENTED_MSG)


def attention_rollout(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError(NOT_IMPLEMENTED_MSG)
