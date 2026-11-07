"""Random forests fit on all cores but predict on one thread.

With ``n_jobs`` other than 1, scikit-learn's ``RandomForestClassifier.predict_proba`` adds the
trees' predictions together in the order its threads finish. Floating-point addition depends on
the order, so two identical runs can give probabilities that differ in the last bit, and a tie
between two scores can then break differently (which moves a ROC AUC slightly). Fitting is not
affected: with a fixed ``random_state`` every tree is the same however many threads build them.

So the package fits forests with ``n_jobs=-1`` and then sets ``n_jobs=1`` before any
prediction. Reruns then match bit for bit, and saved models score the same way every time.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def predict_on_one_thread(model):
    """Set ``n_jobs=1`` on a fitted random forest, or on any forest step of a pipeline.

    Other models are returned unchanged. Returns ``model`` so calls can be chained.
    """
    steps = [s for _, s in model.steps] if hasattr(model, "steps") else [model]
    for step in steps:
        if isinstance(step, RandomForestClassifier):
            step.set_params(n_jobs=1)
    return model
