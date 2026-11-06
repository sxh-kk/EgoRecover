"""The four plan.docx experiments as explicit, reviewable configurations."""

from dataclasses import replace

from .config import WindowConfig


def experiment_matrix(selected_window=3):
    base = WindowConfig(window=selected_window, current_weight=0)
    rows = []

    def add(name, config, *, initialize=None, reuse=None, sweep=(1,), purpose=""):
        rows.append(
            {
                "name": name,
                "protocol": config.to_dict(),
                "initialize_from": initialize,
                "reuse": reuse,
                "evaluation_steps": list(sweep),
                "horizons": [1, 2, 4, 10],
                "seeds": [62, 63, 64],
                "purpose": purpose,
            }
        )

    for width in (1, 3, 5, 10):
        add(f"architecture/W{width}", replace(base, window=width), purpose="活动窗口长度")
        if width > 1:
            add(
                f"architecture/W{width}_fixed",
                replace(base, window=width, revise_recent=False),
                purpose="同窗口固定近期状态",
            )
        add(
            f"architecture/W{width}_no_body",
            replace(base, window=width, body_feedback=False),
            purpose="相同21帧观测范围、无身体反馈",
        )
    supervised = replace(base, current_weight=1)
    for source in ("gaussian", "prediction"):
        add(f"warmup/{source}", replace(supervised, source=source), purpose="同预算基础模型；供全部继续训练组共享")
        add(
            f"source/{source}",
            replace(supervised, source=source),
            initialize=f"warmup/{source}",
            sweep=(1, 2, 4, 10),
            purpose="共享P与显式中心；匹配缓存与独立连续运行",
        )
    add("feedback/base", supervised, reuse="source/gaussian", purpose="与反馈组同预算的当前监督基线")
    for kind in ("paired", "direct", "feature", "geometry"):
        add(
            f"feedback/{kind}",
            replace(supervised, feedback=kind),
            initialize="warmup/gaussian",
            purpose="同一基础模型继续训练；冻结P与后续EMA，保留输入梯度",
        )
    add("combination/A", supervised, reuse="feedback/base")
    add("combination/B", replace(supervised, source="prediction"), reuse="source/prediction")
    add("combination/C", replace(supervised, feedback="paired"), reuse="feedback/paired")
    add(
        "combination/D",
        replace(supervised, source="prediction", feedback="paired"),
        initialize="warmup/prediction",
        purpose="预测中心与成对反馈的组合",
    )
    return rows


def interaction(scores):
    """Positive interaction means error drops exceed additive independent gains."""
    if set(scores) != {"A", "B", "C", "D"}:
        raise ValueError("Require matched A/B/C/D error measurements.")
    a, b, c, d = (scores[key] for key in "ABCD")
    return {
        "source_gain": a - b,
        "feedback_gaussian_gain": a - c,
        "feedback_prediction_gain": b - d,
        "combined_gain": a - d,
        "interaction": b + c - a - d,
    }
