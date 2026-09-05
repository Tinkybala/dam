"""Model explanation page."""

from __future__ import annotations

import streamlit as st

from demo.inference import Recommender

from .recommendation import _select_user


def render(engine: Recommender) -> None:
    st.header("模型解释")
    st.markdown(
        "- **BPR**：从用户与动漫的协同关系学习 pairwise 偏好。\n"
        "- **Weighted NeuMF**：用非线性交互学习更复杂的偏好，并保留评分置信度。\n"
        "- **Ensemble**：先做用户内百分位排名，再计算 `0.7 × BPR + 0.3 × Weighted NeuMF`。"
    )
    user_id = _select_user(engine, "explainer_user")
    recommendations = engine.recommend(user_id, top_k=10)
    chart = recommendations.set_index("name")[["bpr_percentile", "neural_percentile", "ensemble_score"]]
    chart.columns = ["BPR 百分位", "NeuMF 百分位", "融合分数"]
    st.subheader("同一批动漫的排序来源")
    st.bar_chart(chart)
    st.dataframe(
        recommendations[
            [
                "ensemble_rank",
                "name",
                "bpr_rank",
                "neural_rank",
                "bpr_percentile",
                "neural_percentile",
                "ensemble_score",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.info("百分位分数只用于解释排序相对位置，不是概率，也不代表线上点击率。")

