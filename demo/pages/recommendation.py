"""Recommendation experience page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from demo.inference import Recommender


def _select_user(engine: Recommender, key: str) -> int:
    users = engine.demo_users
    labels = users["demo_label"].astype(str).tolist()
    selected = st.selectbox("匿名示例用户", labels, key=key)
    return int(users.loc[users["demo_label"].astype(str) == selected, "user_id"].iloc[0])


def render(engine: Recommender) -> None:
    st.header("推荐体验")
    st.caption("匿名已知用户模式：推荐来自冻结的 seed 42 模型，不收集新用户资料。")
    user_id = _select_user(engine, "recommendation_user")
    profile = engine.demo_users.loc[engine.demo_users["user_id"] == user_id].iloc[0]
    history = engine.history(user_id, limit=30)
    left, right = st.columns(2)
    left.metric("展示历史条数", int(profile["history_count"]))
    right.metric("候选 warm catalog", engine.catalog_stats["catalog_count"])
    with st.expander("这个匿名画像看过的动漫", expanded=False):
        st.dataframe(history, hide_index=True, use_container_width=True)

    top_k = st.slider("推荐数量", min_value=5, max_value=20, value=10, step=1)
    recommendations = engine.recommend(user_id, top_k=top_k)
    display = recommendations[
        ["ensemble_rank", "name", "genre", "type", "episodes", "rating", "ensemble_score"]
    ].rename(
        columns={
            "ensemble_rank": "融合排名",
            "name": "动漫",
            "genre": "类型",
            "type": "形式",
            "episodes": "集数",
            "rating": "数据集评分",
            "ensemble_score": "融合分数",
        }
    )
    st.subheader("Top 推荐")
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption("已过滤该用户的全部 observed history；完全相同分数按 anime_id 升序稳定打破平局。")

