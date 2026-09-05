"""Single-panel recommendation experience."""

from __future__ import annotations

from textwrap import shorten

import pandas as pd
import streamlit as st

from demo.inference import Recommender


def _select_user(engine: Recommender) -> int:
    users = engine.demo_users
    labels = users["demo_label"].astype(str).tolist()
    selected = st.selectbox("选择匿名用户", labels)
    return int(users.loc[users["demo_label"].astype(str) == selected, "user_id"].iloc[0])


def _poster_grid(engine: Recommender, frame: pd.DataFrame, *, show_rank: bool) -> None:
    records = frame.to_dict("records")
    for start in range(0, len(records), 5):
        columns = st.columns(5)
        for column, row in zip(columns, records[start : start + 5]):
            with column:
                anime_id = int(row.get("anime_id", row.get("item_id")))
                st.image(str(engine.poster_path(anime_id)), width="stretch")
                prefix = f"**#{int(row['ensemble_rank'])} · " if show_rank else "**"
                st.markdown(f"{prefix}{row.get('name', 'Unknown')}**")
                genre = shorten(str(row.get("genre", "Unknown")), width=52, placeholder="…")
                details = [genre]
                if pd.notna(row.get("rating")):
                    details.append(f"数据集评分 {float(row['rating']):.2f}")
                st.caption(" · ".join(details))


def render(engine: Recommender) -> None:
    st.caption("选择一位数据集中的匿名已知用户，系统会过滤其已看作品并推荐 10 部未看动漫。")
    user_id = _select_user(engine)
    profile = engine.demo_users.loc[engine.demo_users["user_id"] == user_id].iloc[0]

    left, right = st.columns(2)
    left.metric("该用户的历史动漫", int(profile["history_count"]))
    right.metric("候选动漫", f"{engine.catalog_stats['catalog_count']:,}")

    st.subheader("部分历史动漫")
    history = engine.history(user_id, limit=5)
    _poster_grid(engine, history, show_rank=False)
    with st.expander("查看更多历史记录"):
        full_history = engine.history(user_id, limit=30).drop(columns=["anime_id"], errors="ignore")
        st.dataframe(full_history, hide_index=True, width="stretch")

    st.subheader("为该用户推荐的 Top-10")
    recommendations = engine.recommend(user_id, top_k=10)
    _poster_grid(engine, recommendations, show_rank=True)

    st.caption(
        "推荐来自冻结的 BPR 与 Weighted NeuMF 模型及预先锁定的 0.7/0.3 融合；"
        "最终实验指标基于 1+99 sampled candidates，不代表这里的全目录准确率。"
    )
    st.caption("海报元数据来自 AniList，仅用于本地教学展示；缺失图片会显示占位图。")
