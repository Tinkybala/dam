"""Single-panel recommendation experience."""

from __future__ import annotations

from textwrap import shorten

import pandas as pd
import streamlit as st

from demo.i18n import Language, text
from demo.inference import Recommender


def _select_user(engine: Recommender, language: Language) -> int:
    users = engine.demo_users
    user_ids = users["user_id"].astype(int).tolist()
    labels = [
        text(language, "anonymous_user", number=number)
        for number in range(1, len(user_ids) + 1)
    ]
    selected = st.selectbox(
        text(language, "select_user"), labels, key=f"demo_user_label_{language}"
    )
    selected_number = labels.index(selected) + 1
    st.session_state["selected_demo_user_number"] = selected_number
    return user_ids[selected_number - 1]


def _poster_grid(
    engine: Recommender,
    frame: pd.DataFrame,
    *,
    show_rank: bool,
    language: Language,
) -> None:
    records = frame.to_dict("records")
    for start in range(0, len(records), 5):
        columns = st.columns(5)
        for column, row in zip(columns, records[start : start + 5]):
            with column:
                anime_id = int(row.get("anime_id", row.get("item_id")))
                st.image(str(engine.poster_path(anime_id)), width="stretch")
                prefix = f"**#{int(row['ensemble_rank'])} · " if show_rank else "**"
                st.markdown(f"{prefix}{row.get('name', text(language, 'unknown'))}**")
                genre = shorten(
                    str(row.get("genre", text(language, "unknown"))), width=52, placeholder="…"
                )
                details = [genre]
                if pd.notna(row.get("rating")):
                    details.append(
                        text(language, "dataset_rating", rating=float(row["rating"]))
                    )
                st.caption(" · ".join(details))


def render(engine: Recommender, language: Language = "zh") -> None:
    st.caption(text(language, "intro"))
    user_id = _select_user(engine, language)
    profile = engine.demo_users.loc[engine.demo_users["user_id"] == user_id].iloc[0]

    left, right = st.columns(2)
    left.metric(text(language, "history_metric"), int(profile["history_count"]))
    right.metric(text(language, "catalog_metric"), f"{engine.catalog_stats['catalog_count']:,}")

    st.subheader(text(language, "history_heading"))
    history = engine.history(user_id, limit=5)
    _poster_grid(engine, history, show_rank=False, language=language)
    with st.expander(text(language, "more_history")):
        full_history = engine.history(user_id, limit=30).drop(columns=["anime_id"], errors="ignore")
        if language == "zh":
            full_history = full_history.rename(
                columns={
                    "name": text(language, "name_column"),
                    "genre": text(language, "genre_column"),
                    "type": text(language, "type_column"),
                    "episodes": text(language, "episodes_column"),
                    "rating": text(language, "rating_column"),
                }
            )
        st.dataframe(full_history, hide_index=True, width="stretch")

    st.subheader(text(language, "recommendation_heading"))
    recommendations = engine.recommend(user_id, top_k=10)
    _poster_grid(engine, recommendations, show_rank=True, language=language)

    st.caption(text(language, "model_note"))
    st.caption(text(language, "poster_note"))
