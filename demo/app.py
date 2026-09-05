"""Streamlit entry point for the offline visual recommender demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.data_access import BundleError  # noqa: E402
from demo.i18n import Language, text  # noqa: E402
from demo.inference import Recommender  # noqa: E402
from demo import recommendation  # noqa: E402


@st.cache_resource(show_spinner=False)
def load_recommender(bundle_dir: str, metadata_path: str) -> Recommender:
    return Recommender(bundle_dir, metadata_path)


def main() -> None:
    language: Language = st.session_state.get("language", "zh")
    st.set_page_config(page_title=text(language, "page_title"), page_icon="🎬", layout="wide")

    heading, switch = st.columns([8, 1])
    with heading:
        st.title(text(language, "page_title"))
    with switch:
        if st.button(text(language, "switch_language"), use_container_width=True):
            next_language: Language = "en" if language == "zh" else "zh"
            selected_number = int(st.session_state.get("selected_demo_user_number", 1))
            st.session_state[f"demo_user_label_{next_language}"] = text(
                next_language, "anonymous_user", number=selected_number
            )
            st.session_state["language"] = next_language
            st.rerun()

    st.caption(text(language, "subtitle"))
    bundle = Path(os.environ.get("DAM_DEMO_BUNDLE", ROOT / "demo_bundle")).expanduser()
    metadata = Path(os.environ.get("DAM_DEMO_METADATA", ROOT / "demo" / "assets" / "anime_metadata.parquet")).expanduser()
    try:
        with st.spinner(text(language, "loading")):
            engine = load_recommender(str(bundle), str(metadata))
    except (BundleError, OSError, ValueError, RuntimeError) as exc:
        st.error(text(language, "bundle_error"))
        st.code(type(exc).__name__)
        st.stop()

    recommendation.render(engine, language)


if __name__ == "__main__":
    main()
