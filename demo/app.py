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
from demo.inference import Recommender  # noqa: E402
from demo import recommendation  # noqa: E402


@st.cache_resource(show_spinner="正在加载冻结模型…")
def load_recommender(bundle_dir: str, metadata_path: str) -> Recommender:
    return Recommender(bundle_dir, metadata_path)


def main() -> None:
    st.set_page_config(page_title="Anime Top-10 Recommender", page_icon="🎬", layout="wide")
    st.title("Anime Top-10 Recommender")
    st.caption("基于数据集中匿名已知用户的完整 warm-catalog 推荐")
    bundle = Path(os.environ.get("DAM_DEMO_BUNDLE", ROOT / "demo_bundle")).expanduser()
    metadata = Path(os.environ.get("DAM_DEMO_METADATA", ROOT / "demo" / "assets" / "anime_metadata.parquet")).expanduser()
    try:
        engine = load_recommender(str(bundle), str(metadata))
    except (BundleError, OSError, ValueError, RuntimeError) as exc:
        st.error("Demo bundle 验证失败，请检查 README 中的 bundle 放置方式。")
        st.code(type(exc).__name__)
        st.stop()

    recommendation.render(engine)


if __name__ == "__main__":
    main()
