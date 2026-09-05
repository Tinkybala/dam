"""Final experiment dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from demo.experiment_snapshot import EXPERIMENT_FACTS, EXPERIMENT_ROWS
from demo.inference import Recommender


def render(engine: Recommender) -> None:
    st.header("实验看板")
    stats = engine.catalog_stats
    first, second, third = st.columns(3)
    first.metric("Eligible users", f"{stats['eligible_user_count']:,}")
    second.metric("Warm items", f"{stats['catalog_count']:,}")
    third.metric("训练正样本", f"{EXPERIMENT_FACTS['train_positive_count']:,}")

    frame = pd.DataFrame(EXPERIMENT_ROWS).set_index("system")
    table = frame.rename(
        columns={
            "ndcg_mean": "NDCG@10 均值",
            "ndcg_sd": "NDCG@10 SD",
            "hr_mean": "HR@10 均值",
            "hr_sd": "HR@10 SD",
        }
    )
    st.subheader("最终 sampled-candidate test 汇总")
    st.dataframe(table.style.format("{:.6f}"), use_container_width=True)
    st.bar_chart(frame[["ndcg_mean", "hr_mean"]].rename(columns={"ndcg_mean": "NDCG@10", "hr_mean": "HR@10"}))

    st.subheader("实验链路与边界")
    st.write(
        f"{EXPERIMENT_FACTS['formal_run_count']} 次正式运行、"
        f"{EXPERIMENT_FACTS['ensemble_result_count']} 份派生 ensemble；"
        "Validation 负责选择，Test 只负责最后报告。"
    )
    st.write(
        "Ensemble validation→test 差距："
        f"NDCG {EXPERIMENT_FACTS['validation_test_ndcg_gap']:+.6f}，"
        f"HR {EXPERIMENT_FACTS['validation_test_hr_gap']:+.6f}。"
    )
    st.warning(
        "这是每位用户 1 个真实项目 + 99 个负样本的 sampled-candidate 评估，"
        "不等同于完整 7,223 项目录上的绝对指标，也不等同于线上点击率。"
    )

