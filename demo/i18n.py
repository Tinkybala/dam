"""Small, explicit translation table for the bilingual demo UI."""

from __future__ import annotations

from typing import Literal

Language = Literal["zh", "en"]


TEXT: dict[Language, dict[str, str]] = {
    "zh": {
        "page_title": "动漫 Top-10 推荐系统",
        "switch_language": "English",
        "subtitle": "基于数据集中匿名已知用户的完整候选动漫推荐",
        "intro": "选择一位数据集中的匿名已知用户，系统会过滤其已看作品并推荐 10 部未看动漫。",
        "select_user": "选择匿名用户",
        "anonymous_user": "匿名用户 {number:02d}",
        "history_metric": "该用户的历史动漫",
        "catalog_metric": "候选动漫",
        "history_heading": "部分历史动漫",
        "more_history": "查看更多历史记录",
        "recommendation_heading": "为该用户推荐的 Top-10",
        "dataset_rating": "数据集评分 {rating:.2f}",
        "model_note": "推荐来自冻结的 BPR 与 Weighted NeuMF 模型及预先锁定的 0.7/0.3 融合；最终实验指标基于 1+99 抽样候选集，不代表这里的全目录准确率。",
        "poster_note": "海报元数据来自 AniList，仅用于本地教学展示；缺失图片会显示占位图。",
        "loading": "正在加载冻结模型…",
        "bundle_error": "演示数据包验证失败，请检查 README 中的数据包放置方式。",
        "unknown": "未知",
        "name_column": "动漫名称",
        "genre_column": "类型",
        "type_column": "载体",
        "episodes_column": "集数",
        "rating_column": "数据集评分",
    },
    "en": {
        "page_title": "Anime Top-10 Recommender",
        "switch_language": "中文",
        "subtitle": "Full-catalog recommendations for anonymous known users in the dataset",
        "intro": "Select an anonymous known user. The system filters out watched titles and recommends 10 unseen anime.",
        "select_user": "Select an anonymous user",
        "anonymous_user": "Anonymous User {number:02d}",
        "history_metric": "Anime in user history",
        "catalog_metric": "Candidate anime",
        "history_heading": "Sample viewing history",
        "more_history": "View more history",
        "recommendation_heading": "Top-10 recommendations for this user",
        "dataset_rating": "Dataset rating {rating:.2f}",
        "model_note": "Recommendations use the frozen BPR and Weighted NeuMF models with the locked 0.7/0.3 ensemble. Final experiment metrics use 1+99 sampled candidates and do not measure full-catalog accuracy here.",
        "poster_note": "Poster metadata comes from AniList for local educational display only; missing images use a placeholder.",
        "loading": "Loading frozen models…",
        "bundle_error": "Demo bundle validation failed. Check the bundle setup instructions in the README.",
        "unknown": "Unknown",
        "name_column": "Anime title",
        "genre_column": "Genre",
        "type_column": "Format",
        "episodes_column": "Episodes",
        "rating_column": "Dataset rating",
    },
}


def text(language: Language, key: str, **values: object) -> str:
    """Return one translated UI string."""

    return TEXT[language][key].format(**values)
