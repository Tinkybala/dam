# Anime Recommender Offline Demo

这是一个本地单进程 Streamlit 展示系统。它使用冻结的 BPR ensemble
component seed 42 与 Weighted NeuMF seed 42，在 CPU 上为匿名已知用户评分完整
warm catalog，并按固定 `0.7 / 0.3` 百分位融合返回 Top-K。

## 启动

在仓库根目录执行：

```bash
python -m pip install -e ".[demo]"
streamlit run demo/app.py
```

默认从仓库根目录的 `demo_bundle/` 读取 bundle，也可以用环境变量覆盖位置：

```bash
DAM_DEMO_BUNDLE=/path/to/demo_bundle streamlit run demo/app.py
```

真实 bundle 被 `.gitignore` 排除，不能提交 checkpoint、用户交互或密钥。启动时
会先逐文件核对 manifest SHA-256；缺失或篡改会 fail-fast。Demo 不读取训练、验证
或 test prediction 文件，也不连接服务器。

`manifest.example.json` 仅是结构模板；真实 manifest 随本地 bundle 一起生成，
不会提交到仓库。

## 页面

- 推荐体验：选择匿名画像，查看历史与融合 Top-K；
- 模型解释：比较 BPR、Weighted NeuMF 与融合的用户内百分位；
- 实验看板：展示最终报告汇总和 sampled-candidate 评估边界。
