# Anime Recommender Offline Demo

这是一个单面板、本地 Streamlit 展示系统。它使用冻结的 BPR ensemble
component seed 42 与 Weighted NeuMF seed 42，在 CPU 上为 20 位匿名已知用户
评分完整 warm catalog，并按固定 `0.7 / 0.3` 百分位融合返回 Top-10。

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

## 展示内容

- 选择一位匿名用户；
- 使用页面右上角按钮在完整中文与英文界面之间切换；
- 查看 5 部历史动漫和可展开的更多历史；
- 查看带本地缓存海报的全目录 Top-10；
- 缺失海报使用仓库内的占位图。

海报只改善展示效果，不参与模型输入或排序。缓存位于已忽略的
`demo_bundle/posters/`，不能提交 Git。使用以下命令从本地原始评分数据扩充用户、
预计算展示集合并获取 AniList 海报：

```bash
python -m demo.prepare_showcase \
  --ratings /path/to/rating.csv \
  --bundle demo_bundle \
  --user-count 20
```

脚本不会读取 validation/test candidates、labels 或 predictions。AniList 只在准备
阶段访问；Demo 运行时保持离线。
