# 28 — LIJIE 可视化动漫推荐 Demo 实施手册

## 1. 目标

把已经完成的离线推荐实验包装成一个适合答辩和作品展示的本地 Web Demo，
让非技术观众能够：

1. 看见一个匿名用户的历史兴趣；
2. 获得未看动漫的 Top-10 推荐；
3. 理解 BPR、Weighted NeuMF 和固定融合模型各自做了什么；
4. 查看实验规模、完整流程、最终指标和适用边界。

V1 是展示系统，不是商业推荐平台。不得增加登录、支付、在线训练、用户画像
采集或生产数据库。

## 2. 当前资产盘点

### 已具备

- 完整训练、评估和融合源码；
- 19 份冻结的 final configs；
- 最终指标汇总和审计报告；
- 本地 `anime.csv`，包含 `anime_id`、名称、类型、集数、评分和成员数；
- 服务器上保留的完整 final release；
- BPR 与 Weighted NeuMF 的模型结构和固定 `w=0.7` 百分位融合逻辑。

### 本地尚缺

- 最终 BPR ensemble component 的 `model.pt`；
- 最终 Weighted NeuMF 的 `model.pt`；
- `user_mapping.parquet` 和 `item_mapping.parquet`；
- 用于过滤已看动漫的交互数据；
- 可选的封面 URL 或本地缩略图。

现有 final metadata archive 只用于审计，不含 checkpoint、预测表或 prepared
artifacts，不能直接驱动实时推荐。

## 3. V1 产品范围

### 页面 A：推荐体验

- 从若干匿名示例用户或兴趣画像中选择一个；
- 展示该用户部分已喜欢的动漫；
- 对 7,223 个 warm items 评分并过滤已看项目；
- 输出融合模型 Top-10；
- 每项显示名称、类型、集数、数据集评分和模型排名分数；
- 可以展开查看 BPR 排名、Weighted NeuMF 排名和融合后排名。

### 页面 B：模型解释

- BPR：从用户与动漫的协同关系中学习偏好；
- Weighted NeuMF：学习更复杂的非线性交互，并利用评分置信度；
- Ensemble：先在每位用户内部转换成百分位排名，再计算
  `0.7 × BPR + 0.3 × Weighted NeuMF`；
- 用同一批动漫展示三个排序的差异。

### 页面 C：实验看板

- 60,384 个 eligible users；
- 7,223 个 warm items；
- 5,087,394 个训练正样本；
- 19 次正式运行和 6 份派生 ensemble 结果；
- 各模型 NDCG@10、Hit Rate@10 和 seed 波动；
- “Validation 负责选择，Test 只负责最后报告”的实验链路；
- sampled-candidate 评估边界说明。

## 4. 用户模式边界

V1 推荐使用“匿名已知用户体验模式”。训练好的协同过滤模型只为训练数据中的
已知用户学习了 user embedding，不能把一个新访客随便选择的三部动漫直接当成
经过最终实验验证的 ensemble 输入。

如果以后增加“选择几部喜欢的动漫”功能，必须明确标记为以下二者之一：

- **冷启动 fallback**：内容相似度加流行度；不继承最终 ensemble 指标；
- **临时用户 embedding 适配**：冻结 item embedding，只优化新用户向量；属于
  新方法，需要单独验证，不能沿用本轮 test 结论。

答辩版 V1 不应把冷启动扩展伪装成已验证的最终模型。

## 5. 推荐技术方案

- UI：Streamlit；
- 推理：PyTorch CPU；
- 数据处理：Pandas + PyArrow；
- 图表：Streamlit 原生图表或 Altair；
- 默认模型：seed 42 的 BPR ensemble component 与 Weighted NeuMF；
- 排序范围：完整 7,223 warm-item catalog；
- 启动方式：本地单进程，不依赖服务器和网络。

训练必须使用 GPU；Demo 的单用户推理规模很小，CPU 是合理部署方式，不属于
训练资源浪费。只有批量生成全部 60,384 用户推荐时才需要重新评估 GPU 方案。

## 6. 预期目录

```text
demo/
├── app.py
├── inference.py
├── data_access.py
├── pages/
│   ├── model_explainer.py
│   └── experiment_dashboard.py
├── assets/
│   └── anime_metadata.parquet
└── README.md

demo_bundle/                 # Git ignored
├── manifest.json
├── bpr_seed42_model.pt
├── weighted_neumf_seed42_model.pt
├── user_mapping.parquet
├── item_mapping.parquet
├── observed_by_user.parquet
└── demo_users.parquet
```

`demo_bundle/` 必须加入 `.gitignore`。模型、用户交互和大文件不得提交 GitHub。

## 7. Stage A — 制作最小推理包

### A1. 重新申请一次性 SSH 凭据

使用新的临时密钥连接服务器。不得复用或提交旧密钥；主机名、用户名和密钥路径
只放在当前终端环境变量中，不写入脚本、Markdown、Git config 或日志。

### A2. 固定来源

只从 final-run commit
`b2f4d6b8222f9f5a9afd0633f54a235f50e52c69` 对应的保留 release 读取文件。
选择 seed 42 只是为了部署一个确定版本，不改变三 seed 的正式报告。

### A3. 文件白名单

允许进入 bundle 的内容：

- BPR ensemble component seed 42 `model.pt`；
- Weighted NeuMF seed 42 `model.pt`；
- user/item mapping；
- 过滤已看物品所需的最小交互表；
- 10–20 个匿名 demo users 及其展示历史；
- 不含机器路径的 manifest 和文件 SHA-256。

禁止进入 bundle：

- SSH 私钥、公钥、authorized key 内容；
- test labels、test candidates 和 test predictions；
- 服务器地址、账户名和绝对目录；
- 其他模型 checkpoint、训练日志和无关项目文件。

### A4. Gate A

- 两个 checkpoint 均可由固定配置加载；
- mapping 行数与模型 embedding shape 一致；
- demo users 全部存在于 user mapping；
- 每个 bundle 文件均记录 SHA-256；
- 敏感信息扫描为 0；
- 下载完成并验证 hash 后，立即撤销一次性公钥并删除本地密钥。

未通过 Gate A，不进入编码阶段。

## 8. Stage B — 建立独立推理层

新增推理接口，不直接调用训练入口：

```python
recommend(user_id: int, top_k: int = 10) -> list[Recommendation]
```

接口必须完成：

1. 加载两个冻结 checkpoint 和 mapping；
2. 对完整 warm catalog 生成两个模型的分数；
3. 过滤该用户所有已观察物品；
4. 按现有 `src.ensemble.percentile_ranks` 语义计算用户内百分位；
5. 使用固定 `0.7 / 0.3` 权重融合；
6. 按融合分数降序、`anime_id` 升序打破完全相同的分数；
7. 与 `anime.csv` 连接，返回可展示字段。

模型只在进程启动时加载一次，并使用 Streamlit resource cache。请求期间不得训练、
修改权重或读取 test 数据。

### Gate B

- checkpoint shape 和 config 匹配；
- 同一输入重复调用得到完全相同的 Top-10；
- 推荐列表不含用户已观察项目；
- 融合分数与离线 ensemble 函数的一致性测试通过；
- 缺失 user、item 或 checkpoint 时返回清楚错误，不静默 fallback；
- CPU 上单次请求达到可交互速度，建议目标低于 2 秒。

## 9. Stage C — 构建 Streamlit UI

### C1. 推荐体验

- 顶部用一句话解释项目；
- 左侧选择匿名兴趣画像，避免直接暴露原始 user ID；
- 主区域先展示历史兴趣，再展示 Top-10；
- 每张推荐卡只保留必要字段；
- 用“为什么出现”展开区展示两个组件的相对排名；
- 无封面时使用统一占位图，不让外部图片失败阻塞页面。

### C2. 实验解释

- 使用现有 `docs/EXPERIMENT_PIPELINE_OVERVIEW.md` 的阶段结构；
- 指标图从已归档的 final summary 派生，不手工重复录入多份数值；
- NDCG 和 HR 分开绘制，避免把不同含义的指标混成一个“准确率”；
- 显示三 seed 均值与标准差；
- 明确写出 1 positive + 99 negatives。

### Gate C

- 首屏在不了解算法的情况下也能知道“输入是什么、输出是什么”；
- 320px、普通笔记本和投影宽度下无文字重叠；
- 键盘可完成用户选择和页面切换；
- 任何页面都不出现 test label、机器路径或凭据；
- 指标用语与最终报告一致。

## 10. Stage D — 测试与答辩验收

新增自动化测试：

- checkpoint 加载与维度检查；
- seen-item filtering；
- 百分位融合和 tie-breaking；
- Top-K 数量及排序；
- metadata 缺失值处理；
- Streamlit 启动 smoke test；
- bundle 缺失和 hash 不匹配时 fail-fast。

人工答辩脚本：

1. 选择一个“科幻 / 悬疑”匿名画像；
2. 指出其历史兴趣；
3. 生成 Top-10；
4. 展开一个推荐，解释两个模型如何共同决定排名；
5. 切到实验看板，说明 test 在选择结束后只打开一次；
6. 展示最终 `0.799658` NDCG@10 和 `0.956335` HR@10；
7. 主动说明这是 sampled-candidate、warm-user 离线结果。

### Gate D

- 原有测试保持 `29 passed`；
- 新增 Demo 测试全部通过；
- 断网情况下核心推荐和指标页面仍可用；
- 连续演示三个用户无崩溃、无状态串扰；
- Git 敏感信息和绝对机器路径扫描为 0。

## 11. Stage E — 留痕与交付

- 提交 Demo 源码、测试、说明和 bundle manifest 模板；
- 不提交真实 checkpoint、交互表、原始数据和封面缓存；
- README 写明本地启动命令和 bundle 放置方式；
- 截图或录屏只展示匿名画像；
- 记录最终 commit 和测试结果；
- GitHub 推送后再次确认分支同步且工作区干净。

## 12. 工作量与推荐顺序

| 工作 | 预计时间 |
|---|---:|
| 安全导出并验证最小推理包 | 1–2 小时 |
| 推理层与测试 | 3–5 小时 |
| 推荐体验页面 | 3–4 小时 |
| 实验看板与解释页面 | 2–3 小时 |
| 视觉整理、答辩脚本和验收 | 2–3 小时 |

建议先完成“匿名已知用户 + 实时 Top-10 + 实验看板”的 V1。冷启动自选动漫
属于 V2，只有在 V1 稳定且时间充足时再做。

## 13. 停止条件

出现以下任一情况立即停止，不绕过 Gate：

- checkpoint 与 config、mapping 或 source commit 不一致；
- 需要读取 test labels 才能驱动页面；
- 推荐包含用户已观察动漫；
- 页面展示的指标无法追溯到 final summary；
- 凭据、服务器坐标或绝对机器路径进入 staged diff；
- 为了支持新用户而改变算法，却仍沿用原 final test 指标。
