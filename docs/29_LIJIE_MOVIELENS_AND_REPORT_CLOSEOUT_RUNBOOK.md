# MovieLens 1M 与 Project 1 报告收尾操作手册

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-05
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1
- Owner: Lijie model-based recommender workstream

## 0. 本轮目标

今天下午完成当前项目相对于课程要求的三个缺口：

1. 在第二个数据集 MovieLens 1M 上完成可复核的迁移实验；
2. 生成可直接用于报告的比较表、参数图和成功/失败案例；
3. 建立正式报告与提交包的完整骨架，并通过提交前 Gate。

本轮不重新训练或重新选择 Anime 模型，不重新搜索 ensemble 权重，也不根据
Anime test 结果继续调参。

### 完成定义

以下条件必须全部满足，才能说本轮完成：

- MovieLens 1M 数据制品通过 schema、split、candidate 和 checksum Gate；
- 至少 Popular、BPR、NeuMF、Weighted NeuMF 在 MovieLens 上产生正式结果；
- 固定 `0.7 BPR + 0.3 Weighted NeuMF` ensemble 被作为迁移验证，而不是重新调权；
- MovieLens test 只在配置与 commit 锁定后读取一次；
- 结果表明确区分 Anime 与 MovieLens，不直接比较两者绝对分数；
- 至少生成 3 张分析图和 1 组成功/失败案例；
- report 包含 Abstract、Introduction、Methods、Experiments、Conclusion；
- contribution PDF 和 source-only ZIP 的文件清单已确定；
- Git 中没有 dataset、checkpoint、prediction、海报缓存、密钥或服务器坐标。

## 1. 时间安排

| 时间段 | 阶段 | 目标 |
|---|---|---|
| 13:00-13:45 | Stage A | MovieLens 数据适配与自动化测试 |
| 13:45-14:15 | Stage B | 配置、分析脚本和本地 Gate |
| 14:15-14:45 | Stage C | 服务器部署、数据准备、CUDA smoke test |
| 14:45-16:00 | Stage D | 锁定后的 MovieLens 正式实验 |
| 16:00-16:45 | Stage E | 聚合、图表和案例分析 |
| 16:45-18:00 | Stage F | 报告正文骨架与关键讨论 |
| 18:00-18:30 | Stage G | 三个提交文件的最终 Gate |

MovieLens 1M 规模明显小于 Anime，预计正式 GPU 训练不是主要耗时。若 16:00 前仍未
完成 Stage D，应先保住第二数据集的四个核心方法，暂停非必要的 GMF 扩展，不得通过
减少数据、查看 test 后改参数或改用 CPU 来赶时间。

## 2. Experiment Overview

- **Title**: Cross-dataset verification of model-based Top-N recommendation
- **Objective**: 检查 Anime 上的主要方法结论能否迁移到 MovieLens 1M
- **Primary question**: BPR、NeuMF 和 Weighted NeuMF 的相对表现是否与 Anime 一致？
- **Secondary question**: 固定 `0.7/0.3` ensemble 是否仍优于其两个 component？
- **Type**: ETL + GPU training + offline ranking analysis
- **Primary metric**: NDCG@10
- **Secondary metric**: Hit Rate@10
- **Protocol**: 每位用户 1 个 held-out positive + 99 个固定 unseen negatives
- **Positive definition**: MovieLens rating `>= 4`
- **Warm-start rule**: positive user/item iterative 5-core
- **Seeds**: 42、43、44；Popular 只运行一次

### 迁移原则

MovieLens 不是第二轮模型竞赛。所有模型结构和核心超参数从 Anime 已锁定配置迁移，
只允许修改以下数据集相关字段：

- artifacts/output 路径；
- `positive_threshold: 4`；
- `maximum_rating: 5`；
- seed；
- `evaluate_test` 状态；
- 为显存或输入规模所需的 batch size，但必须对直接比较的方法一致记录。

禁止根据 MovieLens test 重新选择学习率、embedding、negative ratio、alpha 或
ensemble weight。若需要 smoke-test 修复，只能修复代码错误或资源错误，不能依据指标
高低替换配置。

## 3. 目录与产物

所有示例路径均相对于仓库根目录。服务器实际根目录必须通过环境变量或手工进入 release
目录指定，不要把用户名、主机名或绝对服务器路径写进 Git。

| 类型 | 路径 | 是否入 Git |
|---|---|---:|
| MovieLens 配置 | `configs/movielens/` | 是 |
| 数据适配与分析代码 | `src/` | 是 |
| 自动化测试 | `tests/` | 是 |
| MovieLens 原始数据 | `data/movielens-1m/` | 否 |
| 冻结数据制品 | `artifacts/movielens-1m-r4/` | 否 |
| metrics/checkpoint/predictions | `results/movielens/` | 否 |
| 报告图表源文件 | `results/movielens/analysis/` | 否 |
| 无敏感信息结果摘要 | `evidence/movielens_transfer_results_20260905.md` | 是 |
| 最终提交文件 | `submission/` | 否，交 NTULearn |

## 4. Stage A - MovieLens 数据适配

### A1. 实现范围

在现有代码上做最小改动，不把已经归档的 Anime artifacts 重新生成：

1. 在 `src/data.py` 增加 MovieLens 1M loader；
2. loader 读取 `ratings.dat` 的 `UserID::MovieID::Rating::Timestamp`；
3. 丢弃 timestamp，将 MovieID 映射到现有内部 item-ID 列；
4. 在 `src.prepare` 增加 `--dataset anime|movielens-1m`；
5. manifest 必须正确记录 dataset、原始文件 SHA-256、rating 范围和参数；
6. 其余 split、warm-item repair、observed exclusion 和 candidate sampling 复用现有逻辑。

内部暂时沿用现有 `anime_id` 列名是兼容旧 artifacts 的工程选择。报告和 MovieLens
结果中必须称它为 movie/item ID，不能把 MovieLens item 写成 anime。

### A2. 必须新增的测试

- `ratings.dat` 的双冒号解析正确；
- timestamp 不进入模型输入；
- 输出 ID 与 rating 类型为整数；
- rating `>= 4` 才进入 positives；
- rating `< 4` 仍属于 observed，不可作为该用户负样本；
- 同 seed 两次准备得到相同逻辑内容；
- validation/test 每位用户各一个 positive；
- 所有 held-out items 在 train 中可见；
- 每位用户恰好有 100 个候选且只有一个 positive；
- 原有 Anime 测试不回归。

### A3. 本地 Gate

```powershell
python -m pytest
git diff --check
git status --short
```

Gate：所有测试通过，且修改范围只包含预期 source、config、test 和 runbook。Stage A
若失败，不得部署服务器。

## 5. Stage B - 配置锁定

### B1. MovieLens 配置矩阵

创建下列 validation-sealed 配置，初始必须全部为 `evaluate_test: false`：

```text
configs/movielens/
  movielens_popular.yaml
  movielens_bpr_seed42.yaml
  movielens_bpr_seed43.yaml
  movielens_bpr_seed44.yaml
  movielens_gmf_seed42.yaml
  movielens_gmf_seed43.yaml
  movielens_gmf_seed44.yaml
  movielens_neumf_seed42.yaml
  movielens_neumf_seed43.yaml
  movielens_neumf_seed44.yaml
  movielens_weighted_neumf_seed42.yaml
  movielens_weighted_neumf_seed43.yaml
  movielens_weighted_neumf_seed44.yaml
```

参数来源：

- BPR 使用 Anime ensemble-component 的锁定结构；
- GMF、NeuMF、Weighted NeuMF 使用对应 Anime final config；
- Weighted NeuMF 保持 `confidence_alpha: 0.5`；
- trainable configs 使用 `device: cuda` 和 `gpu_sampling: true`；
- MovieLens 使用 `positive_threshold: 4`、`maximum_rating: 5`；
- epochs 和 early stopping 与来源配置保持一致。

### B2. 配置审计

```powershell
rg -n "evaluate_test|device|gpu_sampling|positive_threshold|maximum_rating|seed" configs/movielens
python -m pytest
```

Gate：

- 13/13 配置均为 `evaluate_test: false`；
- 12/12 trainable 配置均为 `device: cuda`；
- 12/12 均为 `gpu_sampling: true`；
- 不存在 `device: auto`；
- seeds 只能是 42、43、44；
- 不存在 test-derived 参数。

通过后提交并推送一个 **MovieLens candidate-lock commit**。记录完整 commit SHA；后续
服务器 release 必须由该 commit 创建。

## 6. Stage C - 数据准备与 CUDA preflight

### C1. 数据获取

只从 GroupLens MovieLens 1M 官方发布包取得 `ml-1m.zip`。数据留在服务器 data 目录，
不上传 Git，也不放进 release 源码包。记录：

- 下载时间；
- 文件字节数；
- SHA-256；
- 解压后的 `ratings.dat` 行数。

### C2. 准备 artifacts

Stage A 完成后，命令接口应为：

```bash
python -m src.prepare \
  --dataset movielens-1m \
  --ratings <SERVER_DATA_ROOT>/movielens-1m/ratings.dat \
  --output artifacts/movielens-1m-r4 \
  --positive-threshold 4 \
  --core-size 5 \
  --seed 42 \
  --negative-count 99 \
  --development-user-count 10000
```

MovieLens 合格用户若少于 10,000，development file 应自然包含全部合格用户，不得重复
采样或虚构用户。

### C3. Artifact Gate

必须检查：

- manifest dataset 是 `movielens_1m`；
- 原始 ratings SHA 与下载文件一致；
- train/validation/test 用户集合一致；
- validation/test positive 各为每用户一条；
- validation/test candidate 各为每用户 100 条；
- 所有负样本从未被对应用户观察；
- mapping 行数与 manifest user/item count 一致；
- 关键数值全部有限；
- test candidate 文件已生成但未被训练命令读取。

### C4. 一轮 CUDA smoke test

从 seed 42 的 BPR 和 Weighted NeuMF 配置复制 smoke config，只允许修改：

- `epochs: 1`；
- `early_stopping_patience: 1`；
- output path；
- 保持 `evaluate_test: false`。

运行时一张卡一个任务：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config <BPR_SMOKE_CONFIG>
CUDA_VISIBLE_DEVICES=1 python -m src.train --config <WEIGHTED_NEUMF_SMOKE_CONFIG>
```

Gate：

- 两个任务 exit code 均为 0；
- metrics 只有 validation，无 test key；
- 日志无 traceback、OOM、non-finite 或 timeout；
- `nvidia-smi` 证明两个进程分别附着 GPU；
- metrics commit 等于 candidate-lock commit。

若日志显示 CPU，立即停止，不得继续正式运行。

## 7. Stage D - MovieLens 正式实验

### D1. Test 解封

只有 Stage C 全部通过后，才将 13 个 MovieLens 配置中的
`evaluate_test: false` 改为 `evaluate_test: true`。除此之外配置不得变化。

完成以下检查后提交 **MovieLens final-run commit**：

```bash
git diff <CANDIDATE_LOCK_COMMIT> -- configs/movielens
python -m pytest
```

人工确认 diff 只包含 test 解封。如果混入学习率、结构、alpha、negative ratio、epoch
或 seed 变化，Gate 失败。

### D2. 调度

- GPU 0：BPR 与 NeuMF seeds；
- GPU 1：GMF 与 Weighted NeuMF seeds；
- Popular 在 CPU 单独运行；
- 同一 GPU 同时只允许一个训练任务；
- 每个任务独立 log、metrics、checkpoint 和 prediction 目录；
- 不自动重试，不覆盖已有 metrics；
- hard timeout 建议 45 分钟/任务；MovieLens 正常情况应远低于此值。

运行期间至少监控：

```bash
nvidia-smi
ps -ef | grep -E "src.train|src.tune" | grep -v grep
```

### D3. Final Gate

- 13/13 metrics 文件存在；
- 13/13 status complete；
- 12/12 trainable run 记录 CUDA；
- 每个 metrics 同时有 validation 与 test；
- 所有 commit 等于 final-run commit；
- 无 retry、timeout、覆盖或人工改参；
- 每模型 seeds 42/43/44 齐全；
- Popular 仅一条确定性结果。

失败时保留现场并停止。不得删除失败日志后假装完整，也不得看到 test 后更改配置重跑。

## 8. Stage E - Ensemble、聚合与图表

### E1. 固定 ensemble

对每个 seed 使用已有 per-user percentile-rank 逻辑：

```text
ensemble_score = 0.7 * BPR_percentile + 0.3 * Weighted_NeuMF_percentile
```

权重不可重新搜索。分别对 validation 和 test 生成三个 seed 的 ensemble 结果，并验证：

- component 用户和 candidate pair 完全一致；
- 每用户恰好一个 positive；
- 无 duplicate user-item pair；
- score 和 metric 为有限数值；
- 输出不覆盖 component predictions。

### E2. 最终聚合表

MovieLens 表至少包含：

| Dataset | Model | Seeds | NDCG@10 mean ± SD | HR@10 mean ± SD | Runtime |
|---|---|---:|---:|---:|---:|

Anime 与 MovieLens 分成两个 panel 或两张表。讨论“模型排序和增益方向是否一致”，不要用
`MovieLens NDCG 高于 Anime` 这种方式得出数据集更容易或模型更好的结论。

### E3. 三张必需图

只生成分析图，不制作装饰性流程图：

1. `model_comparison.png`：每个数据集内各模型 NDCG@10，三 seed error bar；
2. `component_ablation.png`：NeuMF、Weighted NeuMF、BPR component、ensemble；
3. `parameter_analysis.png`：使用已有 Anime validation-only 结果展示 alpha 或
   negative-sampling/learning-rate 对 NDCG@10 的影响。

图表 Gate：

- 坐标轴、metric、dataset、candidate protocol 和 seed aggregation 标注完整；
- error bar 明确是跨三个训练 seed 的 SD，不是置信区间；
- 不截断 y 轴来夸大微小差异，若截断必须显式标注；
- validation 参数图不得混入 test 结果；
- PNG 清晰，报告中字号可读；
- 每张图都能追溯到 summary/config/commit。

### E4. 成功与失败案例

从正式 test predictions 中只读选取：

- 成功案例：held-out positive rank = 1；
- 边界成功：held-out positive rank = 10；
- 失败案例：held-out positive rank > 10；
- 分歧案例：BPR hit 但 Weighted NeuMF miss，或反之。

选择规则必须确定，例如每类取 user ID 最小的 1-2 位，避免挑选最漂亮的个案。匿名化
用户 ID；展示历史摘要、正样本排名和主要推荐，不展示原始身份。案例只用于解释，不能
反过来改模型。

## 9. Stage F - 正式报告

目标长度建议 6-8 页，正文 11pt 或 12pt。课程最低要求是 4 页，但不要靠压缩字号或
堆附录满足。

### 9.1 Abstract

150-200 词，包含：任务、两个数据集、比较方法、evaluation protocol、最佳结果和主要
限制。最后写结果，不写实验过程流水账。

### 9.2 Introduction

- Top-N recommendation 问题；
- 为什么比较传统 latent-factor 与 neural methods；
- 为什么加入 confidence weighting 与 fixed ensemble；
- 研究问题与贡献；
- 两个数据集的作用：Anime 主实验、MovieLens 迁移验证。

### 9.3 Methods

- Popular、BPR、GMF、MLP、NeuMF、Weighted NeuMF；
- confidence-weight 公式与 `alpha=0` control；
- percentile-rank ensemble；
- 统一 positive/negative、split、candidate 和 metric 定义；
- sampled-candidate 与 full-catalog demo 的区别。

### 9.4 Experiments

至少包含：

1. 两个数据集的规模、阈值和预处理表；
2. 主要模型比较表；
3. 参数设置和 validation-only tuning 说明；
4. component/architecture ablation；
5. MovieLens 迁移结果；
6. 成功与失败案例；
7. runtime/GPU 说明；
8. 优缺点与影响性能的关键因素。

讨论必须回答：

- BPR 为什么强或弱；
- neural model 是否稳定优于简单模型；
- confidence weighting 的收益是否一致；
- ensemble 的收益来自互补还是只跟随强 component；
- sparsity、negative sampling 和 candidate protocol 如何影响结果；
- 为什么 sampled HR/NDCG 不能解释为线上推荐准确率。

### 9.5 Conclusion

用一段总结结果，一段说明限制和可行改进。改进方向可包括 cold-start fallback、内容特征、
全目录评估和在线反馈，但不能声称已经实现。

## 10. Stage G - 提交文件

课程要求提交三个文件：

```text
group_XX_report.pdf
group_XX_contribution.pdf
group_XX_code.zip
```

在获得真实 group ID 前保留 `XX`，不得猜测。

### G1. Report PDF Gate

- 至少 4 页；
- 11pt 或 12pt；
- 五个规定章节齐全；
- 表格和图片无裁切、重叠或模糊；
- 所有数字与 final summary 一致；
- Anime 与 MovieLens protocol 清楚标注；
- limitation 与失败案例存在；
- 不依赖 GitHub、Google Drive 或其他云端附件。

### G2. Contribution PDF Gate

按成员逐项列出：代码、实验、分析、图表、报告文字和整合工作。每一项必须可核实，不写
模糊的“helped with everything”。需要全组确认后再导出 PDF。

### G3. Source-only ZIP Gate

ZIP 可以包含 source、configs、tests、必要 README 和复现脚本，但不得包含：

- dataset；
- model/checkpoint；
- predictions 或 bulk results；
- `.git`、`.venv`、cache；
- SSH key、API key、token、服务器地址或本机绝对路径；
- 指向云端文件来替代应提交内容的链接。

打包前从干净临时目录构建，不要直接压缩整个工作区。解压后执行：

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

记录三个最终文件的 SHA-256 和字节数，再由组内另一位成员人工打开两个 PDF、解压 ZIP
并复核一次。

## 11. 监控与停止规则

- 单任务预计上限：45 分钟；无日志和 GPU 活动超过 5 分钟先检查，不自动 kill；
- hard timeout 触发后保存日志并停止整轮，不自动 retry；
- 出现 OOM 时允许统一减小同类模型 batch size，记录前后值并重新锁定 commit；
- 出现 CPU fallback 时立即停止；
- 出现 non-finite loss、schema mismatch、commit mismatch 或 test 提前读取时立即停止；
- 不删除 Anime final results，不覆盖任何已有 evidence；
- 不在聊天、日志、Git 或报告中粘贴密钥。

## 12. 最终回报模板

执行完成后只回报下面这些信息，不先挑最好看的结果：

```ini
MOVIELENS_DATA_GATE=PASS|FAIL
MOVIELENS_CONFIG_GATE=PASS|FAIL
MOVIELENS_GPU_GATE=PASS|FAIL
MOVIELENS_FINAL_RUNS=13/13
MOVIELENS_ENSEMBLE_GATE=PASS|FAIL
ANALYSIS_FIGURES=3/3
CASE_ANALYSIS=PASS|FAIL
REPORT_GATE=PASS|FAIL
CONTRIBUTION_GATE=PASS|FAIL
CODE_ZIP_GATE=PASS|FAIL
SOURCE_COMMIT=<full sha>
RESULT_SUMMARY_SHA256=<sha256>
```

同时提供：

- MovieLens summary 的本地路径；
- 三张图的本地路径；
- report/contribution/code ZIP 的本地路径与 SHA-256；
- 异常、timeout、retry、参数偏离和缺失项；
- 一次性 SSH key 是否已从服务器和本地清理。

## 13. 本轮停点

本手册发布后先停在计划阶段。下一步应从 Stage A 开始实现 MovieLens adapter 和测试；
不要直接在服务器手工改 CSV 列名并绕过代码与 manifest Gate。
