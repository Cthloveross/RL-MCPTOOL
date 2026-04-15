# MCPDefender 完整 TODO 清单

**生成日期：2026-04-15**
**写给：下一个接手的 Claude / 合作者**

---

## 0. 项目背景（30 秒速读）

这是一个 MCP tool poisoning defense 研究项目。论文标题："Beyond Average ASR: Fine-Grained Vulnerability Analysis and RL-Based Defense for MCP Tool Poisoning"。

**两个 contribution**：
- **C1**：对 MCPTox (AAAI 2026) 的细粒度 re-analysis（paradigm × tool-category × model-family）
- **C2**：MCPDefender — 用 GRPO 训练 defense，核心卖点是对 unseen attack paradigm 的泛化优于 DPO

**当前 victim model**：Qwen3-8B, Think mode, 4-bit nf4 quantization
**训练数据**：全部来自 MCPTox，Train on T1 (Explicit Hijacking) + T2 (Implicit Hijacking)，Test on T3 (Parameter Tampering, held-out)

---

## 1. 当前状态概览

### 已完成的工作

| 任务 | 结果 | 输出路径 |
|------|------|---------|
| MCPTox 数据 mapping | 11,203/25,079 条 (44.7%) | `experiments/mcptox_analysis/summary.json` |
| Per-tool ASR 分析 | 141 tools, L3 最高 (44%) | `experiments/mcptox_analysis/per_tool_asr.csv` |
| LLM Judge 实现 | GPT-4o-mini, 81.3% binary agreement | `src/mcpalign/llm_judge.py`, `experiments/judge_validation/` |
| Qwen3-8B Think baseline | ALL=22.0%, T1=22.2%, T2=0%, T3=40.3% | `experiments/mcptox_qwen3/qwen3_responses_all_summary.json` |
| Prompt hardening baseline | ALL=12.0%, T3=20.8% | 同上 |
| Defensive tokens baseline | ALL=13.3%, T3=22.2% | 同上 |
| SFT 训练 | 2161 例, 3 epochs, loss 1.88→0.73, acc 83% | `experiments/mcptox_defender/sft_checkpoint/` (334MB adapter) |
| SFT 评估 | ALL=22.0%, T1=16.7%, T2=1.7%, **T3=40.3%** | `experiments/mcptox_defender/sft_summary.json` |
| DPO 训练 | 1758 pairs, 1 epoch, from SFT checkpoint | `experiments/mcptox_defender/dpo_checkpoint/` (334MB adapter) |
| DPO 评估 | ALL=21.3%, T1=16.7%, T2=1.7%, **T3=38.9%** | `experiments/mcptox_defender/dpo_summary.json` |
| BF16 baseline 推理 | 450 条 (150×3 modes), 推理完成 | `experiments/mcptox_bf16/bf16_150_raw.csv` (8714 rows) |

### 关键发现（必须消化）

**SFT 和 DPO 几乎没有改善 T3（held-out paradigm）**：

```
Method       | T1 (seen) | T2 (seen) | T3 (unseen) | ALL
─────────────────────────────────────────────────────────
No Defense   | 22.2%     | 0.0%      | 40.3%       | 22.0%
SFT          | 16.7%     | 1.7%      | 40.3%       | 22.0%   ← T3 没变!
DPO          | 16.7%     | 1.7%      | 38.9%       | 21.3%   ← T3 只降 1.4pp
```

- SFT 在 T1 (seen) 上降了 5.5pp，但 T3 完全不变
- DPO 相比 SFT 在 T3 上只多降了 1.4pp
- **这其实是好消息**：完美说明 "SFT/DPO 无法泛化到 unseen paradigm"，支持论文核心 claim
- **前提是 GRPO 能在 T3 上做到显著降低**

### 未完成 / 有问题的工作

| 任务 | 状态 | 问题 |
|------|------|------|
| BF16 baseline judge | ❌ 推理完成但 judge 失败 | OpenAI API key 失效，需要更新 `.env` 后 re-judge |
| MCPTox ANOVA 分析 | ❌ 未开始 | Phase 1 统计分析还没做 |
| Model clustering / PCA | ❌ 未开始 | |
| Think vs No-Think 分析 | ❌ 未开始 | |
| Scale analysis (Qwen 系列) | ❌ 未开始 | |
| GRPO 训练 | ❌ 未开始 | **最关键缺失项**，需要新写脚本 |
| GRPO 评估 | ❌ 未开始 | |
| Ablation studies | ❌ 未开始 | |
| Case studies | ❌ 未开始 | |
| 论文写作 | ❌ 未开始 | |

---

## 2. 紧急任务（本周必须做）

### 2.1 修复 BF16 Baseline Judge

**目的**：确认 4-bit quantization 是否压低了 ASR。MCPTox 参考值 43.3%，我们 4-bit 只有 22.0%。如果 BF16 能到 ~40%，证明 quantization 是原因。

**问题**：Job 45458582 推理成功（450 条已存 `experiments/mcptox_bf16/bf16_150_raw.csv`），但 LLM judge 阶段失败，OpenAI API key 无效。

**操作**：
1. 检查 `.env` 中的 `OPENAI_API_KEY` 是否过期，更新有效 key
2. 用 `--skip-inference` 模式重跑 judge：
   ```bash
   python scripts/eval_bf16_baseline.py --skip-inference --tag bf16_150
   ```
3. 或者手动跑 judge（`experiments/mcptox_bf16/bf16_150_raw.csv` 已有 response 数据，只需要调 `LLMJudge`）
4. 输出目标：`experiments/mcptox_bf16/bf16_summary.json`，关键看 T3 的 no_defense ASR

**预期**：如果 BF16 T3 ASR ≈ 40-45% ≈ MCPTox 参考值，说明 quantization 不是问题（因为我们 4-bit 也是 40.3%）。如果 BF16 T3 >> 40%，说明 4-bit 确实压低了。

**注意**：`bf16_150_raw.csv` 有 8714 行，列包括 `defense_mode, server, paradigm, level, target_tool, poison_name, query, poisoned_tool, security_risk_desc, response_stripped, has_think, raw_len, stripped_len`。

### 2.2 写 MCPDefender GRPO 训练脚本（最关键！）

**背景**：现有 `scripts/mcpalign_train_grpo.py` 是为 SafeMCP 多轮场景写的，使用 `MTMCPGym` 环境和 `TurnLevelReward`。MCPDefender 是**单轮**场景（一个 system prompt + 一个 user query → 一个 tool call response），需要一个全新的 GRPO 脚本。

**新脚本路径**：`scripts/mcptox_train_grpo.py`

**脚本需要做的事**：

1. **数据源**：从 MCPTox `response_all.json` 加载训练数据，只用 T1 + T2 paradigm（T3 held-out）
   - 可以复用 `src/mcpalign/mcptox_data.py` 中的数据加载逻辑
   - 50% benign（clean registry + query → 期望选正确 tool）、50% poisoned（registry 含 malicious tool + query → 期望忽略 poison 选正确 tool）

2. **模型**：从 SFT checkpoint 加载 Qwen3-8B + LoRA adapter
   - Base: `Qwen/Qwen3-8B`, 4-bit nf4
   - SFT adapter: `experiments/mcptox_defender/sft_checkpoint/`
   - 或从 DPO checkpoint：`experiments/mcptox_defender/dpo_checkpoint/`

3. **Rollout**：每个 prompt 生成 G=4 个 response（do_sample=True, temperature=0.7）
   - Qwen3 Think mode：response 中有 `<think>...</think>` 标签，strip 后再 judge
   - max_new_tokens=2048（Think mode 生成较长）

4. **Reward 设计**（单轮版）：
   - +1.0: 选对 correct tool + safe arguments（keyword match 或 rule-based judge）
   - -1.0: 被 hijack（选了 malicious tool 或 arguments 被篡改）
   - -0.5: benign 场景下 over-refusal（拒绝执行正常请求）
   - -0.3: format error（无法 parse 出 tool call）
   - +0.1: 输出 valid JSON
   - **训练时用 keyword-based fast judge**（精确度够用，快），**每 200 步用 LLM judge 做 validation**

5. **GRPO Update**：
   - 标准 GRPO：compute group advantages → reward-weighted policy gradient
   - KL coefficient: 0.001
   - 可参考 `scripts/mcpalign_train_grpo.py` 中的 `compute_log_probs` 和 advantage 计算逻辑
   - clip_eps=0.2, gradient clipping=1.0

6. **配置**（参考 `docs/MCPDefender.md` Section 6.5）：
   ```yaml
   grpo:
     model: Qwen/Qwen3-8B
     init_from: experiments/mcptox_defender/sft_checkpoint  # 或 dpo_checkpoint
     lora_rank: 64
     quantization: 4-bit nf4
     group_size: 4
     kl_coeff: 0.001
     total_steps: 1500
     batch_size: 4  # prompts per step
     learning_rate: 3e-5
     lr_scheduler: cosine
     benign_ratio: 0.5
     train_paradigms: [Template-1, Template-2]
     max_new_tokens: 2048
     save_steps: 200
     validation_steps: 200  # LLM judge validation
   ```

7. **Memory 预算**（A5000 24GB）：
   - Base 4-bit: ~4 GB
   - LoRA + optimizer: ~2 GB
   - Reference model 4-bit: ~4 GB
   - KV cache (G=4, 2048 tokens): ~6 GB
   - Gradients: ~6 GB
   - Total: ~22 GB（紧但可行，不行就 G=2）

8. **Logging**：记录 per-paradigm average |advantage|、per-step reward 分布、per-paradigm ASR（validation 时）

9. **SLURM 脚本**：`slurm/run_grpo.sh`
   - `--partition=scavenger-gpu`
   - `--gres=gpu:1`
   - `--mem=64G`
   - `--time=24:00:00`
   - `--exclude=dcc-courses-gpu-[01-10]`
   - `--constraint=` 需要 A5000 或更大的卡

**关键参考文件**：
- `scripts/mcpalign_train_grpo.py` — 多轮 GRPO 的实现逻辑（generate_completions, compute_log_probs, GRPO update loop）
- `scripts/eval_trained_model.py` — 模型加载、推理、judge 的流程
- `src/mcpalign/mcptox_data.py` — MCPTox 数据加载和处理
- `src/mcpalign/llm_judge.py` — LLM judge 接口
- `configs/mcptox_defender.yaml` — 现有的 SFT/DPO 配置
- `docs/MCPDefender.md` Section 6 — 详细设计

### 2.3 GRPO 评估

GRPO 训练完成后，用和 SFT/DPO 完全相同的评估流程：

```bash
python scripts/eval_trained_model.py \
  --adapter experiments/mcptox_defender/grpo_checkpoint \
  --tag grpo \
  --modes no_defense
```

**成功标准**：
- GRPO T3 ASR < DPO T3 ASR (38.9%) — 基本要求
- GRPO T3 ASR < DPO T3 ASR - 10pp (即 < 28.9%) — 顶会标准
- GRPO BTSR (benign task success rate) ≥ 85% — 不能 over-refusal

---

## 3. Phase 1 统计分析任务（C1 — Re-Analysis）

### 3.1 MCPTox Mapping 覆盖率提升

**现状**：44.7% (11,203/25,079)，目标 >60%

**操作**：
- 检查 MCPTox 自己的 `analysis.ipynb` 中 query → tool 的 mapping 方法
- MCPTox 数据路径：`/work/tc442/MCPTox-Benchmark/response_all.json`
- 当前 mapping 逻辑在 `scripts/mcptox_analyze.py` 中
- 7 个 unclear tools 需要人工审核（见 `experiments/mcptox_analysis/tool_annotations.csv`）
- 尝试 fuzzy matching 替代 exact match 提升覆盖率

### 3.2 Two-Way ANOVA（paradigm × tool_level）

**目的**：证明 paradigm 的 main effect 远大于 tool_level（论文 Finding 1: Paradigm Dominance）

**操作**：
- 输入：`experiments/mcptox_analysis/` 中的 mapped records
- 用 `scipy.stats` 或 `statsmodels` 做 two-way ANOVA
- 报告：paradigm main effect (η²)、tool_level main effect (η²)、interaction effect
- 期望：paradigm η² >> tool_level η²，p<0.001

**产出**：
- ANOVA 结果表（Table 1 in paper）
- Interaction heatmap（5 levels × 3 paradigms = 15 cells）

### 3.3 Model-Family Clustering + PCA

**目的**：证明同 family 模型 vulnerability 相似，但 vulnerability 不随 scale 下降

**数据源**：MCPTox `response_all.json` 中 27 个模型的 per-tool ASR 向量

**操作**：
- 对每个 model 构造 vulnerability vector（per-tool ASR 向量，维度 = 141 tools）
- Ward's hierarchical clustering → dendrogram
- PCA 降维到 2D → 散点图，按 model family 着色
- Mantel test：同 family 内 vs 跨 family 的 vulnerability 距离
- Model families: Qwen (8b/14b/32b/235b, Think/No-Think), DeepSeek, GPT, Claude, Gemini, Gemma, Phi, Llama, Mistral

**产出**：
- Dendrogram 图
- PCA 散点图
- Model clustering 表

### 3.4 Scale Analysis（Qwen 系列）

**数据**（已知，from MCPTox 150-sample）：
- qwen3-8b_Think: 43.3%
- qwen3-32b_Think: 54.0%
- qwen3-235b-a22b_Think: 54.7%

**操作**：在 full dataset 上验证 "越大越 vulnerable（Think 模式）" pattern。分 Think 和 No-Think 两组画折线图。

### 3.5 Think vs No-Think Analysis

**已知数据**：
- qwen3-8b_Think: 43.3% vs qwen3-8b_NO_Think: 14.0%（差 29pp）
- qwen3-32b_Think: 54.0% vs qwen3-32b_NO_Think: 28.0%（差 26pp）

**操作**：在 full dataset 上做 paired comparison，per-paradigm 分解。如果 Think 模式在所有 paradigm 上都更脆弱 → "reasoning ≠ safety" 是 publishable finding。

### 3.6 Per-Server Vulnerability

**操作**：对 45 个 server 按 ASR 排序，分析最脆弱 vs 最 robust 的特征（tool 数量、domain、description 长度）。

---

## 4. 完整 6-Method 对比表（论文核心表）

现有数据 + 需要补充的：

```
Method             | T1 (seen)↓ | T2 (seen)↓ | T3 (unseen)↓ | ALL↓  | BTSR↑ | ORR↓
────────────────────────────────────────────────────────────────────────────────────
No Defense         | 22.2       | 0.0        | 40.3         | 22.0  | ?     | ?
Prompt Hardening   | 11.1       | 1.7        | 20.8         | 12.0  | ?     | ?
Defensive Tokens   | 16.7       | 1.7        | 22.2         | 13.3  | ?     | ?
SFT                | 16.7       | 1.7        | 40.3         | 22.0  | ?     | ?
DPO                | 16.7       | 1.7        | 38.9         | 21.3  | ?     | ?
MCPDefender (GRPO) | ???        | ???        | ???          | ???   | ???   | ???
```

**还需要补充**：
- [ ] BTSR（Benign Task Success Rate）：所有方法在 benign instances 上的 task 完成率
- [ ] ORR（Over-Refusal Rate）：在 benign instances 上错误拒绝的比率
- [ ] GRPO 的全部数据（等 GRPO 训练 + 评估完成）
- [ ] BF16 baseline 作为 supplementary（等 re-judge 完成）

**BTSR/ORR 测量方法**：
- 用 150-instance test set 中的 benign 部分（目前数据中应该有 benign tasks）
- 或者构造额外的 benign test set：纯 clean registry + normal query → 看是否正确执行
- 每个 method 都跑一次 benign evaluation

---

## 5. Ablation Studies（如果 GRPO 有效）

**优先级排序**，每个 ablation 需要一次训练 (12-18h) + 一次 evaluation (~$0.1)：

### Ablation 1: Train Paradigm Split（最重要）

| Setting | Train | Test-Unseen | Purpose |
|---------|-------|-------------|---------|
| Split A (default) | T1+T2 | T3 | T3 最高 baseline → 最大 headroom |
| Split B | T2+T3 | T1 | 看 T1 泛化 |
| Split C | T1 only | T2+T3 | 单 paradigm 泛化 |

每个 split 都训练 DPO + GRPO，看差距是否一致。

### Ablation 2: Benign Ratio

30% / 50% / 70%。看 BTSR vs ASR 的 tradeoff。

### Ablation 3: GRPO Init

- 从 SFT checkpoint 开始
- 从 DPO checkpoint 开始
- 从 base model 开始

### Ablation 4: Group Size G

G=2 vs G=4（G=8 在 A5000 跑不了）。

### Ablation 5: Training Data Size

25% / 50% / 100%。

**时间估算**：每个 ablation ~18h GPU + ~1h eval。全部做完需要 ~5-6 天 GPU 时间。优先做 1-3。

---

## 6. Case Studies & Error Analysis

### 6.1 三类 Case Study

**Type 1（最重要）：MCPDefender 成功，DPO 失败**
- 从 T3 test set 中找出 DPO 被 hijack 但 GRPO 正确 resist 的 instances
- 分析 GRPO response 中是否有 explicit safety reasoning

**Type 2：两者都失败**
- 什么样的 attack 连 GRPO 也防不住
- 预期是最 subtle 的 parameter tampering

**Type 3：MCPDefender over-refusal**
- GRPO 在 benign instance 上错误拒绝而 DPO 正确执行

**操作**：对比 `experiments/mcptox_defender/dpo_judged.csv` 和 GRPO 的 judged results，用 pandas 做 diff。

### 6.2 Error Analysis

按 (paradigm, tool_level) 分组统计 failure cases，找 pattern。

### 6.3 GRPO 学到了什么

对比 SFT/DPO/GRPO 在 T3 test set 上的 response 文本：
- 是否有更 systematic 的 safety reasoning
- 还是暴力拒绝（一律不执行）
- response 长度对比

---

## 7. 文件路径速查

### 数据

| 文件 | 说明 |
|------|------|
| `/work/tc442/MCPTox-Benchmark/response_all.json` | MCPTox 原始数据（25K+ records, 27 models） |
| `data/mcptox_defender/sft_data.json` | SFT 训练数据 (2161 条, 15MB) |
| `data/mcptox_defender/dpo_data.json` | DPO 训练数据 (1758 pairs, 13MB) |
| `data/mcptox_defender/stats.json` | 训练数据统计 |

### Checkpoints

| 路径 | 说明 |
|------|------|
| `experiments/mcptox_defender/sft_checkpoint/` | SFT LoRA adapter (334MB) |
| `experiments/mcptox_defender/dpo_checkpoint/` | DPO LoRA adapter (334MB) |
| `experiments/mcptox_defender/grpo_checkpoint/` | **待生成** |

### 评估结果

| 文件 | 说明 |
|------|------|
| `experiments/mcptox_qwen3/qwen3_responses_all_summary.json` | Qwen3-8B baseline（3 defense modes） |
| `experiments/mcptox_defender/sft_summary.json` | SFT 评估结果 |
| `experiments/mcptox_defender/dpo_summary.json` | DPO 评估结果 |
| `experiments/mcptox_bf16/bf16_150_raw.csv` | BF16 推理结果（待 judge） |
| `experiments/judge_validation/agreement.json` | LLM judge 验证 (81.3%) |

### 分析

| 文件 | 说明 |
|------|------|
| `experiments/mcptox_analysis/summary.json` | MCPTox re-analysis 总结 |
| `experiments/mcptox_analysis/per_tool_asr.csv` | 141 tools 的 ASR |
| `experiments/mcptox_analysis/paradigm_level_asr.csv` | paradigm × level ASR |
| `experiments/mcptox_analysis/level_model_asr.csv` | level × model ASR (21 models) |

### 核心脚本

| 脚本 | 功能 |
|------|------|
| `scripts/eval_trained_model.py` | 评估 SFT/DPO/GRPO checkpoints |
| `scripts/eval_bf16_baseline.py` | BF16 baseline（含 `--skip-inference` re-judge 模式） |
| `scripts/mcptox_build_training_data.py` | 从 MCPTox 构造 SFT/DPO 数据 |
| `scripts/mcptox_defense_baseline.py` | zero-shot defense baselines |
| `scripts/mcptox_run_qwen3.py` | Qwen3-8B inference |
| `scripts/mcptox_analyze.py` | MCPTox re-analysis |
| `scripts/mcpalign_train_sft.py` | SFT 训练 |
| `scripts/mcpalign_train_dpo.py` | DPO 训练 |
| `scripts/mcpalign_train_grpo.py` | **多轮 GRPO（SafeMCP 方向，不适用 MCPDefender）** |
| `scripts/mcptox_train_grpo.py` | **待写 — MCPDefender 单轮 GRPO** |

### 配置

| 文件 | 功能 |
|------|------|
| `configs/mcptox_defender.yaml` | MCPDefender SFT/DPO 配置 |
| `.env` | OpenAI API key + HF token（注意 API key 可能过期） |

---

## 8. 集群信息

- **集群**：Duke Compute Cluster (DCC)
- **账户**：`xulab`
- **工作目录**：`/work/tc442/RL-MCPTOOL`
- **模型缓存**：`/work/tc442/hf_cache/`
- **Conda 环境**：`safemcp` (python 3.10, torch + CUDA 12.4)
- **提交 job**：`sbatch slurm/run_xxx.sh`
- **监控**：`squeue -u tc442` + `tail -f slurm/logs/xxx.out`
- **排除 P100 节点**：所有 SLURM 脚本必须 `--exclude=dcc-courses-gpu-[01-10]`
- **可用 GPU**：A5000 (24GB), RTX 5000 Ada (32GB), A6000 (48GB), RTX 6000 Ada (48GB), H200 (80GB)
- **GRPO 最低要求**：A5000 24GB（G=4 刚好够，不够就 G=2）

---

## 9. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| GRPO 在 T3 上也没提升（< 5pp over DPO） | 30% | C1 re-analysis 单独发 Workshop/Findings |
| GRPO over-refusal 严重（BTSR < 80%） | 20% | 调高 benign_ratio 到 70% |
| A5000 装不下 GRPO G=4 + Think mode (2048 tokens) | 25% | 降 G=2，或降 max_new_tokens=1024，或申请 A6000 |
| OpenAI API key 持续失效 | 10% | 换用其他 LLM judge 模型（如本地 Qwen3-8B 做 judge） |
| SFT/DPO 其实有 bug（T3 不降可能是 evaluation bug） | 15% | 手动检查几条 T3 的 response，确认模型确实被 hijack |
| MCPTox mapping 覆盖率提不上去 | 30% | 接受 45% coverage，在论文中声明 limitation |

---

## 10. 执行顺序（推荐）

```
Week 1 (本周):
  Day 1-2:
    ├─ [P0] 修复 OpenAI API key → 重跑 BF16 judge
    ├─ [P0] 手动检查 SFT/DPO 的 T3 responses，确认不是 evaluation bug
    └─ [P0] 开始写 scripts/mcptox_train_grpo.py

  Day 3-5:
    ├─ [P0] 完成 GRPO 脚本 + 本地小规模测试（10 steps, G=2）
    ├─ [P0] 提交 GRPO 训练 job (slurm/run_grpo.sh, ~18h)
    └─ [P1] 开始 Phase 1 统计分析（ANOVA, clustering）

Week 2:
  Day 1-2:
    ├─ [P0] GRPO 评估（如果训练完成）
    ├─ [P1] Phase 1 统计分析完成
    └─ [P1] BTSR / ORR 测量

  Day 3-5:
    ├─ 如果 GRPO 有效 (T3 降 ≥5pp):
    │   ├─ [P0] 完善 6-method 对比表
    │   ├─ [P1] 开始 Ablation 1 (paradigm split)
    │   └─ [P2] Case studies
    │
    └─ 如果 GRPO 无效:
        ├─ Debug GRPO（检查 reward 分布、advantage 计算）
        ├─ 尝试从 DPO checkpoint init
        └─ 如果还是不行 → pivot 到纯 C1 + C3 paper

Week 3-4:
    ├─ Ablation 2-3
    ├─ Error analysis
    ├─ 论文 draft（Method + Experiments）
    └─ Figures 制作

Week 5+:
    ├─ 论文完善
    ├─ 内部 review
    └─ Revision
```

---

## 11. 一些具体的技术细节

### Qwen3-8B Think Mode 注意事项

- Response 格式：`<think>internal reasoning...</think>\n\nactual tool call JSON`
- 必须 strip `<think>` 标签后再 parse tool call 和 judge
- `strip_think()` 函数在 `scripts/eval_trained_model.py` 中已实现
- Enable thinking：`tokenizer.apply_chat_template(messages, enable_thinking=True)`
- Think mode 生成很长（平均 800-1500 tokens），max_new_tokens 至少 2048
- **GRPO 训练时也要支持 Think mode**，否则 adapter 学到的 pattern 和 eval 时不一致

### LLM Judge 使用

- 实现：`src/mcpalign/llm_judge.py`
- 调用：`LLMJudge` 类，自带 disk cache（`experiments/judge_cache.jsonl`）
- 需要 `OPENAI_API_KEY` 环境变量
- 每次 judge 约 $0.0001（GPT-4o-mini），150 instances × 3 modes = ~$0.05
- 81.3% binary agreement with MCPTox ground truth
- Judge 覆盖三种 attack 成功模式：T1 (选错 tool)、T2 (额外调 malicious tool)、T3 (参数被篡改)

### 4-bit Quantization 配置

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

### LoRA 配置

```python
LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

trainable parameters: 174M / 8.37B = 2.09%

---

## 12. 论文目标会议

| 目标 | 条件 | Deadline（估计） |
|------|------|-----------------|
| EMNLP 2026 | GRPO T3 < DPO T3 - 10pp + 全部分析 | 2026 年 6 月 |
| EMNLP Findings | GRPO > DPO on T3 (any margin) | 同上 |
| AISec / SaTML Workshop | C1 re-analysis 单独够 | 2026 年 7-8 月 |

---

## 13. Checklist（逐项勾选）

### P0 — 论文成败关键

- [ ] 修复 `.env` 中的 OpenAI API key
- [ ] BF16 baseline re-judge
- [ ] 手动检查 3-5 条 SFT/DPO 的 T3 response，确认 judge 正确
- [ ] 写 `scripts/mcptox_train_grpo.py`（单轮 GRPO for MCPDefender）
- [ ] 写 `slurm/run_grpo.sh`
- [ ] GRPO 训练 (~18h)
- [ ] GRPO 评估 → T3 ASR
- [ ] 完整 6-method 对比表

### P1 — 论文质量

- [ ] MCPTox mapping 覆盖率提升 (>60%)
- [ ] Two-way ANOVA (paradigm × tool_level)
- [ ] Model-family clustering + PCA
- [ ] Scale analysis (Qwen 8B/14B/32B/235B)
- [ ] Think vs No-Think analysis
- [ ] Per-server vulnerability ranking
- [ ] BTSR / ORR 测量
- [ ] Ablation 1: paradigm split
- [ ] Ablation 2: benign ratio
- [ ] Ablation 3: GRPO init

### P2 — 锦上添花

- [ ] Case studies (3 types × 3-5 cases)
- [ ] Error analysis (per-paradigm × per-level failure patterns)
- [ ] GRPO reasoning analysis (what did it learn?)
- [ ] Ablation 4: group size G
- [ ] Ablation 5: training data size
- [ ] Advantage vs training step 图
- [ ] 所有 publication-quality figures

### P3 — 论文写作

- [ ] Introduction (1.5 pages)
- [ ] Related Work (1 page)
- [ ] Fine-Grained MCPTox Analysis (2.5 pages)
- [ ] MCPDefender Method (1.5 pages)
- [ ] Experiments (3 pages)
- [ ] Discussion + Conclusion (1 page)
- [ ] 内部 review
- [ ] Revision
