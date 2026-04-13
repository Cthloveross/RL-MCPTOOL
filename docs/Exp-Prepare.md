# MCPDefender 实验执行手册

**最后更新：2026-04-11**

---

## 0. 项目现状（pivot 历史简述）

- ❌ **Multi-turn SafeMCP**（ΔASR 随 step 递增）：自建数据上 ΔASR = -38.9%，假设被 falsify
- ❌ **VulnGRPO**（low-risk tools 更脆弱）：MCPTox 上 L1 ASR (30.3%) 反而 < L4 ASR (34.0%)，Spearman = +0.052，假设被 falsify
- ✅ **MCPDefender**（GRPO 泛化到 unseen attack paradigm）：当前方向

Pivot 的驱动力：前两个方向都试图找 "一个 novel empirical finding 来驱动 RL defense"，现在策略变了——**re-analysis 本身是独立 contribution，RL 的卖点是 generalization 而不是 profiling**。

---

## 1. 当前资产

| 资产 | 状态 | 说明 |
|------|------|------|
| MCPTox response_all.json | ✅ | 25K+ records, 20 models, 1348 malicious instances |
| MCPTox Re-Analysis v2 | ✅ | 11,203 mapped records, per-paradigm × per-level ASR |
| Tool Taxonomy (148 tools, L1-L5) | ✅ | 自动标注 + 7 unclear 待人工审核 |
| 自建 Profiling (Qwen-7B, 2700 trials) | ✅ | 作为 supplementary evidence |
| MCP-Gym 环境（单轮）| ✅ | 可复用于 GRPO 训练 |
| SFT / DPO 训练脚本 | ✅ | scripts/vulngrpo_mini.py（TRL 1.0.0 API 已适配）|
| Qwen2.5-7B-Instruct | ✅ | 已缓存，`/work/tc442/hf_cache/` |

---

## 2. 论文结构（MCPDefender）

Title：**Beyond Average ASR: Fine-Grained Vulnerability Analysis and RL-Based Defense for MCP Tool Poisoning**

### 两个 Contribution

**C1（Fine-Grained Re-Analysis — 稳赢）**：第一个在 MCPTox 上做 paradigm × level × model 三维分解的分析。

已确认的 3 个 findings：
1. **Inverse Risk Paradox**：L3 Communicate (44.0%) 最脆弱，而非 L1 Read-Only
2. **Paradigm Dominance**：P3 Parameter Tampering (29-51%) 比 P1 Explicit Hijacking (3-24%) 有效 2-5 倍
3. **Model-Family Divergence**：有的模型 L4 > L1 by 17-20pp，有的 L1 ≈ L4，vulnerability pattern 模型间高度不一致

**C2（MCPDefender — 有 upside）**：GRPO 训练的 defense，**核心实验是 cross-paradigm generalization（train on P1+P2, test on P3）**。

核心假设：DPO 只学到 seen paradigm 的 preference pattern，无法泛化到 unseen P3 parameter tampering。GRPO 通过 online exploration 学到通用的 safety reasoning，能泛化。

---

## 3. 实验 Phase 划分

### Phase 1：MCPTox Re-Analysis 完善（Week 1-2）

**3.1 数据修复**
- 当前 mapping rate 44.7%，需提升到 60%+
- 人工审核 7 个 unclear tools
- 按 MCPTox 自己的 `analysis.ipynb` 过滤 `wrong_data==1`

**3.2 统计分析**
- Three-way ANOVA（paradigm × level × model_family）
- Cohen's d (P1 vs P3)
- Model-family Ward's hierarchical clustering + PCA
- Per-server vulnerability ranking

**3.3 产出**
- Table 1: Per-Level ASR
- Table 2: Paradigm × Level ASR
- Table 3: Per-Model L1-L4 Gap
- Figure 1: Interaction plot
- Figure 2: Model clustering dendrogram
- Figure 3: Per-server heatmap

### Phase 1.5：Zero-Shot Defense Baseline（今晚，1-2 小时）

**目的**：在投入 GRPO 训练之前，先建立 baseline——测量 Qwen-7B 在 MCPTox 攻击下的 vulnerability，并验证 prompt-based defenses 的效果上限。

**设计**：
- 从 MCPTox 数据中 stratified 采样 ~150 instances（跨 45 servers × 3 paradigms × 5 levels）
- 3 个 defense 条件：
  1. **no_defense**：直接用 MCPTox 原始 system prompt（含 poisoned tool）
  2. **prompt_hardening**：加安全警告到 system prompt 顶部
  3. **defensive_tokens**：在 tool list 前加 "[DEFENSIVE BOUNDARY]" 标记
- 每个 instance × 每个条件 = 450 次 inference
- Qwen2.5-7B-Instruct 4-bit，greedy decoding
- 记录：per-paradigm ASR、per-level ASR、parse rate

**关键看点**：
1. 如果 `prompt_hardening` 就把 P3 ASR 降到 <20% → MCPDefender 没有空间做，可能需要换方向
2. 如果 `defensive_tokens` 对 P1/P2 有效但对 P3 无效 → 验证 proposal.tex 的核心假设（prompt defenses 不能 generalize 到 P3）
3. 如果所有 prompt-based defenses 都没什么用 → GRPO 有充足提升空间

**脚本**：`scripts/mcptox_defense_baseline.py`
**配置**：`configs/mcptox_defense.yaml`
**SLURM**：`slurm/run_mcptox_defense.sh`

**预计运行时间**：~22 分钟（inference）+ 5 分钟（模型加载）≈ 30 分钟

**Go/No-Go 判定**：
- 如果 P3 no_defense ASR > 40% AND prompt defenses 没把 P3 降到 < 30% → 继续 Phase 2 MCPDefender
- 如果 prompt defenses 已经把 P3 降到 < 25% → 重新评估（可能需要更强攻击或换方向）

### Phase 2：MCPDefender 训练（Week 3-5）

**前提**：Phase 1.5 显示 prompt defenses 对 P3 不够有效。

**2.1 训练数据构造**
- 全部来自 MCPTox（不自建）
- 只用 P1 + P2（Template-1, Template-2）
- P3 完全 held-out for evaluation

**2.2 Pipeline**
```
SFT warm-start (~2h, 600 examples)
  ↓
DPO baseline (~1h, 600 pairs, P1+P2 only)
  ↓
GRPO training (~12-18h, 1500 steps, G=4)
  - Environment: MCPTox registries with P1+P2 poisons
  - Uniform paradigm sampling（让 GRPO 自己发现 P1 vs P2 的难度差异）
  - Log per-paradigm advantage magnitude
```

**2.3 Memory 预算（A5000 24GB）**
- Base 4-bit: ~4GB
- LoRA + optimizer: ~2GB
- Reference model: ~4GB
- KV cache (G=4): ~6GB
- Gradients: ~6GB
- **Total: ~22GB**（紧但可行）

### Phase 3：Evaluation（Week 6-7）

**核心评估表**：

| Method | Seen (P1+P2)↓ | **Unseen (P3)↓** | All↓ | BTSR↑ | ORR↓ |
|--------|:---:|:---:|:---:|:---:|:---:|
| No Defense | ~18 | ~45 | ~28 | ~92 | ~2 |
| Prompt Hardening | ? | ? | ? | ? | ? |
| DefensiveTokens | ? | ? | ? | ? | ? |
| SFT | ~14 | ~42 | ~26 | ~90 | ~5 |
| DPO | ~10 | ~40 | ~22 | ~88 | ~7 |
| **MCPDefender** | **~8** | **~28** | **~16** | **~89** | **~6** |

**Unseen (P3) 列是论文成败的关键。**

**成功标准**：
- EMNLP/CCS 顶会：MCPDefender Unseen-P3 ASR < DPO Unseen-P3 ASR − 10pp
- Findings：任何 MCPDefender > DPO on P3 的 margin
- Workshop：Re-analysis 单独发

---

## 4. 今晚的具体执行

### 已提交的 job

```
Job ID: 45351805 (scavenger-gpu)
Script: scripts/mcptox_defense_baseline.py
Config: configs/mcptox_defense.yaml
```

### 监控

```bash
squeue -u tc442
tail -f slurm/logs/mcptox_def_45351805.out
```

### 结果文件

```
experiments/mcptox_defense_baseline/
├── defense_baseline_raw.csv      # 所有 trials
├── per_paradigm_asr.csv          # 3 defenses × 3 paradigms pivot
├── per_level_asr.csv             # 3 defenses × 5 levels pivot
└── summary.json                  # 汇总
```

### 结果分析后的下一步

1. 如果 P3 no_defense ASR > 40% 且 defenses 没降到 <30% → 立刻开始 Phase 2 MCPDefender 数据构造
2. 如果 defenses 太有效 → 讨论是不是换更难的攻击 subset，或者调整 scope
3. 无论如何，更新 temp-result.md 记录 baseline 数据

---

## 5. 下一步（Phase 2 启动条件）

Phase 1.5 Go 之后立刻做的事：
1. 写 `scripts/mcptox_grpo_train.py`——从 MCPTox 构造 GRPO episodes（P1+P2 only）
2. 写 SFT + DPO 脚本（可复用 `scripts/vulngrpo_mini.py` 的部分）
3. 在 A5000 24GB 上跑 SFT warm-start
4. 然后 DPO baseline
5. 最后 GRPO

---

## 6. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| Phase 1.5 显示 prompt hardening 就能降 P3 < 30% | 20% | 换更难的攻击 subset 或换模型 |
| GRPO 在 unseen P3 上差距 < 5pp | 30% | Re-analysis + baselines 发 Findings |
| GRPO unseen-P3 差距 5-10pp | 35% | 可发，加上 re-analysis |
| GRPO unseen-P3 差距 > 10pp | 15% | 强结果，投 EMNLP/CCS |
| A5000 24GB 装不下 GRPO G=4 | 15% | G=2 或换 RTX 5000 Ada 32GB |
| MCPTox mapping 提升无望 | 30% | 用当前 45% coverage 发 paper，声明 limitation |
