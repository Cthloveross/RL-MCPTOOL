# MCPDefender 实验记录

**最后更新：2026-04-12**

---

## 1. 已验证 & 放弃的假设

| 假设 | 结果 | 影响 |
|------|------|------|
| "L1 (read-only) tools 比 L4 (execute) 更脆弱" | **FAIL**。MCPTox 上 L1=30.3% vs L4=34.0%，方向相反，Spearman≈0 | VulnGRPO 的 "automatic weakness targeting" 前提不成立，放弃 per-tool-type disparity story |
| Qwen2.5-7B 作为 victim model | **FAIL**。ASR=7.3% (LLM judge)，远低于 20% 阈值，无 defense 改进空间 | 切换到 Qwen3-8B |
| No-Think 模式 | **FAIL**。ASR=12.7%，仍然太低 | 切换到 Think 模式 |
| 直接用 MCPTox ground truth labels 做 no_defense baseline | **FAIL**。我们 22.0% vs MCPTox ref 43.3%，差 21pp（4-bit 量化 + judge 差异） | 必须用自己的推理 + judge 结果 |

**结论**：方向从 VulnGRPO (per-tool disparity) 转为 **MCPDefender** (paradigm-based adaptive defense，GRPO vs DPO cross-paradigm generalization)。

---

## 2. 当前实验设置

- **Victim model**: Qwen3-8B, Think mode, 4-bit nf4 quantization
- **Benchmark**: MCPTox (AAAI 2026), 150 stratified instances (seed=42)
- **Judge**: GPT-4o-mini LLM-as-judge, binary agreement 81.3% vs MCPTox ground truth
- **Train/Test split**: Train on T1+T2, Test on T3 (held-out)

---

## 3. Baseline ASR (Qwen3-8B Think, no training)

**Per-Defense × Per-Paradigm ASR**

| Defense | T1 (Explicit) | T2 (Implicit) | T3 (Param Tamper) | **ALL** |
|---|---|---|---|---|
| **no_defense** | 22.2 | 0.0 | **40.3** | **22.0** |
| prompt_hardening | 11.1 | 1.7 | 20.8 | 12.0 |
| defensive_tokens | 16.7 | 1.7 | 22.2 | 13.3 |

**Defense reduction (pp)**

| Defense | T1 | T3 | ALL |
|---|---|---|---|
| prompt_hardening | -11.1 | **-19.4** | **-10.0** |
| defensive_tokens | -5.6 | **-18.1** | **-8.7** |

**关键观察**：
1. T2=0% — Qwen3-8B Think 对 implicit hijacking 完全免疫，排除出训练
2. T3=40.3% — 最难防的攻击面，held-out test 的主战场
3. Prompt defenses 在 T3 上有效（-18~19pp），但 SFT/DPO/GRPO 应该能做更多

---

## 4. SFT 训练 (完成)

### 数据

| 类别 | 数量 | 来源 |
|---|---|---|
| poisoned (T1+T2) | 1286 | MCPTox malicious instances, correct response 从 FI model responses 提取 |
| benign | 875 | MCPTox clean queries, 含 prompt_hardening/defensive_tokens 增广 |
| **Total** | **2161** | T3 完全排除 |

### 训练

| 参数 | 值 |
|---|---|
| GPU | RTX 5000 Ada 32GB |
| LoRA | r=64, alpha=128 (2.09% trainable = 174M/8.37B) |
| lr / scheduler | 5e-6 / cosine + 10% warmup |
| Epochs / Runtime | 3 / **2h 18min** |

### 训练曲线

| Epoch | Loss | Token Accuracy |
|---|---|---|
| 0 | 1.88 | 67% |
| 1 | 0.95 | 78% |
| 2 | 0.81 | 82% |
| **3** | **0.73** | **83%** |

稳定下降，无 collapse。Checkpoint: `experiments/mcptox_defender/sft_checkpoint/`

### **还未 evaluate ASR** — 需要加载 adapter 跑 Think mode 推理 + LLM judge

---

## 5. DPO 训练 (排队中)

1758 preference pairs (1286 poisoned + 472 benign)。Rejected 来源：60.6% real model Success responses, 39.4% synthetic。

从 SFT checkpoint 继续，lr=5e-7, beta=0.1, 1 epoch。Job 45409787 排队中。

---

## 6. 这些结果有意义吗？

### 能做的论文 story

**MCPDefender 的核心论点**：DPO 只在 seen paradigms (T1+T2) 上训练，对 unseen paradigm (T3) 泛化差；GRPO 通过 online exploration 自动发现 T3 pattern，泛化更好。

**数据支撑**：
- no_defense T3=40.3% → 有 defense headroom
- prompt_hardening 已经把 T3 砍到 20.8%（-19pp）→ 证明 T3 是 **可防的**
- 如果 GRPO T3 < DPO T3 by ≥10pp → 核心 claim 成立

### 风险

| 风险 | 严重度 | 说明 |
|---|---|---|
| T3 baseline 40% 不够高 | 中 | MCPTox ref 是 68%，我们低了 28pp。但所有 methods 用同一 setup，相对差距有效 |
| T2=0% 没有训练信号 | 低 | T2 被排除，训练实际只有 T1 信号 (22.2%)。T1 样本少 (127 base) |
| SFT/DPO 可能就把 T3 降到很低 | 高 | 如果 DPO 已经把 T3 降到 <15%，GRPO 没有 further improvement space → 论文核心 claim 不成立 |
| 22% baseline 太低 | 中 | 绝对数字小，每个 paradigm 只有 50 test instances，1-2 个 flip 就是 2-4pp |

### 底线判断

**能继续**。22% baseline + 40% T3 + prompt defense -19pp 的组合说明：
1. 攻击是 real 的（22% 的 instances 被骗）
2. Defense 有效（prompt 就能砍半）
3. T3 是最有价值的 test case（40% 且 training 没见过）

**关键实验**：SFT/DPO evaluate 后如果 T3 ASR 仍 >20%，GRPO 就有 room to improve。如果 DPO 已经把 T3 降到 <10%，实验意义存疑。

---

## 7. 下一步

1. ⏳ DPO 训练完成
2. **Evaluate SFT + DPO** on 150 instances × Think mode → LLM judge → per-paradigm ASR
3. 如果 DPO T3 >20% → 进入 GRPO 训练
4. 6-method comparison table (No Defense / Prompt Hard. / Def. Tokens / SFT / DPO / GRPO)

---

## 附录：关键文件

| 文件 | 用途 |
|---|---|
| `src/mcpalign/llm_judge.py` | GPT-4o-mini judge (81.3% binary agreement) |
| `src/mcpalign/mcptox_data.py` | SFT/DPO 数据构造 |
| `experiments/mcptox_qwen3/qwen3_responses_all_judged.csv` | Qwen3-8B Think baseline 450 rows |
| `experiments/mcptox_defender/sft_checkpoint/` | SFT LoRA adapter (349MB) |
| `data/mcptox_defender/{sft_data.json, dpo_data.json}` | 训练数据 |
| `configs/mcptox_defender.yaml` | 训练配置 |
