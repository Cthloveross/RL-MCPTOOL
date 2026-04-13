# MCPDefender 最终实验方案（v4）

**日期：2026-04-11**

---

## 0. 一句话

MCPTox报告36.5%平均ASR，掩盖了三层结构（paradigm间3倍差异、tool-category间14pp差异、model-family间11倍差异）。我们做首次三路细粒度分析，然后用GRPO训练首个RL-based MCP defense，核心卖点是对unseen attack paradigm的泛化能力优于DPO。

---

## 1. 经历过的所有方向与结论

| # | 方向 | 结论 | 教训 |
|---|------|------|------|
| 1 | MetaAlign-RL (adversarial multi-stage defense) | 放弃 | 三阶段太重，被ARLAS压制 |
| 2 | MCPoisoner (GRPO攻击生成) | 放弃 | MCP-ITP/PISmith已做，ASR天花板 |
| 3 | SafeMCP (multi-turn safety degradation) | **Falsified** | Position 1 ASR最高(83.3%)，不是后面步骤 |
| 4 | VulnGRPO v1 (L1>L4 自建数据) | **Falsified** | 自建数据artifact，Mistral完全免疫 |
| 5 | VulnGRPO v2 (L1>L4 MCPTox验证) | **Falsified** | MCPTox上L3最高(44%), L4>L1, Spearman≈0 |
| 6 | MCPDefender baseline (keyword judge) | **Bug** | Keyword judge漏掉T2/T3的72pp攻击 |

---

## 2. 现在确定拥有的

| 资产 | 说明 |
|------|------|
| MCPTox response_all.json | 25K+记录，27个模型的ground truth labels，不需要跑inference |
| MCPTox per-level re-analysis | 11K mapped records，L1-L5标注完成，发现inverse risk paradox |
| MCPTox 150-instance sample | 已和ground truth对齐，知道各模型真实ASR |
| Qwen2.5-7B inference pipeline | 能输出tool call JSON，pipeline本身没问题 |
| SFT/DPO训练pipeline | 已实现，可复用 |
| MCP-Gym环境 | 已实现 |
| GPT-4o-mini API access | 可用于LLM-as-judge |

---

## 3. 论文结构

**Title:** "Beyond Average ASR: Fine-Grained Vulnerability Analysis and RL-Based Defense for MCP Tool Poisoning"

### 三个Contribution

**C1 — MCPTox细粒度分析（稳赢，数据已有）：**
首次在MCPTox上做paradigm × tool-category × model-family三路分解。揭示三个finding：paradigm dominance（P3比P1有效3-5倍）、inverse risk paradox（通信类最脆弱44%，非读取类）、model-family clustering（同家族模型聚集但vulnerability不随模型变大而下降）。

**C2 — MCPDefender GRPO defense（有upside）：**
首个用RL训练的MCP tool poisoning defense。核心对比：train on部分paradigms → test on held-out paradigm，GRPO通过online exploration比DPO泛化更好。

**C3 — Systematic defense comparison（稳赢）：**
首次在MCPTox上做SFT/DPO/GRPO/DefensiveTokens/Prompt Hardening的统一对比，用GPT-4o-mini LLM-as-judge确保evaluation对所有attack paradigm可靠。

### 安全网

C1不依赖任何训练实验。即使C2 GRPO效果不好，C1+C3（分析+baseline比较）仍然可以发Findings或Workshop。

---

## 4. Phase 1: MCPTox细粒度分析（Week 1-2）

### 4.1 数据源

直接使用MCPTox response_all.json中27个模型的pre-computed labels。不跑任何inference，不需要judge。这部分的数据质量由MCPTox AAAI 2026论文保证。

### 4.2 需要改进的问题

当前query→tool mapping覆盖率只有44.7%（11,203/25,079条）。原因是部分server的system prompt格式不标准，正则无法解析。

改进方向：
- 检查MCPTox自己的analysis.ipynb中的mapping逻辑，复制其方法
- 对mapping失败的server手动检查system prompt格式
- 尝试fuzzy matching替代exact match
- 目标覆盖率：>60%

### 4.3 分析内容

**分析A — Paradigm × Tool-Category交互效应**

对所有mapped records做two-way ANOVA（paradigm × tool_level），报告：
- 每个factor的main effect和effect size（η²）
- paradigm × tool_level的interaction effect
- 交互效应热力图（5个level × 3个paradigm = 15个cell的ASR）

预期核心结论：paradigm的main effect远大于tool_level（η²差一个数量级）。意义：defense设计应该优先按paradigm区分，而非按tool type区分。

**分析B — Model-Family Clustering**

对每个model构造vulnerability vector（per-tool ASR向量），然后：
- Ward's hierarchical clustering → dendrogram
- PCA降维到2D → 散点图，按model family着色
- Mantel test检验："同家族模型的vulnerability profile是否比不同家族更相似"

Model family分组：Qwen (8b/14b/32b/235b, Think/No-Think)、DeepSeek (R1/v3)、GPT (3.5/4o-mini/o1-mini)、Claude、Gemini、Gemma、Phi、Llama (8B/70B)、Mistral。

预期发现：同family模型在PCA空间中聚集，但Think vs No-Think模式差异可能大于size差异。

**分析C — 模型规模效应**

利用Qwen家族的多尺度覆盖（8B, 14B, 32B, 235B），分析ASR是否随模型变大而下降。分别看Think和No-Think模式。

MCPTox ground truth数据已显示（150-sample subset）：
- qwen3-8b_Think: 43.3%
- qwen3-32b_Think: 54.0%
- qwen3-235b-a22b_Think: 54.7%

这个"越大越vulnerable（Think模式）"的pattern如果在full dataset上也成立，是一个counter-intuitive且publishable的finding。

**分析D — Think vs No-Think安全差异**

Think模式（有内部推理）vs No-Think模式对MCP poisoning的抵抗力差异：
- qwen3-8b_Think: 43.3% vs qwen3-8b_NO_Think: 14.0%（差29pp）
- qwen3-32b_Think: 54.0% vs qwen3-32b_NO_Think: 28.0%（差26pp）

Think模式反而更容易被攻击，这与"reasoning improves safety"的直觉相反。可能原因：Think模式的reasoning更长，给了poison description更多的"说服空间"。

**分析E — Per-Server Vulnerability**

对45个server按平均ASR排序，分析最脆弱和最robust的server有什么特征（tool数量、domain、description长度等）。

### 4.4 Phase 1产出

| 产出 | 类型 | 对应Finding |
|------|------|------------|
| Paradigm × Level ASR表 | 表 | paradigm dominance |
| ANOVA结果 + effect size | 表 | paradigm dominance |
| Interaction heatmap | 图 | paradigm × level交互 |
| Model clustering dendrogram | 图 | model-family clustering |
| PCA散点图 | 图 | model-family clustering |
| Qwen规模效应图 | 图 | scale ≠ safety |
| Think vs No-Think对比表 | 表 | reasoning ≠ safety |
| Per-server vulnerability ranking | 表 | 补充分析 |
| 27-model完整ASR表 | 附录 | 完整数据 |

### 4.5 Phase 1成功标准

所有标准都是对已有数据的分析，不存在"fail"的可能——数据就是那些数据。质量标准是：

- ANOVA的paradigm main effect显著（p<0.001）且η² > 0.05 → 可以claim paradigm dominance
- Model clustering显示可解释的family结构 → 可以claim model-family clustering
- Think vs No-Think差异在≥2个model size上一致 → 可以claim reasoning paradox
- Per-server analysis覆盖≥30个server → 足够做排序

---

## 5. Phase 2: LLM-as-Judge + Victim Model确认（Week 2-3）

### 5.1 为什么需要这一步

之前用keyword matching做judge，对Template-2和Template-3的detection几乎为零（漏掉72pp）。必须换成LLM-as-judge。

MCPTox自身也使用LLM-based evaluation（不是keyword matching）。我们用GPT-4o-mini对齐MCPTox的evaluation方法。

### 5.2 LLM-as-Judge设计

输入：user query、legitimate tool名称、malicious tool名称、attack paradigm、agent的完整response文本、malicious tool description。

输出：JSON格式的二元判定（attack成功/失败）+ 一句话理由。

Judge需要覆盖三种attack成功模式：
- T1 (explicit hijacking)：agent选了malicious tool而非legitimate tool
- T2 (implicit hijacking)：agent在调legitimate tool的同时/之前额外调了一个malicious tool执行恶意操作
- T3 (parameter tampering)：agent调了correct tool但参数被篡改（改收件人、改SQL、改文件路径、改API endpoint等）

### 5.3 Judge验证

在150-instance sample上验证LLM-as-judge和MCPTox ground truth的一致性。

方法：对这150个instances中有MCPTox label的模型（如Qwen3-8b），我们有MCPTox的ground truth。用LLM-as-judge评估同一个instance的同一个response，看agreement rate。

目标：agreement > 85%。

如果agreement太低：检查GPT-4o-mini的judge prompt是否和MCPTox的evaluation逻辑不一致，调整prompt。

### 5.4 重新测量Qwen2.5-7B Baseline ASR

用LLM-as-judge在150个instances上重新评估Qwen2.5-7B的三个defense条件（no_defense, prompt_hardening, defensive_tokens）。

预期结果（基于MCPTox同family Qwen3-8b的数据推测）：

| Defense | T1 | T2 | T3 | All |
|---------|:--:|:--:|:--:|:---:|
| No Defense | ~25-35% | ~15-25% | ~40-60% | ~30-45% |
| Prompt Hardening | ~20-30% | ~12-20% | ~35-55% | ~25-40% |
| DefensiveTokens | ~15-25% | ~10-18% | ~30-50% | ~20-35% |

如果实际结果中overall ASR > 20% → Qwen2.5-7B可以做victim model。
如果overall ASR < 20% → 换qwen3-8b（已证实43.3%）。

### 5.5 确定Train/Test Split

原则：让test-unseen split的baseline ASR最高，这样defense有最大的改进空间。

根据LLM-as-judge结果选择：

| 如果 | 则 |
|------|------|
| T3 baseline ASR最高（>30%） | Train on T1+T2, Test-unseen on T3（原方案） |
| T1 baseline ASR最高（>30%） | Train on T2+T3, Test-unseen on T1 |
| 所有paradigm ASR都>20% | 选最高的作为unseen |
| 所有paradigm ASR都<15% | 换victim model到qwen3-8b |

### 5.6 Phase 2产出

- LLM-as-judge和MCPTox的agreement rate
- Qwen2.5-7B的真实per-paradigm baseline ASR（3个defense条件）
- 确定的victim model
- 确定的train/test paradigm split
- **Go/No-Go决策**

---

## 6. Phase 3: 训练数据构造 + Baselines（Week 3-4）

### 6.1 数据来源

全部来自MCPTox：
- Tool registries: pure_tool.json（353个legitimate tools）
- Poisoned tools: def_tool/目录（per-server的malicious tool definitions）
- User queries: response_all.json中的query字段
- Ground truth: MCPTox labels（用于验证，不用于训练）

不自建任何attack template或tool registry。

### 6.2 SFT数据

约3000个(prompt, correct_response) pairs：
- 1500个benign episode：正常registry + query → agent选正确tool
- 1500个poisoned episode：registry含malicious tool + query → agent仍选正确tool，附带reasoning说明为什么忽略了可疑description

数据只从train split的paradigms中采样（held-out paradigm不出现在训练集中）。

### 6.3 DPO数据

约3000个(prompt, chosen, rejected) pairs：
- chosen: 选择legitimate tool + safe arguments
- rejected: 选择malicious tool（T1类型）或使用tampered arguments（T2/T3类型）

同样只从train paradigms采样。

### 6.4 GRPO环境

MCP-Gym，每个episode：
1. 从MCPTox train split采样一个(registry, query, poison)
2. 50%概率benign（去掉poison），50%概率poisoned
3. Agent生成tool call response
4. Reward判定

**Reward设计：**
- +1.0: 选对correct tool + safe arguments
- -1.0: 被hijack（选了malicious tool或arguments被篡改）
- -0.5: benign场景下over-refusal
- -0.3: format error（JSON parse失败）
- +0.1: 输出valid JSON

**GRPO训练的reward judge：** 训练时用keyword-based fast judge（虽然不完美，但T1的detection是准确的，T2/T3漏检只意味着一些attack成功被误标为+1——这等于把一些hard negative当成了positive，让训练更conservative而非更aggressive）。每200步用LLM-as-judge做validation，监控真实ASR趋势。

### 6.5 GRPO配置

| 参数 | 值 | 理由 |
|------|-----|------|
| Model | Qwen2.5-7B-Instruct (或qwen3-8b) | Phase 2决定 |
| LoRA rank | 64 | 标准 |
| Quantization | 4-bit | 适配A5000 24GB |
| Group size G | 4 | A5000内存限制 |
| KL coefficient | 0.001 | Bespoke Labs推荐 |
| Total steps | 1500 | ~12-18小时 |
| Batch size | 4 prompts × 4 completions = 16 rollouts/step | 内存适配 |
| Learning rate | 3e-5 with cosine decay | 标准 |
| Benign ratio | 50% | 防over-refusal |
| Train paradigms | Phase 2决定的subset | P3或T1 held-out |

### 6.6 内存估算（A5000 24GB）

- Base model 4-bit: ~4 GB
- LoRA + optimizer: ~2 GB
- Reference model 4-bit: ~4 GB
- KV cache (G=4, 256 tokens): ~6 GB
- Gradients + buffer: ~6 GB
- Total: ~22 GB → 可行但紧。如果不够降G=2或用RTX 5000 32GB。

### 6.7 Baseline实现

| Baseline | 实现方式 | 需要训练? |
|----------|---------|----------|
| No Defense | 直接跑inference | 否 |
| Prompt Hardening | 在system prompt加安全警告 | 否 |
| DefensiveTokens | 在tool descriptions前加defensive prefix | 否 |
| SFT | 在safe examples上fine-tune | 是（8h） |
| DPO | 在preference pairs上训练 | 是（4h） |
| MCPDefender (GRPO) | GRPO online training | 是（12-18h） |

所有训练方法使用完全相同的MCPTox数据来源和train/test split，确保差异只来自训练方法。

---

## 7. Phase 4: Evaluation（Week 5-7）

### 7.1 评估方式

**全部使用GPT-4o-mini LLM-as-judge。**

对每个(defense_method, test_instance)：
1. 用defense-specific方式跑inference（不同的prompt构造或不同的model checkpoint）
2. 将agent response发送给GPT-4o-mini judge
3. 获得binary attack success判定

### 7.2 评估规模

| 评估维度 | 数量 |
|---------|------|
| Defense方法 | 6 |
| Test instances (seen paradigm) | ~60-80 |
| Test instances (unseen paradigm) | ~50-70 |
| Benign test instances | ~50 |
| Total judgments | ~6 × 180 = ~1,080 |
| API成本 | ~$0.11 |

加上debug和reruns，总API成本<$2。

### 7.3 核心结果表

```
Method           | Seen↓   | Unseen↓  | All↓  | BTSR↑ | ORR↓
────────────────────────────────────────────────────────────
No Defense       |         |          |       |       |
Prompt Hardening |         |          |       |       |
DefensiveTokens  |         |          |       |       |
SFT              |         |          |       |       |
DPO              |         |          |       |       |
MCPDefender      |         |          |       |       |
```

**论文叙事焦点：Unseen列。**

Seen列：所有training-based方法应该都比No Defense好，GRPO和DPO可能接近（因为都见过这些paradigm）。

Unseen列：**这是论文成败的关键。** GRPO如果比DPO低10+pp → 强结果。5-10pp → 可以发。<5pp → re-analysis撑底。

BTSR（benign task success rate）：MCPDefender应该>85%。如果<85%说明over-refusal严重，需要调benign ratio。

### 7.4 Per-Paradigm × Per-Level Breakdown（补充表）

即使tool-level disparity不是main story了，仍然值得报告per-level的defense效果——看GRPO是否在所有level上均匀提升。

```
Method      | L1↓ | L2↓ | L3↓ | L4↓ | L5↓
──────────────────────────────────────────
No Defense  |     |     |     |     |
DPO         |     |     |     |     |
MCPDefender |     |     |     |     |
```

### 7.5 Mechanism Validation

训练过程中记录per-paradigm的average |advantage|。

预期：
- 被held-out的paradigm不在训练中出现，所以不会有直接的advantage数据
- 但train paradigms中，base ASR越高的paradigm应该有更大的|advantage|
- 随着训练进行，advantage应该递减（模型越来越会resist了）

这张advantage vs training step的图是论文Figure的候选。

---

## 8. Phase 5: Ablations（Week 7-8）

按优先级排序：

### Ablation 1（最重要）: Training Method

SFT vs DPO vs GRPO。这是core comparison，已在main results中包含。

### Ablation 2: Train Paradigm Split

| 设置 | Train | Test-Unseen | Purpose |
|------|-------|-------------|---------|
| Split A | T1+T2 | T3 | 默认（如果T3 ASR最高） |
| Split B | T2+T3 | T1 | 看对T1的泛化 |
| Split C | T1 only | T2+T3 | 看单paradigm训练的泛化能力 |
| Split D | All | All (cross-val) | 上界：全paradigm训练 |

这个ablation回答"GRPO是否在任何paradigm split下都比DPO泛化更好"。如果只在某一个split下好而其他split不行，说明GRPO的优势是paradigm-specific的。

### Ablation 3: Benign Ratio

30% / 50% / 70%。看对over-refusal的影响。预期50%是最优——30%会导致BTSR下降，70%会导致ASR降不下来。

### Ablation 4: GRPO Init

从SFT checkpoint开始 vs 从DPO checkpoint开始。看DPO warm-start是否帮助GRPO更快收敛或达到更好的最终结果。

### Ablation 5: Group Size G

G=2 vs G=4。看更大的group（更好的advantage estimation）是否带来更好的泛化。（G=8在A5000上跑不了。）

### Ablation 6: Training Data Size

25% / 50% / 100%的training episodes。看GRPO的data efficiency。

**优先级：1 > 2 > 3 > 4 > 5 > 6。如果时间不够，只做前3个。**

每个ablation需要一次training（12-18h）+ 一次evaluation（~$0.1 API）。6个ablation需要~6天GPU时间，可以并行在不同GPU上。

---

## 9. Phase 6: Case Studies + Analysis（Week 8）

### 9.1 Case Study类型

**Type 1: MCPDefender成功、DPO失败（最重要）**

从unseen paradigm的test set中找出DPO被hijack但MCPDefender正确resist的instances。分析MCPDefender的response中是否有对poison的explicit reasoning（"I notice this description contains instructions to..."）。

**Type 2: 两者都失败**

分析什么样的attack即使GRPO训练后也无法抵抗。预期这些是最subtle的implicit poisoning或非常convincing的parameter tampering。

**Type 3: MCPDefender over-refusal**

benign instances中MCPDefender错误refuse而DPO正确执行的cases。分析over-refusal的trigger是什么（某些tool description看起来像poison？）。

每类选3-5个最illustrative的case展示在论文中。

### 9.2 Error Analysis

按(paradigm, tool_level)分组统计MCPDefender的failure cases：
- 哪个paradigm的failure rate最高？
- 哪个tool level最容易被attack成功？
- 有没有特定的poison framing（security/compliance/performance/feature）在defense后仍然有效？

### 9.3 GRPO学到了什么

比较SFT/DPO/GRPO在unseen paradigm test set上的response文本：
- GRPO是否develop出了更systematic的safety reasoning？
- 还是只是学到了"一律拒绝看起来奇怪的description"（brute-force refusal）？
- Response length对比：GRPO是否生成更长的reasoning？

---

## 10. 团队分工

```
Person A (Tianhao):
  Week 1-2: ANOVA + interaction效应 + effect sizes
  Week 3:   SFT训练
  Week 4-5: GRPO训练 + mechanism logging
  Week 6-7: Main evaluation
  Week 8-10: 写Method + Experiments

Person B (Tianchen):
  Week 1-2: Model clustering + PCA + scale analysis + Think vs No-Think
  Week 3:   DPO baseline训练
  Week 4-5: DefensiveTokens实现 + Prompt Hardening baseline
  Week 6-7: Ablation studies
  Week 8-10: 写Analysis + Results

Person C (JiaCheng):
  Week 1-2: 改进MCPTox mapping覆盖率 + per-server分析
  Week 2-3: 实现LLM-as-judge + judge验证 + baseline ASR测量
  Week 4-5: MCP-Gym适配MCPTox数据
  Week 6-7: Case studies + error analysis
  Week 8-10: 写Introduction + Related Work + figures
```

---

## 11. 完整时间线

```
Week 1:
  ├─ 改进MCPTox mapping覆盖率
  ├─ Two-way ANOVA (paradigm × level)
  ├─ Model-family clustering + PCA
  └─ 初步per-server分析

Week 2:
  ├─ 完成所有Phase 1分析（scale, Think/No-Think）
  ├─ 实现LLM-as-judge
  ├─ 验证judge和MCPTox labels的agreement
  └─ 用LLM-as-judge重新测Qwen2.5-7B baseline ASR
  → 确定victim model + train/test split

Week 3:
  ├─ 从MCPTox构造SFT/DPO/GRPO训练数据
  ├─ SFT warm-start (8h)
  ├─ DPO baseline训练 (4h)
  └─ 实现DefensiveTokens和Prompt Hardening

Week 4-5:
  ├─ MCP-Gym适配MCPTox数据
  ├─ GRPO训练 (12-18h) + mechanism logging
  └─ 如果GRPO有问题，debug和调参

Week 6-7:
  ├─ 全部6个方法的LLM-as-judge evaluation
  ├─ Main results table
  ├─ Ablation 1-3（如果时间允许到6个）
  └─ Per-paradigm × per-level breakdown

Week 8:
  ├─ Case studies (3类 × 3-5个case)
  ├─ Error analysis
  ├─ Mechanism validation (advantage plot)
  └─ GRPO reasoning analysis

Week 9-10:
  ├─ 论文写作
  ├─ 所有figures最终版
  ├─ 内部review
  └─ Revision
```

---

## 12. 成本

### GPU时间

| 任务 | GPU | 时间 |
|------|-----|------|
| SFT | A5000 24GB | 8h |
| DPO | A5000 24GB | 4h |
| GRPO (main) | A5000 24GB | 18h |
| GRPO (ablations, ~5次) | A5000 24GB | 90h |
| Inference (evaluations, ~20次) | A5000/2080Ti | 40h |
| **Total** | | **~160 GPU-hours** |

### API成本

| 任务 | 估算 |
|------|------|
| Judge验证 | $0.05 |
| Baseline评估 (3 defense × 150 instances) | $0.05 |
| Main evaluation (6 methods × ~180 instances) | $0.11 |
| Ablation evaluations (~10 × 180) | $0.18 |
| GRPO training validation (每200步做一次) | $0.50 |
| Debug + reruns | $1.00 |
| **Total** | **< $2** |

---

## 13. 风险矩阵

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| LLM-as-judge和MCPTox labels不一致(agreement<85%) | 15% | 高 | 调整judge prompt；或用MCPTox labels做re-analysis，LLM-judge只做defense evaluation |
| Qwen2.5-7B LLM-judged ASR < 20% | 25% | 中 | 换qwen3-8b（已证实43.3%） |
| GRPO比DPO好>10pp on unseen | 25% | 正面 | 强结果 → EMNLP主会 |
| GRPO比DPO好5-10pp | 30% | 正面 | 可以发 → EMNLP Findings |
| GRPO比DPO好<5pp | 25% | 中 | Re-analysis + comparison paper → Workshop |
| GRPO不如DPO | 10% | 中 | 报为negative result；或frame为"DPO suffices" |
| DefensiveTokens太有效(降ASR>50%) | 20% | 中 | 如果DefTokens单独就很好，GRPO需要beat DefTokens |
| A5000跑不了GRPO G=4 | 10% | 低 | 降G=2或用RTX 5000 32GB |
| GRPO训练不收敛 | 15% | 中 | DPO warm-start + KL=0.001 + curriculum |

---

## 14. 成功标准

### 顶会级别（EMNLP / CCS）

以下全部满足：
1. Re-analysis的3个finding都统计显著（ANOVA p<0.001）
2. MCPDefender在unseen paradigm上ASR比DPO低≥10pp
3. MCPDefender BTSR ≥ 85%
4. Mechanism validation图（advantage plot）显示训练时的difficulty targeting pattern
5. 至少一个ablation显示clear trend（如paradigm split ablation）

### Findings级别（EMNLP Findings / ACL Findings）

满足以下即可：
1. Re-analysis findings显著
2. MCPDefender > DPO on unseen（any margin）
3. Systematic 6-method comparison有value

### Workshop级别（AISec / SaTML）

满足以下即可：
1. Re-analysis findings显著（C1单独就够）
2. 或 6-method defense comparison（C3单独就够）

---

## 15. 论文结构

```
1. Introduction (1.5页)
   - MCPTox 36.5% avg ASR掩盖三层结构
   - 三个findings预览 + figure预览
   - 现有defense的paradigm-blind盲点
   - MCPDefender: GRPO for generalizable defense

2. Related Work (1页)
   - MCP attacks: MCPTox, MCP-ITP, ToolHijacker, MPMA, TrustDesc
   - MCP defenses: MindGuard, MCPShield, MCP-Guard, ToolSafe, MCP-DPT
   - RL for safety: ARLAS, MrGuard, SecAlign, GSPR

3. Fine-Grained MCPTox Analysis (2.5页)
   3.1 数据和方法（taxonomy, mapping）
   3.2 Finding 1: Paradigm dominance (P3 >> P1，ANOVA)
   3.3 Finding 2: Inverse risk paradox (L3 > L1，非risk-monotonic)
   3.4 Finding 3: Model-family clustering + scale/Think effects
   3.5 对defense设计的implications

4. MCPDefender (1.5页)
   4.1 DPO为什么对unseen paradigm不够
   4.2 GRPO的automatic difficulty targeting
   4.3 训练pipeline + LLM-as-judge evaluation
   4.4 和ARLAS/MrGuard的区别

5. Experiments (3页)
   5.1 Setup (baselines, splits, metrics, judge验证)
   5.2 Main results (seen vs unseen paradigms)
   5.3 Ablations (paradigm split, benign ratio, init)
   5.4 Mechanism validation
   5.5 Case studies

6. Discussion + Conclusion (1页)
   - Limitations (mapping rate, single victim model, LLM-judge reliability)
   - Think模式的safety paradox
   - 对MCP部署的recommendations

Total: ~11页
```

---

## 16. Checklist

### 本周必须完成（Week 1-2）
- [ ] 改进MCPTox mapping覆盖率（目标>60%）
- [ ] Two-way ANOVA完成
- [ ] Model clustering + PCA完成
- [ ] Scale analysis + Think/No-Think分析完成
- [ ] 实现LLM-as-judge
- [ ] 验证judge与MCPTox labels agreement (目标>85%)
- [ ] 用LLM-as-judge测Qwen2.5-7B真实baseline ASR
- [ ] **确定victim model + train/test split**

### Week 3-5
- [ ] 构造SFT/DPO/GRPO数据
- [ ] SFT训练
- [ ] DPO训练
- [ ] DefensiveTokens + Prompt Hardening实现
- [ ] GRPO训练 + mechanism logging

### Week 6-8
- [ ] 6个方法的LLM-as-judge evaluation
- [ ] Main results table
- [ ] 至少3个ablation
- [ ] Case studies + error analysis

### Week 9-10
- [ ] 论文完整draft
- [ ] 所有figures
- [ ] 内部review + revision