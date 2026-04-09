# SafeMCP 实验执行手册

## 总览

所有代码已就绪（2798 行，语法和 import 全部验证通过）。你需要做的事只有一件：在 A100 上跑 MVE，把结果发回来。

---

## 第一步：环境搭建

在 GPU 服务器上执行以下操作：

1. 创建 conda 环境：safemcp, Python 3.10
2. 安装 PyTorch 2.4.0（CUDA 12.1）
3. 克隆 https://github.com/Cthloveross/RL-MCPTOOL.git
4. 进入项目目录，执行 pip install -e . 安装所有依赖
5. 执行 huggingface-cli login 登录 HuggingFace（需要 Read 权限 token）
6. 预下载 Qwen/Qwen2.5-3B-Instruct 模型（约 6GB）
7. 运行一行验证脚本确认 PyTorch、TRL、PEFT 版本正确，GPU 可用

---

## 第二步：验证 MCPGym 能跑

运行一个简短的 Python 验证：import MTMCPGym，加载 tool_registry.json + multistep_tasks.json + attack_templates/，确认输出 14 tools、5 tasks、30 P1 templates。采样 3 个 episode 看看 poisoned/benign 分布是否正常。

预期：能看到 POISON 和 BENIGN episode 交替出现，每个 episode 有 3 步，poisoned episode 有 malicious tool name。

---

## 第三步：跑 MVE（最关键）

执行 scripts/mcpalign_mve.py --config configs/mcpalign_mve.yaml

这个脚本做的事：
- 加载 Qwen2.5-3B-Instruct（4bit 量化，无任何训练）
- 对 5 个多步任务 × 最多 3 个匹配的 P1 攻击模板，运行 3 步 ReAct loop
- 每步让模型生成 tool call，用 rule-based judge 判定是否被攻击成功
- 输出 per-step ASR 和 ΔASR

预计运行时间：2-4 小时（取决于 GPU 速度）

### MVE 通过标准

- Step 1 ASR > 15%：攻击有效。如果 < 15% 说明模板太弱或模型太强。
- ΔASR > 10%：核心假设成立。ASR 确实随步数增长。
- 至少有 late-step compromise：不是随机噪声。

三项全过 → 继续做完整实验。

### 如果 MVE 失败

- 全部 ASR < 10% → 换更弱的模型（Llama-3.1-8B），或删掉 system prompt 里的安全提醒
- ASR 高但 ΔASR < 5% → 增加到 5 步任务，或用 late-step 型 poison（指令只在后面步骤才可执行）
- ΔASR 5-10% → 可以继续但论文 story 会弱
- Step 3 ASR < Step 1 ASR → 检查 context 构造，poison 可能被截断了

---

## 第四步：MVE 通过后的完整流程

按顺序执行 6 个脚本：

1. **生成训练数据**（~1 分钟）：mcpalign_generate_data.py → 产出 sft_data.json（500 样本）+ dpo_pairs.json（500 pairs）

2. **SFT warm-start**（~2-4 小时）：mcpalign_train_sft.py → 产出 sft_checkpoint/。之后验证 JSON 格式合规率 > 90%（让 SFT 模型跑 20 个 episode，检查 parse_agent_action 是否能解析输出）。如果 < 90% 则增加 SFT epochs。

3. **GRPO 训练**（~12-20 小时）：mcpalign_train_grpo.py → 产出 grpo_checkpoint/。建议用 tmux 挂后台。训练日志中观察 avg_reward 是否从负值逐渐上升。

4. **DPO baseline**（~2-4 小时）：mcpalign_train_dpo.py → 产出 dpo_checkpoint/。可以和 GRPO 并行跑如果有两张卡。

5. **评估**（~1-2 小时）：mcpalign_evaluate.py → 输出 per-step ASR + ΔASR + BTSR + ORR 对比表（No Defense / DPO / GRPO）。

6. **生成图表**（~1 分钟）：mcpalign_analyze.py → 产出 figures/ 目录下的 asr_comparison.pdf + pareto_frontier.pdf + results_table.tex。

---

## OOM 排查

如果显存不够，在 configs/mcpalign_mve.yaml 中：
- num_generations 从 4 降到 2
- per_device_train_batch_size 从 4 降到 2，gradient_accumulation_steps 从 2 升到 4
- max_completion_length 从 256 降到 128

---

## 你需要发回来的结果

跑完 MVE 后，把以下信息发给我：

1. 终端输出的 per-step ASR 表格和 ΔASR 值
2. experiments/mcpalign_mve/mve_results.json 文件内容
3. 如果失败了，具体是哪个标准没过

我会根据结果决定下一步：调攻击模板、调模型、还是继续推进完整训练。
