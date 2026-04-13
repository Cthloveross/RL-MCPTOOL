# DCC Xu Lab 运行指南

## 集群信息

- **集群**: Duke Compute Cluster (DCC)
- **账户**: `xulab`
- **Login 节点**: dcc-login-01/02
- **工作目录**: `/work/tc442/RL-MCPTOOL`

## GPU 分区

| 分区 | GPU 类型 | 时限 | 说明 |
|------|---------|------|------|
| `scavenger-gpu` | A6000 (48GB), RTX 6000 Ada (48GB), RTX 5000 Ada (32GB), A5000 (24GB), 2080 (11GB) | 7 天 | 可被抢占，适合 MVE |
| `scavenger-h200` | H200 (80GB) | 7 天 | 可被抢占，适合完整训练 |
| `gpu-common` | RTX 5000 Ada (32GB), 2080 (11GB) | 2 天 | 不抢占 |

**注意**: `dcc-courses-gpu-[01-10]` 节点是 P100 16GB，VRAM 不够跑完整实验。SLURM 脚本里必须 `--exclude` 这些节点。

## 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n safemcp python=3.10 -y
conda activate safemcp

# 2. 安装 PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 安装项目依赖
cd /work/tc442/RL-MCPTOOL
pip install -e .

# 4. 预下载模型到 /work（避免占用 home 配额）
mkdir -p /work/tc442/hf_cache
export HF_HOME=/work/tc442/hf_cache
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct')"
```

Qwen2.5-3B-Instruct 不需要 gated access，无需 HF login。模型约 5.8GB。

## SLURM 提交

提交 MVE:
```bash
sbatch slurm/run_mve.sh
```

监控:
```bash
squeue -u tc442                           # 查看作业状态
tail -f slurm/logs/mve_<JOBID>.out        # 实时输出
cat experiments/mcpalign_mve/mve_results.json  # 查看结果
```

## 已知问题与规避

### P100 16GB VRAM 不够

第一次 MVE 跑在了 `dcc-courses-gpu-03`（P100 16GB）上。虽然 3B 4-bit 模型可以跑，但容易 OOM 且推理慢。

**规避**: SLURM 脚本加 `--exclude=dcc-courses-gpu-[01-10]`。

### mve_02 全部 format_error

第一次跑 mve_02（run_sql → export_data → send_email）的所有 9 个 step 都输出了 invalid JSON。原因：
1. Prompt 格式不够明确（只给了 schema，没给具体 example）
2. Parser 缺少 partial JSON recovery
3. 3B 模型在 DB 类任务上 JSON 合规率较低

**已修复**: prompts.py 加了 3 个具体 JSON example，actions.py 加了 partial JSON recovery（Strategy 4）。

### ΔASR 为负的根本原因

第一次 MVE 的 ΔASR = -20%（Step 1 ASR 40%, Step 3 ASR 20%）。原因是**实验设计缺陷**，不是假设不成立：

- Template 随机选取，很多 poison 恰好 target 的是 step 1 的 tool（如 read_file），导致 step 1 ASR 虚高
- 全局聚合 ASR 混合了"poison 激活"和"poison 休眠"的 step，信号被淹没
- P1 攻击是 tool replacement，只在目标 step 生效，非目标 step 的 ASR 接近 0

**已修复**: 重写 MVE 脚本，改用 Active Step ASR 分析——只在 poison 指向当前 step tool 时计算 ASR，按 position 分组比较。

## SLURM 脚本说明

`slurm/run_mve.sh` 关键参数：

```
--account=xulab                       # Xu Lab 账户
--partition=scavenger-gpu             # GPU 分区
--gres=gpu:1                          # 1 张 GPU
--mem=64G                             # CPU 内存
--time=06:00:00                       # 6 小时上限
--exclude=dcc-courses-gpu-[01-10]     # 排除 P100 节点
```

## 资源使用估算

| 实验阶段 | GPU | VRAM | 时间 | 说明 |
|---------|-----|------|------|------|
| MVE (46 trials) | 任意 ≥ 24GB | ~6-8 GB | ~30 min | 纯推理，无训练 |
| SFT warm-start | A6000/H200 | ~18-23 GB | ~2-4 h | QLoRA 训练 |
| GRPO 训练 | H200 推荐 | ~30-40 GB | ~12-20 h | 自定义训练循环 |
| DPO baseline | A6000/H200 | ~18-23 GB | ~2-4 h | TRL DPOTrainer |
| 评估 | 任意 ≥ 24GB | ~6-8 GB | ~1-2 h | 纯推理 |
