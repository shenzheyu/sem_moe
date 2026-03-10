# CLAUDE.md — 对齐论文文本的 Sem-MoE / Semantic Parallelism 复现指南

你在这个仓库里当我的研究助理写代码。

目标：在 **vLLM v0.17.0** 上尽量贴近论文
`Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling`
（arXiv:2503.04398v5，ICLR 2026）的算法语义与实验口径。

注意区分两件事：
- **算法复现**：离线 profiling、联合 co-scheduling solver、expert 重排、DP/TP 在线调度都要对齐论文。
- **系统性能复现**：论文系统是 **SGLang + DeepEP + 自定义 SRS/SAG kernel**。如果我们把目标引擎换成 vLLM，那么只有在通信路径也实现了与论文等价的改造后，才可以拿论文中的吞吐/时延收益做强对比。

如果某一步只能先做简化版，请明确标成：
- `debug fallback`
- `prototype baseline`

不要把简化启发式写成“论文已复现”。

---

## 0) 硬约束（必须遵守）

- 不要过度工程化：不要引入复杂架构、设计模式、依赖注入、工厂类等。
- 优先可读性：变量名清晰，流程直观。
- 只做必要的错误处理；不要写一堆冗长的健壮性代码。
- 尽量少改动 vLLM 已有代码；改动范围越小越好。
- 需要新增功能时：优先写成 1–2 个小函数，保持短文件、短函数。
- 如果实现的是简化版或调试版，必须在注释、README、PR 描述里明确写清，不要混称为“论文方案”。

---

## 0.5) 默认开发目标（优先按这个 setting 做）

默认优先支持模型：
- `Qwen/Qwen3.5-35B-A3B`

默认开发与联调分成两套 setting：
- 第一步到第三步：`DP=2, TP=1, EP=2`
- 第四步：`DP=1, TP=2, EP=2`

执行要求：
1. 离线 Token-Expert 激活建模、离线模型调度、Attention-DP 在线数据调度，优先在 `Qwen/Qwen3.5-35B-A3B, DP=2, TP=1, EP=2` 上先跑通
2. Attention-TP 在线数据调度必须单独在 `Qwen/Qwen3.5-35B-A3B, DP=1, TP=2, EP=2` 上跑通

除非我显式说明做跨模型或跨并行度泛化，否则实现和验证时默认先围绕这两套 setting 收敛。

---

## 1) 四步复现路径

### 第一步：离线 Token-Expert 激活建模
目标：
1. 按 **每个 MoE 层独立** 收集 token→expert 激活频率，构建 `count`、`freq`、`Cp` 与 `a`
2. 为 TP 路径额外收集 inter-layer activation trace，用于后续构建
   `A_prob` / `A` / `Ap`
3. 把 token 级预测结果扩展到 full vocabulary，生成在线可查的 `T_full` / `Tp_full` 或其前置统计

这一步的产物本质上是：
- token-expert affinity
- token frequency distribution
- inter-layer activation trace / device transition statistics

### 第二步：离线模型调度
目标：
1. 运行 co-scheduling solver，生成每层的：
   - `E`: expert -> cluster/device
   - `T`: token -> cluster/device
   - `Tp`: token -> cluster/device 的置信度
   - `A` / `Ap`: device sequence -> next device 的预测表
2. 在部署前按 `E` 规划 expert placement，并准备好 gating 矩阵列重排所需的映射
3. 保证 locality 与 load balance 两个目标都被纳入优化，而不是只做启发式静态分桶

> 只做 `T` 表 + request 调度，不做 `E` 的 expert 重排，只能算 paper-inspired baseline，不算论文主体复现。
> 本仓库第二步默认以 `iclr2026_conference.tex` 实际引用的
> `method_zhang.tex + motivation_zhang.tex` 为准，实现的是
> alternating solver，不采用旧稿里 CEO/CEM 采样版作为主线。

### 第三步：Attention-DP 在线数据调度
目标：
1. 基于 `T` / `Tp` / `T_score_full` 对 request 做 inter-request 调度
2. 用 `dev_mask` 或等价轮转机制保持 DP ranks 负载均衡
3. 在不破坏 decode 粘性的前提下，提高 local activation rate，降低 remote activation / all2all volume

默认开发 setting：
- `Qwen/Qwen3.5-35B-A3B`
- `DP=2, TP=1, EP=2`

验证至少包括：
- LAR（Local Activation Rate）
- load imbalance rate
- remote activation / all2all volume
- 在可用 workload harness 下的 TTFT/E2E SLO 下 throughput

### 第四步：Attention-TP 在线数据调度
目标：
1. 在 MoE 前按 `T/Tp` 与 `A/Ap` 选择 token 的目标 device，并完成 rebatch
2. 将 token shuffle 真正接入 `shuffled-reduce-scatter (SRS)` 和延迟的 `shuffled-allgather (SAG)`，或等价通信路径
3. 在 MoE 后恢复原 token 顺序，并验证 TP 场景下的 TTFT / E2E latency

默认跑通 setting：
- `Qwen/Qwen3.5-35B-A3B`
- `DP=1, TP=2, EP=2`

> “只做 token 重排，但仍走原始通信路径”只可作为 `debug fallback`。它可以验证逻辑正确性，但不能当成 TP 论文复现完成。

### 允许存在的调试里程碑
以下内容允许先做，但必须明确标成非最终目标：
- 不带 `E` 的 request 调度启发式
- `top-1 expert -> device` 的静态映射
- 只做 `rebatch/resume`、不改通信
- 小规模 prompt 列表上的冒烟测试

---

## 2) 论文中的核心对象（语义必须统一）

所有表都按 **per-layer** 构建。

### 2.1 Profiling / solver 输入
- `count[layer][token_id, expert_id]`
  含义：token 在该层命中该 expert 的次数
- `freq[layer][token_id]`
  含义：token 在该层出现的次数
- `Cp[layer][token_id, expert_id]`
  含义：按论文文本定义的 `Pr(expert | token)`，由
  `count[layer][token_id, expert_id] / sum_e count[layer][token_id, e]`
  沿 expert 维归一化得到
- `a[layer][token_id]`
  含义：token 频率分布

### 2.2 Solver 输出（论文主线）
- `E[layer][expert_id] = device_id`
  含义：expert 该放到哪个 expert-group / device
- `T[layer][token_id] = device_id`
  含义：token 在该层更倾向被调度到哪个 device
- `Tp[layer][token_id] = confidence`
  含义：`T[layer][token_id]` 这个选择的置信度

实现约定（为了和论文正文/appendix 的两种写法对齐）：
- 离线内部先构建
  `T_score[layer][token_id, device_id]`
- 然后导出：
  - `T[layer][token_id] = argmax(T_score[layer][token_id])`
  - `Tp[layer][token_id] = max(T_score[layer][token_id])`
- `T_score` 必须由最优 `p_matrix_req` 统计 `p_matrix_tk_opt`
  后得到；不要直接从 `Cp` 或启发式 affinity 上取 `argmax`

### 2.3 TP 额外表（inter-layer activation conjugacy）
论文这里建模的是 **device sequence -> next device distribution**。

按论文文本，`A/Ap` 的来源不是 token-expert co-scheduling solver 主表本身，
而是额外基于 inter-layer activation conjugacy 建出来的 lookup table。

推荐按下面方式离线构建：
1. 先收集每层 token 的 activation trace
2. 用当前层的 `E[layer][expert_id] -> device_id` 将 expert activation
   投影成 device activation
   - 当前仓库实现时，先把 top-k routed experts 投影到 device
   - 再对该 token 在该层取单个 `device_label`
   - 默认规则：多数票；若平票，取较小的 `device_id`
3. 对每个 layer、每个 `seq_id`（前 `lookback` 层 device sequence）统计
   当前层各 `next_device` 的出现次数
4. 沿 next-device 维归一化，得到
   `A_prob[layer][seq_id, device_id]`
5. 再导出：
   - `A[layer][seq_id] = argmax(A_prob[layer][seq_id])`
   - `Ap[layer][seq_id] = max(A_prob[layer][seq_id])`

注意：
- 论文正文没有把这一步写成显式计数公式，但正文和 appendix 都明确说明：
  先构造一个形状为 `[E^lookback, E]` 的 device transition probability
  表，再得到 `A` 与 `Ap`
- 在线 TP rebatch 使用的是 label + confidence 语义：
  `Tp[token]` 与 `Ap[seq_id]` 比较，置信度更高的表负责给出 device id

建议离线保存两种表示中的一种：

方案 A（推荐，更不容易混淆）：
- `A_prob[layer][seq_id, device_id]`
- `A[layer][seq_id] = argmax(A_prob[layer][seq_id])`
- `Ap[layer][seq_id] = max(A_prob[layer][seq_id])`

方案 B（更省空间）：
- 直接只存
  - `A[layer][seq_id] = next_device_id`
  - `Ap[layer][seq_id] = confidence`

其中：
- `seq_id` 编码前 `lookback` 层的 device 序列
- 论文实践中 `lookback = 2`
- 若采用 full distribution，表形状是 `[E^lookback, E]`
- 若只存 `A`/`Ap` label+confidence，则 `A` 形状是 `[E^lookback]`

不要再把“`A[layer][device_seq] = next_device_id`”和“表形状 `[E^2, E]`”混写在一起。

### 2.4 在线查表格式
在线路径需要 **全词表 O(1) lookup**。

因此最终用于服务时，至少要导出：
- `T_full[layer]`: `[vocab_size] -> device_id`
- `Tp_full[layer]`: `[vocab_size] -> confidence`
- `E[layer]`: `[num_experts] -> device_id`
- `A[layer]`, `Ap[layer]`

可选但推荐额外导出：
- `T_score_full[layer]`: `[vocab_size, num_devices]`

这样 DP request 打分可以直接做加权求和，而不是只做硬投票。

OOV / full-vocab 扩展默认规则：
- 第二步默认使用第一步 embedding nearest-neighbor 结果，将
  seen-token 的 `T_score` / `T` / `Tp` **直接 copy** 到 full vocabulary
- 不默认做 similarity smoothing
- 若要做 smoothing，只能明确标成 `debug fallback`

默认 solver 超参数（论文未给出精确数值时，本仓库统一用这个默认）：
- `alpha_e = 1.0`
- `beta_e = 1.0`
- `gamma_e = 1.0`
- `alpha_r = 1.0`
- `beta_r = 1.0`
- `theta = 0.5`
- `n_steps = 8`
- `ft_steps = 64`
- `lookback = 2`

---

## 3) 先在 vLLM 代码库中定位插入点（必须先做）

不要先拍脑袋写代码。先搜索真实插入点。

请按下面关键词在 vLLM 代码库中搜索：
- `"MoE"`, `"router"`, `"gating"`, `"topk"`, `"experts"`
- `"all2all"`, `"all_to_all"`, `"all2allv"`
- `"reduce_scatter"`, `"allgather"`, `"all_reduce"`
- `"scheduler"`, `"batch"`, `"rebatch"`, `"queue"`

重点关注这些目录，但要以实际搜索结果为准：
- `vllm/model_executor/layers/fused_moe/`
- `vllm/core/scheduler.py`
- `vllm/distributed/`
- `vllm/worker/`

你需要在 PR 描述里明确列出：
1. request 被加入 batch 并决定发往哪个 DP worker/rank 的地方
2. token 在进入 MoE 前的表示是什么
   - `token_ids`
   - `positions`
   - `hidden_states`
   - 以及任何需要跟随 token 顺序一起重排的 metadata
3. MoE dispatch / combine 的通信点在哪里
4. post-attention 的 TP 通信点在哪里
5. 哪个插入点是“最小改动”且不破坏 vLLM 现有调度假设

---

## 4) 离线 profiling：怎么收集 `Cp` / `a`

论文出处：
- `method_zhang.tex` §3.2
- Appendix `motivation_zhang.tex` 的背景与算法部分

### 4.1 数据口径
若要尽量贴近论文：
- 只使用下面三套数据集的 **prompt 部分**
  - `MMLU`
  - `lmsys-chat-1m`
  - `ShareGPT-Vicuna-unfiltered`
- 采用 **20% 作为 profiling / training，80% 作为评测** 的口径
- 保留 **vLLM-only** 路线：数据集口径对齐论文，但系统层默认仍做
  `vLLM baseline vs vLLM + Sem-MoE`，不把它写成对
  `SGLang / MoETuner` 引擎对比的直接复现

开发默认模型与并行配置：
- 模型优先：`Qwen/Qwen3.5-35B-A3B`
- 第一步到第三步优先：`DP=2, TP=1, EP=2`
- 第四步优先：`DP=1, TP=2, EP=2`

如果一开始只能用本地 prompt 列表跑通，也可以；但结果只能叫 prototype，不要声称对齐论文实验。

### 4.2 采集方法
在每个 MoE 层的 gating / route 位置加一个受开关控制的小 hook：
1. 跑模型 forward（prefill 即可）
2. 对每个 token occurrence，记录它在该层被路由到的 top-k expert
3. 每层独立统计：
   - `count[layer][token_id, expert_id] += 1`
   - `freq[layer][token_id] += 1`
4. 归一化：
   - `Cp[layer][token_id, expert_id] = count[layer][token_id, expert_id] / max(1, sum_e count[layer][token_id, e])`
   - `a[layer][token_id] = freq[layer][token_id] / sum(freq[layer])`

注意：
- 一个 token 可能同时命中多个 top-k expert，要全部计入 `count`
- 由于 top-k expert 都会计入 `count`，`Cp` 必须沿 expert 维归一化；
  不要再写成 `count / freq`
- profiling 是 **per-layer** 的，不要把所有层混到一起

### 4.3 OOV / 未见 token 处理
论文明确要求：
- 用 **embedding cosine nearest-neighbor** 把表扩展到 full vocabulary

因此：
- 最终在线使用的 `T_full/Tp_full` 不能只覆盖 profiling 中见过的 token
- “默认发到 0 号卡”或“均匀随机分配”只能作为临时 debug fallback

---

## 5) 离线 solver：按论文实现联合 co-scheduling

论文出处：
- `method_zhang.tex` §3.3
- `motivation_zhang.tex` Algorithm 1 (`algo:cem`)

### 5.1 必须对齐的优化目标
论文要同时考虑两件事：
1. **提升 locality / 降低 remote activation**
2. **保持负载均衡**

因此 solver 里至少要保留这些约束或近似约束：
- 每个 token 最终属于一个 cluster
- 每个 expert 最终属于一个 cluster
- 每个 cluster 的 expert 数量相等（`N / E`）
- token 频率负载尽量均衡

### 5.2 论文主线：交替优化
不要把主线写成 `expert_id % num_devices`。

应实现一个简化但语义正确的 alternating optimization：

```python
def expert_place(Cp, a, p_matrix_req, num_clusters):
    # 1. loads[e] = sum_token(Cp[token, e] * a[token])
    # 2. 按 expert hotness / load 降序
    # 3. 对每个 expert:
    #    计算它与每个 cluster 的 expert-expert affinity
    #    和 request-expert affinity
    #    再减去 cluster 当前 load penalty
    # 4. 选择得分最高且未满的 cluster
    # 5. 随机 swap 微调若干轮
    return p_matrix_ep

def request_schedule(Cp, requests, p_matrix_ep, num_clusters):
    # 1. 按 request 长度排序
    # 2. 对每个 request:
    #    计算 req-req affinity 和 req-expert affinity
    # 3. 选择得分最高的 cluster，并保持分配均衡
    return p_matrix_req

p_matrix_req = init_request_clusters_by_expert_affinity(...)
best = None
for step in range(n_steps):
    p_matrix_ep = expert_place(Cp, a, p_matrix_req, E)
    p_matrix_req = request_schedule(Cp, requests, p_matrix_ep, E)
    best = keep_better_schedule(best, p_matrix_ep, p_matrix_req)

E_table = argmax(best.p_matrix_ep, axis=1)
T_score = aggregate_request_clusters_to_token_scores(best.p_matrix_req, requests)
T_table = argmax(T_score, axis=1)
Tp_table = max(T_score, axis=1)
```

### 5.3 允许保留的启发式
如果为了先跑通，需要一个更简单的初始化：
- 可以用 hottest expert
- 可以先做均匀 expert bucket 作为 warm start

但这只能是：
- 初始化
- 或 ablation baseline

不能替代论文主线 solver。

---

## 6) 在线 model scheduling：`E` 是必须上线的

论文不是只做数据调度；它还做 **expert placement**。

因此在线部署时必须完成：
1. 按 `E[layer][expert_id]` 重排每层 expert
2. 同步重排 gating 矩阵列，使原有 gate 输出在新 expert 顺序下语义不变
3. 保留必要的 index 映射，便于调试与 correctness check

这一步是论文复现的必要条件，不是可选优化。

验证要求：
- expert reorder 前后，在“关闭在线调度”的情况下，模型数值语义应保持一致
- 若只改变 expert 顺序，输出不应出现系统性错误

---

## 7) 在线：Attention-DP inter-request 调度

论文出处：
- `method_zhang.tex` §3.3 中 Attention-DP 部分
- `motivation_zhang.tex` Algorithm 2 (`algo:req_sched_ol`)

### 7.1 推荐函数签名
```python
def pick_dp_rank_for_request(token_ids, T_full, Tp_full=None, T_score_full=None, dev_mask=None):
    # token_ids: 当前 request 的 token ids
    # T_full: [vocab] -> device_id
    # Tp_full: [vocab] -> confidence
    # T_score_full: 可选，[vocab, num_devices]
    # dev_mask: [num_devices] bool
    # return: device_id
```

### 7.2 打分逻辑
论文 Algorithm 2 的本质是：
- 汇总 request 内所有 token 对各 device 的偏好
- 选分数最高且当前未 mask 的 device
- 用轮转式 `dev_mask` 保持负载平衡

优先级如下：
1. **首选**：若有 `T_score_full`
   - `dev_score[dev] = sum(T_score_full[token, dev] for token in token_ids)`
2. **退化版**：若只有 `T_full/Tp_full`
   - `dev_score[dev] = sum(Tp_full[token] if T_full[token] == dev else 0 for token in token_ids)`
3. **最简硬投票**
   - `dev_score[dev] = count(T_full[token] == dev)`

然后：
1. `dev_score[~dev_mask] = -inf`
2. `dev_id = argmax(dev_score)`
3. `dev_mask[dev_id] = False`
4. 若 `dev_mask` 全 False，则 reset

### 7.3 调度语义
- request 一旦被分到某个 DP rank，应在其后续 decode 生命周期中保持粘性
- 不要在 decode 中途随意迁移 request
- 轮转 mask 是为了避免连续若干个相似 request 全打到同一 rank 上

插入点：
- 在 vLLM 的 request 调度器中，request 被确定发往哪个 DP worker / rank 之前
- 只在 `semmoe_enabled` 时生效

---

## 8) 在线：Attention-TP intra-request token rebatching

论文出处：
- `method_zhang.tex` §3.3 中 Attention-TP 部分
- `motivation_zhang.tex` Algorithm 3 (`algo:fast_shf`)

### 8.1 需要的表
对每一层：
- `T[layer]`, `Tp[layer]`
- `A[layer]`, `Ap[layer]`

其中：
- `T/Tp` 由 token identity 直接预测当前层 device
- `A/Ap` 由前 `lookback=2` 层的 device 序列预测当前层 device

### 8.2 推荐函数签名
```python
def rebatch_tokens(token_ids, seq_ids, T, Tp, A, Ap):
    # token_ids: [B]
    # seq_ids: [B]，由前 lookback 层 device sequence 编码得到
    # 返回：
    #   shf_idx: [B]
    #   token_ids_shuffled: [B]
    #   target_dev_ids: [B]
```

### 8.3 逻辑（按论文语义）
```python
dev_ids = where(Tp[token_ids] > Ap[seq_ids], T[token_ids], A[seq_ids])
shf_idx = argsort(dev_ids)
g_shf_idx = group_by_key(shf_idx, dev_ids)
g_shf_idx = align(g_shf_idx)   # 保证每个 rank 接收等长切片
shf_idx = concat(g_shf_idx)
token_ids_shuffled = token_ids[shf_idx]
```

MoE 后恢复：
```python
inv = argsort(shf_idx)
x = x[inv]
```

### 8.4 与通信路径的关系
完整论文复现要求：
- token shuffle 要嵌入 TP 的 `reduce_scatter`
- MoE 后用延迟的 `allgather` / 等价路径恢复

也就是说，最终目标不是单独加一个 `argsort`，而是要把它接到真实通信路径里。

### 8.5 vLLM 实现注意事项
重排的不只是 `token_ids`，还要检查并同步重排：
- `hidden_states`
- `positions`
- `slot_mapping`
- `seq_lens` 或其他依赖 token 顺序的 metadata

如果这些元数据没跟着重排，逻辑会错，但错误不一定马上暴露。

---

## 9) 离线落盘格式（保持简单，但要自洽）

优先使用每层一个 `.npz`：

```python
semmoe_layer{L}.npz
```

建议内容：
- `E`
- `T_full`
- `Tp_full`
- `A`
- `Ap`
- 可选：`T_score_full`
- 可选：`A_prob`
- 可选：`token_ids_seen`
- 可选：`Cp`
- 可选：`a`

注意：
- 在线服务真正需要的是 `E/T_full/Tp_full/A/Ap`
- `Cp/a` 更偏离线分析与复现实验，不一定必须在线加载

---

## 10) 开关与配置（保持极简）

只保留少量必要开关：
- `SEM_MOE=1`
- `SEM_MOE_TABLES=/path/to/semmoe_tables`
- `SEM_MOE_MODE=dp|tp|both`
- `SEM_MOE_DEBUG_FALLBACK=0|1`

可选：
- `SEM_MOE_LOOKBACK=2`

不要引入复杂 config 系统；环境变量或少量 args 即可。

---

## 11) 验证与实验口径（按论文分层）

### 11.1 Predictor / profiling 层
至少输出：
- 每层 token→expert 预测的命中情况
- hottest top-k 的 precision / F1（若实现方便）
- OOV 覆盖率

### 11.2 Algorithm 层
至少输出：
- LAR（Local Activation Rate）
- load imbalance rate
- remote activated token 数 / all2all volume

对比对象至少包括：
- baseline：原始 expert placement + 原始调度
- Sem-MoE：`E + T/Tp (+ A/Ap for TP)`

如果实现了启发式简化版，也可以额外对比：
- heuristic-only baseline

### 11.3 System 层
若要对齐论文主结果：

- 保留 **vLLM-only** 路线：系统层默认比较
  `vLLM baseline` vs `vLLM + Sem-MoE`
- 不要把 vLLM-only 的系统对比写成对 `SGLang` 或 `MoETuner`
  结果数值的直接复现

Attention-DP：
- 默认优先在 `Qwen/Qwen3.5-35B-A3B, DP=2, TP=1, EP=2` 上开发和验证
- 固定 input/output length
- 变化 request rate
- 画 latency-throughput 曲线
- 在 TTFT SLO / E2E latency SLO 下比较 throughput

Attention-TP：
- 默认优先在 `Qwen/Qwen3.5-35B-A3B, DP=1, TP=2, EP=2` 上开发和验证
- request rate 固定为 1 req/s
- 变化 input length（如 256 / 512 / 1024）
- 比较 TTFT 与 E2E latency

### 11.4 数据划分
若使用论文数据集：
- 数据集固定为 `MMLU`、`lmsys-chat-1m`、
  `ShareGPT-Vicuna-unfiltered` 的 prompt 部分
- 20% 数据用于 profiling / 构表
- 80% 数据用于评测

### 11.5 结果声明规则
只有当下面两条都满足时，才可以写“基本复现论文主线”：
1. `E` 已上线并参与真实 expert placement
2. TP 路径已接入真实 SRS/SAG 或与之等价的通信改造

否则请明确写成：
- `DP-only reproduction`
- `algorithmic prototype`
- `logic-only TP fallback`

---

## 12) 交付要求（你提交 PR 时要包含）

1. 你改动/新增了哪些文件
2. 你选择的插入点在哪里，为什么是“最小改动”
3. 当前实现属于哪一档：
   - debug fallback
   - prototype baseline
   - DP-only reproduction
   - DP+TP paper-aligned reproduction
4. 如何跑：
   - 离线 profiling 命令
   - 离线 solver / 构表命令
   - 在线开启开关的最小 demo 命令
   - 实验脚本命令
5. 至少给出一份最小表文件样例
6. 给出一组 baseline vs semmoe 的关键指标

---

## 13) 你不要做的事（避免跑偏）

- 不要把启发式 baseline 写成“论文方案”
- 不要只做 request 调度却省略 `E`
- 不要把 `A` / `Ap` 的形状和语义写混
- 不要把“只重排 token 不改通信”当成 TP 复现完成
- 不要重构 vLLM 的 scheduler / engine
- 不要引入新的大型依赖（除非 vLLM 已在用）
- 不要写一堆“以后可能用得上”的抽象层
- 不要为了“通用”牺牲可读性

按这四步执行：
1. 先做离线 Token-Expert 激活建模
2. 再做离线模型调度
3. 再上线 Attention-DP 在线数据调度
4. 最后做 Attention-TP 在线数据调度与通信路径改造
