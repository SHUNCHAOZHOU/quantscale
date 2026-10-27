# QuantScale

研究 AI 研究员数量、协作方式与研究预算，如何影响量化策略发现与研究过拟合的 Python 实验框架。

这是一版可运行的研究 MVP。`demo` 离线演示使用**合成行情 + 带种子的规则搜索**，用于检查实验管线，不代表真实 LLM 的研究能力或真实市场收益。`research` 和 `scaling` 默认使用 `.env` 配置的 DeepSeek Flash；也可接入 OpenAI Responses API 或自己的模型适配器。

## 快速运行

需要 Python 3.9+、NumPy 1.24+、Pandas 2.0+。在项目目录运行：

```bash
python3 -m pip install -e .
python3 -m quantscale demo --output results/my-demo --backtests 4
python3 -m quantscale demo --output results/my-independent --topology independent --fleet-size 4 --backtests 8
python3 -m quantscale demo --output results/my-centralized --topology centralized --fleet-size 4 --backtests 8
python3 -m unittest discover -s tests -v
```

已安装依赖时，无需 editable install，可以直接使用 `python3 -m quantscale`。输出目录必须不存在，避免覆盖旧实验。演示使用 48 只合成股票、2018 上半年训练、2018 下半年验证、2019 上半年隐藏测试；正式日期见 `configs/default.json`。

每次运行保存：

- `research/protocol.json`：预先确定的配置、预算、拓扑、数据指纹与选择规则。
- `research/trajectory.jsonl`：完整模型请求、响应、用量、候选策略、训练/验证指标与错误。
- `research/process.json`：研究过程指标。
- `research/selection.json`：仅按验证结果选出的候选。
- `research/frozen.json`：策略、配置及实现版本的 SHA-256 冻结凭据。
- `research/implementation/`：冻结时的 Python 源码快照与依赖版本，供恢复原实验实现。
- `evaluation/evaluation.json`：单次隐藏测试指标、验证与测试 Sharpe 差值、按年表现。
- `evaluation/hidden_daily.csv`：决策/执行/收益日期、毛/净收益、换手、成本、敞口。

## 首版实现范围

已实现 CSV 校验、按月滞后股票池、时间划分、确定性回测、敞口约束、交易/借券成本、规则策略、五种信息流拓扑、全局预算、验证选择、冻结与独立隐藏评估、研究矩阵运行和日志。

策略接口为 `generate_weights(history, universe)`。`history` 是字段到 DataFrame 的字典，行是截至决策日的日期，列是股票。开放代码模式默认提供全部已发生历史，不固定信号窗口；`code_history_days` 可显式配置数据窗口。`universe` 是当日股票代码 Index；返回值必须是覆盖全部股票的 `pd.Series`，零持仓也须显式返回。

**当前默认是 `strategy_mode="python"` 的开放代码研究**：模型自主提出假设并提交完整 Python 策略，不预设策略家族、因子列表或参数范围；可以组合信号、改变选股和仓位算法。prompt 不提供具体策略或参数示例。保留无预测模型训练、数据边界、交易约束和资源预算。四家族配置模式以 `strategy_mode="rules"` 保留，用于旧实验对照；已有先导结果来自受限模式，不能当成开放代码实验结果。

开放策略在 macOS `sandbox-exec` 子进程中运行，不在研究主进程里直接 exec。密钥不会传入子进程；策略不能读取项目、其他研究员或隐藏数据文件，不能联网或启动子进程。父进程按决策时点逐步传入已发生数据，子进程没有完整未来行情文件。代码保存于 `strategies/<AST-SHA256>.py`，冻结凭据包含完整源代码。调度仍按固定轮次执行；管理器可以分配自由文本任务，尚无动态配额重分配或通用分析终端工具。

## 开放代码研究

```bash
python3 -m quantscale research \
  --data data/sp500-yahoo-2016-2025/research-prices.csv \
  --config configs/sp500-open.json \
  --backend deepseek --topology independent --fleet-size 4 \
  --output results/open-independent-4
```

不限定信号类别、公式、组合方式、选股比例或因子窗口。允许 NumPy、Pandas 和标准数值统计库，代码可定义辅助函数。交易约束仍由引擎逐日验证；代码或权重错误消耗候选预算并反馈给研究员。相同 AST 的重复提案仍消耗预算，可以复用指标。没有强制研究员换家族，也没有预设固定分工。

无法建立 OS 隔离时直接失败，不降级为主进程执行。限制包括单次调用超时、回测墙钟和 CPU 时间。`code_memory_mb` 是尽力而为的堆限制，macOS 可能拒绝应用，实际情况记录于 `execution_limits`，不是严格的系统总内存上限。静态检查禁止外部包、动态执行/私有反射和常见训练入口；它不能证明任意数值代码都没有隐式训练预测模型，研究协议仍需代码审核。

普通测试：`python3 -m unittest discover -s tests -v`。完整 OS 隔离测试需在可启动系统隔离的终端运行：`QUANTSCALE_SANDBOX_TESTS=1 python3 -m unittest discover -s tests -v`。

`configs/open-research-smoke.json` 用于小规模合成数据接入验证，真实行情配置为 `configs/sp500-open.json`。此前的 2024–2025 结果已被查看；在同一区间比较新协议属于历史留出对照，不能重新称为从未触达的盲测。

## 数据准备

### 下载标普500真实行情

已实现可恢复的 Yahoo 下载器，成分名单保存为当前快照，不冒充历史逐日成分。

```bash
python3 -m pip install -e '.[data]'
python3 -m quantscale.yahoo --output data/sp500-yahoo-2016-2025
```

默认下载 2016-01-01（含）到 2026-01-01（不含）的日线。重复运行会跳过已保存的股票；`--finalize-only` 仅用本地缓存重新合并，不发送网络请求。输出目录内含：

- `constituents.csv` 与 `constituents-source.html/json`：成分代码、来源快照与获取时间。
- `raw/`：Yahoo 原始返回，包括异常记录。
- `by-ticker/`：清洗后的逐股数据；`quarantine/` 保存被隔离的非空异常记录。
- `prices.csv`：合并行情；`research-prices.csv` 为 2016–2023，`hidden-prices.csv` 为 2024–2025。
- `spy.csv`：用于检查交易日覆盖的真实 SPY 日线。
- `coverage.csv`：逐股起止日期、记录数、内部缺失日期与异常记录计数。
- `download-errors.json` 与 `manifest.json`：无数据股票、下载配置、覆盖统计及文件指纹。

下载器保留 Yahoo OHLC、Adj Close、Volume、分红和拆股字段，使用同日 `Adj Close / Close` 因子计算 adjusted_open。缺失行不填充，不一致的 OHLC 行从清洗数据隔离但保留原始文件。Yahoo OHLC 可能已作拆股调整，不能视为当时原始成交价的无偏重建。

**这份下载是 price-only 面板，未伪造历史市值。** 当前按历史市值构建股票池的 `load_data` 因此不会直接接受它。使用当前成分固定池研究时应另外接入明确的固定股票池模式；若要做历史 PIT 实验，还需补齐历史成分、历史市值与退市事件。

已加入显式的 `universe_mode="fixed"`，该模式不要求或生成 market_cap；`universe_size` 必须容纳全部输入股票，每天再按当日可见价格、成交量和 eligibility 过滤。原有 `lagged_market_cap` 模式保留严格市值要求。

### 真实行情拓扑先导实验

```bash
python3 -m quantscale.pilot --output results/sp500-deepseek-topology-pilot
```

固定配置在 `configs/sp500-pilot.json`：2016–2021 训练、2022–2023 验证、2024–2025 隐藏测试；SAS 为 1 名研究员，其他四种拓扑为 4 名 worker，中心式/混合式另有管理器。每组 12 个候选、20 万 token 上限，通信与管理器均计入；每种拓扑重复两次（标签 7、17），标签不是服务端可控的随机种子。调度顺序预先保存，最多同时运行两个研究系统。

研究阶段仅挂入截至 2023 年的数据。所有 10 个策略冻结并生成 `all-strategies-frozen.json` 后，运行器才开启任何隐藏评估；若任一研究条件失败则不打开隐藏测试。完整结果保存在 `comparison.csv`，描述性两次均值在 `topology-summary.csv`；两次重复不足以做统计显著性判断。

本先导实验在调用模型之前固定 `missing_data_policy="skip_incomplete_sessions"`：若某只已出现过报价的股票当日缺行，整个横截面会话从回测日历中排除，IPO 之前没有报价不视为缺行。价格不填补，相邻开盘区间跨过排除日期，年化仍采用 252 个有效观测；这会带来轻微的日历口径差异，并在每次协议和评估中记录排除日期。当前数据研究期排除 2021-05-05，评估期另外排除 2025-11-12。默认 `strict` 模式仍然拒绝缺失的持仓价格。

为避免重复构造完整历史窗口，旧规则因子支持尾部历史预计算缓存，并测试其权重与原接口一致、未来数据变更不影响过去权重。重复策略仍消耗候选预算。这份历史先导配置使用 `strategy_mode="rules"`；开放代码研究配置另见 `configs/sp500-open.json`，训练预测模型仍不在协议内。

结果受当前成分股幸存者偏差、受限规则空间、简化交易成本等限制；历史隐藏期也不保证不在模型预训练资料中。这是一轮受控先导实验，而非可部署的投资结论。

### 导入完整研究数据

CSV 一行对应一个交易日、一只股票，必须包含以下字段：

```text
date,ticker,open,high,low,close,adjusted_open,adjusted_close,volume,market_cap
```

`eligible` 为可选布尔列，默认 true。日期使用不带时区的 `YYYY-MM-DD`。同一日期/股票不能重复；价格必须有限且大于零；成交量、市值必须有限且非负；OHLC 必须一致。股票代码按字符串读取。

`open/close` 是当时的原始价格；`adjusted_open/adjusted_close` 应是口径一致的总收益调整价格，用于持有期收益和无量纲规则信号。`market_cap` 与 `eligible` 必须是当时可获取的信息。当前规则全部使用比例、收益率、排名，避免用复权价格绝对值筛选股票。

每月股票池使用**上个月最后一个观测交易日**的市值、价格、成交量和 eligibility 筛选并排名；按 ticker 打破市值并列。整月持有该名单，决策日再排除缺行、无成交、或不合资格股票。最初没有上月快照的月份，股票池为空。请提供足够预热期。

输入数据必须包含历史退市股票和完整市场交易日。引擎不会把缺失的持仓收益默认为零，也不会自动前向填充持仓价格；缺少成交/持有期退出价格会直接使候选失败。**首版尚无专用退市清算、停牌成交、借券可得性、冲击成本或容量模型**，真实研究前必须完成相应数据与执行扩展。仅下载今天仍上市的股票会产生幸存者偏差。框架不附带商业市场数据，也不自动下载行情。合成日历仅为工作日，不模拟美股节假日。

由数据操作员切分：

```bash
python3 -m quantscale prepare \
  --data /path/to/equities.csv \
  --config configs/default.json \
  --output data/partitions
```

`research.csv` 截止验证期结束；`evaluator.csv` 含研究预热数据和隐藏测试期。真实实验应把后者放到独立评估账号/容器，研究进程只挂载 research.csv，且不能读取评估注册数据库和结果目录。`prepare` 是操作员步骤，不应让研究员执行。

## 研究与评估

```bash
python3 -m quantscale research \
  --data data/partitions/research.csv \
  --config configs/default.json \
  --topology independent --fleet-size 4 \
  --output results/independent-4

# 由独立评估进程在研究结束后执行
python3 -m quantscale evaluate \
  --frozen results/independent-4/frozen.json \
  --data data/partitions/evaluator.csv \
  --output results/independent-4-hidden
```

研究命令默认使用 DeepSeek Flash；离线搜索须显式加 `--backend demo`，`demo` 子命令始终离线。研究入口会拒绝任何验证结束日以后的行，不会悄悄加载后再隐藏。

隐藏评估首先校验冻结文件、源代码版本、Python/NumPy/Pandas 版本和研究数据指纹；然后在评估数据目录的 `.quantscale-evaluations.sqlite3` 中以事务登记，再运行回测。同一冻结凭据不能再次评估，换输出目录也不能绕过；失败的评估也算一次尝试。保留并备份注册数据库。修改实现后，旧冻结凭据会拒绝运行；需要时可在 `research/implementation/` 目录启动同版本 Python，用保存的源代码快照运行原实现。

这些检查是可信操作员使用的实验协议，不是密码学防篡改签名，也不是对恶意本地进程的访问控制。能修改代码/文件、复制数据或删除数据库的人仍能绕过；严格盲测依赖独立评估环境与访问权限。

## 接入真实 LLM

### DeepSeek Flash（当前默认）

在项目根目录 `.env` 中设置 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL`。适配器会自动读取，已有的进程环境变量优先；不会执行 `.env` 中的 shell 表达式。`.env` 已加入 Git 忽略规则，密钥不会写入研究日志或冻结快照。

```bash
python3 -m quantscale research \
  --data data/partitions/research.csv \
  --config configs/default.json \
  --backend deepseek --topology single --fleet-size 1 \
  --output results/deepseek-single
```

模型固定为 `deepseek-flash`，不会退回其他模型。默认研究者为 `low` 推理、最多 32,768 输出 token；管理者关闭推理、最多 8,192 输出 token。`.env.example` 提供 `DEEPSEEK_RESEARCHER_REASONING_EFFORT` / `DEEPSEEK_RESEARCHER_MAX_OUTPUT_TOKENS` 和对应的 `DEEPSEEK_MANAGER_*` 设置。已有的全局 `DEEPSEEK_REASONING_EFFORT` / `DEEPSEEK_MAX_OUTPUT_TOKENS` 仍作为两类角色的后备配置；角色专用配置优先，显式设置不会被默默覆盖。每次日志记录实际角色设置、供应商实际用量和该请求最终应用的输出上限。推理与正文共享输出额度，降低推理或提高上限都不保证不会截断。

Flash 模型名及调用方式依据 [DeepSeek 官方文档](https://api-docs.deepseek.com/quick_start/agent_integrations/)。别名背后的版本可能变化，日志保留服务返回的 model、system_fingerprint、response_id 和实际 usage，而不声称固定了不可变模型快照。

DeepSeek 使用 Chat Completions 接口并按返回的 `usage.total_tokens` 记账，保留 prompt/completion/reasoning/cache 用量明细。输入预算采用序列化消息 UTF-8 字节数加 1024 的保守预留，再限制生成量；**这不是服务端精确输入计数，也不保证绝对硬上限**。若实际用量超过本轮/全局预算会中止实验；剩余额度不足预留则提前停止。截断/非法 JSON 候选仍消耗实际 token 与试验配额；网络失败不自动重试，避免隐藏未知用量。

管理者协议现在先做确定性规范化：支持缺少 `directions` 外层、无歧义的 worker 标识、`assignment/task/direction` 嵌套文字。附带的主题、约束与共享指导会保留并传给研究者，不会默默丢弃。重复 JSON 键、缺少 worker、冲突任务、未知结构仍被拒绝。

`manager_max_format_repairs` 默认 1（可设 0，最大 2）。只有完整响应的简单分工映射发生 JSON 标点/包装错误、且原始任务和 worker 映射可逐字核对时，才允许模型格式修复；修复结果必须与原始可核对分工一致。复杂或歧义内容不允许猜测，截断代码不拼接。原始正文、错误、规范化步骤、修复请求和实际 token 都写入轨迹，修复调用计入统一 token 预算和通信调用，不占策略候选配额。

单轮管理者分工失败会记录并进入下一轮；连续失败达到 `manager_max_failed_rounds`（默认 2）则结束搜索，保留此前的有效候选用于正常选优和冻结。若没有有效候选则仍失败。网络/TLS错误和用量不明的响应不在此修复范围，不自动重试。

旧版 `results/sp500-deepseek-open-pilot-v2` 及其 HTML 报告描述的是旧 `high / 16,384` 设置和旧失败政策，保持不变。新实验使用 `configs/sp500-open-v3.json`，修改后尚需真实 API 的独立合成数据预实验；本地模拟和旧响应回放不能证明线上生成完成率已经提高。

`configs/deepseek-smoke.json` 只允许两次候选试验、32000 个总 token，供接入测试。它使用既有合成研究数据，不能当成真实行情实验。

### OpenAI 或其他适配器

内置适配器只使用 Python 标准库。设置 `OPENAI_API_KEY` 和 `QUANTSCALE_MODEL`（建议指定实验使用的固定模型版本），然后运行：

```bash
python3 -m quantscale research \
  --data data/partitions/research.csv \
  --config configs/default.json \
  --topology centralized --fleet-size 4 \
  --backend-command 'python3 -m quantscale.openai_adapter' \
  --output results/llm-centralized-4
```

适配器先通过输入 token 计数端点计算相同提示的输入量，再把本轮剩余额度设为 `max_output_tokens`。返回的实际总用量用于记账，包含不可见的生成 token；截断或非法 JSON 响应仍记入用量，失败候选仍消耗试验预算。输入过大则在推理前停止。没有自动重试，避免不确定用量被隐藏。接口行为依据 [OpenAI token counting 官方文档](https://developers.openai.com/api/docs/guides/token-counting)。真实模型可能仍具有服务端非确定性；代码和调度确定不等于模型输出确定。

OpenAI 适配器使用模拟响应测试；DeepSeek 已进行真实调用，包括开放代码生成与隔离执行验证，日志保存在结果目录。网络或适配器异常会中止实验、不提交冻结策略。已确认用量不保证涵盖网络失败时服务端可能发生的未知用量；这类中止实验不能用于预算对照。

其他模型通过 `--backend-command` 接入。命令以 argv 执行，不经过 shell；每轮从 stdin 读取一个 JSON 请求、向 stdout 输出一个 JSON 对象：

```json
{
  "response": {
    "hypothesis": "Test reversal at a shorter horizon",
    "strategy": {"family": "reversal", "lookback": 5, "skip": 1, "fraction": 0.3, "volatility_scale": true}
  },
  "usage": {"total_tokens": 500},
  "provenance": {"model": "your-fixed-model", "response_id": "provider-response-id"}
}
```

管理器请求的 `role` 为 `manager`，`directions` 将 worker ID 字符串映射为开放式研究任务，仅旧 `rules` 模式使用 family 枚举。开放研究员输出 `hypothesis` 和完整 `strategy_code`；旧模式输出 `hypothesis` 和参数 `strategy`。请求含 `max_total_tokens`，适配器须控制输入、输出与推理用量。事后报告超额会中止实验，不能撤销已发生的推理消耗。不得将估计用量冒充实际 usage，适配器不得自行读取其他研究员或隐藏文件。

## 拓扑与预算口径

- `single`：一个研究员拥有全部预算。
- `independent`：每人仅看到自己的最近三次结果，试验配额平均分配、余数按 ID 分配。
- `centralized`：每轮管理器看到各 worker 最近的有效结果并分配方向；worker 看不到彼此结果。
- `decentralized`：没有管理器，worker 看到上一轮结束时其他研究员的最近有效结果。
- `hybrid`：同时启用管理器方向和上一轮 peer 结果。

Peer 交换使用固定轮次快照，同轮先运行的研究员不会把结果提前泄漏给后运行者。`fleet_size` 表示 worker 数；中心式和混合式额外有一个管理器，管理器的全部用量计入同一个总预算。比较时须保留这一人口定义。

预算单位为**候选试验调用**：一次规则提交，对训练和验证分别回测，共消耗一个 `backtests` 配额；失败/重复候选也消耗配额。底层可能因此执行两次引擎调用。隐藏测试不属于研究预算，仅在冻结后执行。所有管理器和 peer 上下文也计入总 token，通信不是免费的。

`communication_turn_tokens_upper_bound` 表示包含 peer 上下文的完整研究员调用用量加管理器用量，是通信相关调用的用量上界，不是精确分离出来的边际通信 token。Demo 的所有 token 是 UTF-8 字节数除以四的估计；只能检验记账逻辑，不能用于模型成本对比。

## 回测与选择约定

在 t 收盘后生成目标权重，在 t+1 开盘交易，赚取 t+1 开盘到 t+2 开盘收益。每个切分从现金开始，允许使用此前历史预热，只计执行日和收益日均位于该切分内的完整区间。t+1 的开盘跳空不归属于 t 收盘后产生的新目标持仓。上一期持仓在该开盘前已经完成对应的持有收益。

每次换手是目标权重减去上一期经收益与成本漂移后的持仓权重的绝对值之和，涵盖退出股票池的旧仓。成本为该绝对交易名义量乘 `cost_bps / 10000`，不再除以二；借券成本按做空名义量与配置年化天数计算。资金/现金利率为零；不模拟盘中价格路径。期末按市值计价，不额外模拟强制清仓。`total_cost` 与 `total_borrow_cost` 是各日 NAV 比例成本的算术和。

环境拒绝违规仓位，不自动裁剪后继续评分。内置策略对多空两侧统一缩放以满足最大个股仓位，因此股票数量较少时可能无法达到全部 gross limit。指标包括净 Sharpe、复合年化收益、波动率、从初始 NAV 起的最大回撤、下行风险、换手和成本。Sharpe 使用零现金基准，年化因子可配置。

最终按**扣费后验证 Sharpe 降序、策略 SHA-256 升序**确定性选择。所有拓扑使用同一规则，没有 LLM 裁判。这里尚未实现 DSR/PBO、多重检验校正或统计显著性推断；应结合重复运行、预注册和隐藏测试解读结果。按年切片只是描述性统计，不等同于正式市场状态分类。

## 实验矩阵

首轮配置为 SAS、Independent-4、Centralized-4，各 40 个试验、三个种子：

```bash
python3 -m quantscale scaling \
  --data data/partitions/research.csv \
  --config configs/default.json \
  --matrix configs/first-experiment.json \
  --output results/first-experiment
```

矩阵默认使用 DeepSeek Flash，也可通过 `--backend-command` 指定其他模型；加 `--backend demo` 离线运行。扩展矩阵见 `configs/scaling.json`。每个条件独立创建 backend、研究日志与冻结文件，`research_summary.csv` 只含研究/验证结果。隐藏评估由 evaluator 分别执行，矩阵运行器不接收隐藏数据路径。

`budget_mode` 可选 `fixed-total` 与 `fixed-per-agent`。后者将矩阵的试验预算和基础 token 预算都乘 worker 数，管理器与通信仍从该总额中扣除。不同模式不要混为同一个控制实验。默认演示不自动跑完整大矩阵。

## 文件结构

```text
quantscale/
  config.py          配置与时序验证
  data.py            数据、滞后股票池、合成数据、操作员切分
  strategies.py      规则策略和权重约束
  backtest.py        执行、成本、漂移、绩效指标
  research.py        研究调度、五种信息流、全局预算与轨迹
  prompts.py         无具体策略示例的开放研究契约
  code_strategy.py   源码提交、AST 指纹和隔离执行客户端
  strategy_worker.py 仅接收当前及过去行情的隔离代码进程
  openai_adapter.py  Responses API 接口与输入预算预检查
  deepseek_adapter.py DeepSeek Flash、.env 读取与实际用量记账
  evaluation.py      冻结、指纹校验和一次性隐藏评估
  cli.py             demo / prepare / research / evaluate / scaling
configs/             正式日期配置和实验矩阵
tests/               时序、会计、信息边界、预算、冻结与 API 模拟测试
```
