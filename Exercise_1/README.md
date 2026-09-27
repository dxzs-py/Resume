# huida — 手写 Agent Runtime

基于 DeepSeek API 从零实现的 Agent Runtime：**不依赖任何 Agent 框架**（无 LangChain / LangGraph / AutoGen / OpenHands），主循环、工具调度、会话管理、上下文压缩全部手工实现，仅使用 `openai` SDK 做 LLM 的 HTTP 调用。

## 功能对照


| 题目要求                                                            | 实现                                                                                     |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 手写 Agent Runtime                                                  | `agent/runtime.py`，循环控制完全手工实现                                                 |
| 基本循环（接收输入→判断直答/调工具→执行工具→依据结果继续或返回） | `AgentRuntime.run()`                                                                     |
| ≥3 个工具 + 注册机制（名称/描述/参数 Schema）                      | calculator（ast 白名单安全求值）、search（mock 知识库）、weather（mock），`ToolRegistry` |
| LLM 基于 Schema 自主决策                                            | schema 以`tools` 参数注入请求，不设 `tool_choice`（默认 auto）                           |
| 输出解析（思考过程/工具调用/最终答案）                              | `agent/parser.py`                                                                        |
| Session 管理与多窗口隔离                                            | 每 session 一个 JSON 文件，`--resume` 恢复                                               |
| Context 管理（轮次上限/追问/基础压缩）                              | `agent/context/manager.py`                                                               |
| 异常处理                                                            | 自定义异常体系 + LLM 指数退避重试 + 工具错误回传自我纠正                                 |
| 工具调用 trace / 执行日志                                           | `logs/trace.jsonl`（JSONL 事件流）+ logging                                              |

## 运行方式

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入真实密钥（项目不附带 `.env`，密钥不入库）：

```ini
DEEPSEEK_API_KEY=sk-你的真实key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

其余均可选：`AGENT_MAX_ITERATIONS`（单次任务最大轮数，默认 8）、`CONTEXT_MAX_MESSAGES`（触发压缩的消息条数阈值，默认 40）、`CONTEXT_KEEP_RECENT`（压缩后保留的最近消息条数，默认 12）、`LLM_TIMEOUT`、`LLM_MAX_RETRIES`、`SESSIONS_DIR`、`trace_file`。

### 3. 启动

```bash
python main.py                  # 新建会话进入交互
python main.py --list           # 列出全部历史会话
python main.py --resume <id>    # 恢复指定会话继续聊（多窗口场景）
python main.py --session-id <id>  # 以自定义 id 新建
```

会话内命令：`/new` 新建会话、`/history` 查看当前会话历史、`/sessions` 列出全部会话、`/exit` 退出。

### 4. 运行测试

```bash
python -m pytest tests/ -v
```

测试共 57 个用例，LLM 使用 fake 实现，**无需真实 API key**，可离线运行。

## 系统设计

### 架构总览

```
main.py (CLI/REPL)
   │
   ▼
AgentRuntime ──────► 主循环（手写）
   │    │    │
   │    │    └── ContextManager ──► 组装 messages（system+摘要+滑动窗口）
   │    └─────── ToolRegistry ────► 注册/Schema 导出/执行
   │               ├─ calculator（ast 白名单求值，禁 eval）
   │               ├─ search（mock 知识库检索）
   │               └─ weather（mock 天气）
   └──────────── LLMClient ───────► DeepSeek API（指数退避重试）
   │
   ▼
SessionStore ─────► sessions/<id>.json（每会话一个文件，落盘持久化）
logs/trace.jsonl ─► 全事件 trace（输入/请求/响应/工具/统计）
```

### 主循环（`AgentRuntime.run`）

1. **接收输入**：用户输入写入 session；
2. **决策**：把 `system + 历史 + 工具 Schema` 发给 LLM，解析响应——有 `tool_calls` 进入工具分支，否则 `content` 即最终答案；
3. **执行工具**：按 `tool_call_id` 逐个执行，结果以 `role="tool"` 消息回填 session；
4. **继续或返回**：回到第 2 步，直到 LLM 给出最终答案，或达到 `max_iterations` 返回系统提示。

关键协议约束：assistant 的 `tool_calls` 消息与 `role="tool"` 结果消息必须**成对相邻**（OpenAI 协议硬性要求，缺对下次请求直接 400）。runtime 在每次工具执行后立即回填，保证消息序列任意时刻合法，落盘后可被 `--resume` 安全重放。

### 工具机制

- 每个工具 = `Tool(name, description, parameters, func)`，`parameters` 为标准 JSON Schema；
- `ToolRegistry.register()` 登记（重名抛异常防静默覆盖）、`export_schemas()` 导出给 LLM、`execute()` 统一执行；
- **错误回传而非中断**：工具失败（缺参、异常、LLM 幻觉出未注册工具名、参数非法 JSON）一律包装为 `"Error: ..."` 字符串作为工具结果回传，LLM 可据此调整参数重试，而不是让整轮循环崩溃。

### 解析器

LLM 输出按三种情形解析：`content + tool_calls`（content 视为思考过程 thought）、仅 `tool_calls`（纯动作）、仅 `content`（最终答案）。参数 JSON 解析失败时保留 id/name 并记录 `parse_error`，由 runtime 以错误文本回传促其自我纠正。

### Session 管理（多窗口隔离）

- 用户 A 的"窗口 1（查天气记待办）"与"窗口 2（写周报记待办）"= 两个独立 `Session`，各自持有独立 `session_id` 与消息历史，各自落盘为 `sessions/<id>.json`，互不共享状态；
- 随时 `--resume <id>` 或 `/sessions` 后切换回任一窗口继续对话，历史完整恢复；
- 文件损坏只影响单个会话，`list_sessions` 自动跳过并告警。

### Context / Memory 管理

**召回时机与放置方式**：


| 信息               | 放置方式                                             | 召回时机                           |
| ------------------ | ---------------------------------------------------- | ---------------------------------- |
| System Prompt      | 恒置顶（第一条 system 消息）                         | 每轮 LLM 请求                      |
| 用户输入           | 追加到 session 历史尾部，`role="user"`               | 每轮 LLM 请求                      |
| Agent 思考过程     | 存为 assistant 消息的`content`（带 `tool_calls` 时） | 每轮 LLM 请求                      |
| 工具调用与结果     | `assistant.tool_calls` + `role="tool"` 成对消息      | 每轮 LLM 请求                      |
| 更早历史（超限后） | 压缩为一条`[历史摘要]` system 消息                   | 仅当历史超过`CONTEXT_MAX_MESSAGES` |

- **召回时机**：不在消息写入时做任何裁剪，Session 始终保存全量历史（即长期记忆）；在**每轮循环即将调用 LLM 前**，由 `ContextManager.build_messages()` 现场组装实际发送的消息。这样"存储全量、发送按需"，追问时历史永远可完整恢复。
- **放置方式**：`system prompt` →（超限时）`[历史摘要]` → 最近 `keep_recent` 条消息。system 恒置顶保证全局指令不受截断影响。
- **基础压缩**：超过 `max_messages` 时启用滑动窗口，只保留最近 `keep_recent` 条；窗口外的旧消息用启发式摘要（抽取早期 user 消息要点，至多 10 条）生成占位文本。窗口起点会**回退到非 tool 消息**处，保证工具调用配对不被切断。
- **追问支持**：纯对话追问依赖历史里的 user/assistant 消息；带工具的追问依赖成对保留的 tool_calls/tool 结果——LLM 能看到"之前调过什么工具、拿到了什么结果"，自主决定是否重新调用。
- **轮次上限**：`max_iterations` 限制单次任务的最大"LLM 调用-解析-执行"循环轮数，防止无限循环烧 token；轮次耗尽返回明确系统提示。

### 异常处理与日志

- 自定义异常体系：`AgentError`（基类）/ `LLMError` / `ToolNotFoundError` / `ToolExecutionError`；
- LLM 暂时性错误（超时/网络/429）按 2^n 秒指数退避重试，确定性错误（401/400）快速失败；
- trace：每次 run 的输入、LLM 请求/响应、工具调用/结果（含耗时、是否错误）、运行统计逐条追加到 `logs/trace.jsonl`，trace 写入失败不影响主流程。

## 目录结构

```
huida/
├── main.py                  # CLI 入口（REPL、多会话命令）
├── requirements.txt
├── .env.example             # 配置模板（复制为 .env 并填 key）
├── agent/
│   ├── runtime.py           # 手写主循环
│   ├── llm_client.py        # DeepSeek 客户端 + 重试
│   ├── parser.py            # 输出解析（思考/工具调用/答案）
│   ├── prompts.py           # System Prompt
│   ├── config.py            # .env 加载与配置
│   ├── exceptions.py        # 异常体系
│   ├── logger.py            # logging + trace.jsonl
│   ├── context/manager.py   # 上下文组装与压缩
│   ├── session/store.py     # Session 模型与持久化
│   └── tools/               # base(注册中心) + calculator/search/weather
├── tests/                   # 57 个测试用例（fake LLM，离线可跑）
└── AI_PROMPT_LOG.md    # AI Prompt 与问题解决记录
```

## 已知限制

- 历史压缩为启发式摘要（抽 user 消息要点），未做 LLM 语义摘要；
- 无流式输出；工具为同步串行执行，无并行；
- 会话存储为单机 JSON 文件，无并发写保护，适合单用户场景。
