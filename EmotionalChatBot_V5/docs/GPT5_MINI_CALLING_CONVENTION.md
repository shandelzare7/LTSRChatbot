# gpt-5-mini 正确调用规范

gpt-5-mini 是 OpenAI reasoning 模型。与 gpt-4o / gpt-4o-mini 等常规模型的 API 参数**不兼容**，必须按本文档的方式调用。

---

## 1. 不支持的参数（禁止传）

以下参数在 reasoning 模型上**不支持**，传了会报错或被忽略：

- `temperature`
- `top_p`
- `frequency_penalty`
- `presence_penalty`

---

## 2. 支持的专属参数

| 参数 | 类型 | 可选值 | 说明 |
|------|------|--------|------|
| `verbosity` | string | `"low"` / `"medium"` / `"high"` | 控制回复长度。low 更简洁，high 更详细。默认 medium。 |
| `reasoning_effort` | string | `"none"` / `"minimal"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` | 控制推理深度。low 更快更省 token，high 推理更完整。默认 medium。 |
| `max_tokens` | int | - | 输出 token 上限，与常规模型相同。 |

---

## 3. 本项目的正确调用方式

### 唯一配置点：`app/services/llm.py` → `get_llm()`

```python
# get_llm() 内部，当检测到 reasoning 模型时：
_is_reasoning_model = model_name and "gpt-5" in str(model_name).lower()

kwargs = {"model": model_name, "api_key": key}

if _is_reasoning_model:
    # 不传 temperature
    kwargs["verbosity"] = "low"
    kwargs["reasoning_effort"] = "low"
else:
    kwargs["temperature"] = temperature
```

这两个参数必须作为 `ChatOpenAI` 的**显式构造参数**传入：

```python
llm = ChatOpenAI(
    model="gpt-5-mini",
    api_key=key,
    verbosity="low",
    reasoning_effort="low",
    # 不传 temperature / top_p / penalty
)
```

**不要**放进 `model_kwargs`：

```python
# ❌ 错误写法——LangChain 会发出警告，且参数可能丢失
llm = ChatOpenAI(
    model="gpt-5-mini",
    model_kwargs={"verbosity": "low", "reasoning_effort": "low"},
)
```

### 调用处（如 reply_planner）：不需要重复传

构造时传入的参数会存入 `ChatOpenAI._default_params`，**每次 API 调用都会自动带上**，包括：

- 直接 `llm.invoke(messages)`
- `llm.with_structured_output(schema).invoke(messages)`

因此 invoke 时**不需要再传** `verbosity` / `reasoning_effort`：

```python
# ✅ 正确——只传 invoke 专属参数
invoke_kwargs = {"max_tokens": 2000}
structured.invoke(messages, **invoke_kwargs)

# ❌ 冗余——verbosity/reasoning_effort 已在构造时设置，不需要重复
invoke_kwargs = {"max_tokens": 2000, "verbosity": "low", "reasoning_effort": "low"}
```

---

## 4. 参数生效链路

```
get_llm(model="gpt-5-mini")
  └─ ChatOpenAI(verbosity="low", reasoning_effort="low")
       └─ 存入实例属性 + _default_params
            ├─ llm.invoke() → _default_params 合并到 API body ✅
            └─ llm.with_structured_output(schema).invoke()
                 └─ 链中 LLM 实例不变，_default_params 保留 ✅
```

已通过测试脚本 `devtools/check_verbosity_gpt5mini.py` 验证：

1. `_default_params` 包含 `verbosity: "low"`, `reasoning_effort: "low"` ✅
2. `_is_planner_gpt5_mini()` 正确识别 ✅
3. `with_structured_output` 链中 LLM 的 `_default_params` 保留 ✅
4. 实际 HTTP 请求中 `verbosity` 出现 ✅

---

## 5. 使用处一览

| 调用方 | 文件 | LLM 实例 | 说明 |
|--------|------|----------|------|
| reply_planner（15 候选生成） | `app/lats/reply_planner.py` | `llm_planner_27`（graph.py 创建） | 5 路并行 × 3 档，通过 `with_structured_output` 调用 |
| graph.py 创建 | `app/graph.py` L83 | `get_llm(role="fast", model="gpt-5-mini")` | 唯一创建点 |

---

## 6. 如何判断是否为 reasoning 模型

```python
def _is_reasoning_model(model_name: str) -> bool:
    return bool(model_name and "gpt-5" in str(model_name).lower())
```

如果未来有新的 reasoning 模型（如 o3、o4-mini 等），需要扩展此判断。

---

## 7. 常见错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 构造时传 `temperature` | API 报错 | 不传 temperature |
| 放进 `model_kwargs` | LangChain 警告 + 参数可能丢失 | 用显式构造参数 |
| invoke 时重复传 `verbosity` | 冗余（不报错但多余） | 只传 `max_tokens` |
| 在节点内硬编码 `verbosity` | 违反 `.cursorrules`（参数只在 graph/llm.py 配置） | 由 `get_llm()` 统一设置 |
