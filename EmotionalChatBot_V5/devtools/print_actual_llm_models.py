"""
打印当前环境（.env + 环境变量）下各节点实际使用的 LLM 模型与参数。
用法：在 EmotionalChatBot_V5 目录下执行
  python devtools/print_actual_llm_models.py
  或 .venv/bin/python devtools/print_actual_llm_models.py
"""
import os
import sys
from pathlib import Path

# 加载 .env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

# 与 graph.py 相同：模型来自 config/llm_models.yaml，API Key 来自 .env
from app.services.llm import get_llm
from utils.yaml_loader import load_llm_models_config


def _model_of(llm) -> str:
    """从可能被包装的 LLM 实例取出实际模型名。"""
    if llm is None:
        return "N/A"
    m = getattr(llm, "model_name", None) or getattr(llm, "_model", None)
    if m:
        return str(m)
    inner = getattr(llm, "_inner", None)
    if inner is not None:
        return _model_of(inner)
    return "unknown"


def main():
    print("=" * 60)
    print("各节点实际使用的模型与参数（按当前 .env / 环境变量解析）")
    print("=" * 60)

    # 与 graph.py 完全一致的调用
    llm_safety = get_llm(role="fast", temperature=0.05)
    llm_detection = get_llm(role="fast", temperature=0.1)
    llm_monologue = get_llm(role="main")
    llm_extract = get_llm(role="fast", temperature=0.1)
    llm_processor = get_llm(role="fast", temperature=0.3)
    llm_evolver = get_llm(role="fast", temperature=0.18)
    llm_memory_manager = get_llm(role="fast", temperature=0.1)
    llm_fast_safety_reply = get_llm(role="main", temperature=0.55)
    llm_judge = get_llm(role="main")

    _llm_cfg = load_llm_models_config()
    _gen_cfg = (_llm_cfg.get("generate") or {}) if isinstance(_llm_cfg.get("generate"), dict) else {}
    _gen_model = (_gen_cfg.get("model") or "").strip() or "qwen-plus-2024-12-20"
    _gen_base_url = (_gen_cfg.get("base_url") or "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    _gen_api_key = (os.getenv("LTSR_GEN_API_KEY") or "").strip() or None
    if _gen_model and "qwen" in _gen_model.lower():
        _gen_api_key = _gen_api_key or (os.getenv("QWEN_API_KEY") or "").strip() or None
    _gen_temperature = float(_gen_cfg.get("temperature", 1.0)) if _gen_cfg else 1.0
    _gen_n = int(_gen_cfg.get("n", 4)) if _gen_cfg else 4
    llm_gen = get_llm(
        role="fast",
        model=_gen_model,
        api_key=_gen_api_key,
        base_url=_gen_base_url or None,
        temperature=_gen_temperature,
        top_p=float(_gen_cfg.get("top_p", 0.95)) if _gen_cfg else 0.95,
        presence_penalty=float(_gen_cfg.get("presence_penalty", 0.3)) if _gen_cfg else 0.3,
        n=_gen_n,
    )

    rows = [
        ("safety", llm_safety, 0.05, "-"),
        ("fast_safety_reply", llm_fast_safety_reply, 0.55, "-"),
        ("detection", llm_detection, 0.1, "-"),
        ("inner_monologue", llm_monologue, None, "默认 0.3"),
        ("extract", llm_extract, 0.1, "-"),
        ("generate", llm_gen, _gen_temperature, f"presence_penalty=0.3, n={_gen_n}"),
        ("judge", llm_judge, None, "默认 0.3"),
        ("processor", llm_processor, 0.3, "-"),
        ("evolver", llm_evolver, 0.18, "-"),
        ("memory_manager", llm_memory_manager, 0.1, "-"),
    ]

    print(f"\n{'节点':<22} {'实际模型':<28} {'temperature':<12} 其他")
    print("-" * 78)
    for name, llm, temp, other in rows:
        model = _model_of(llm)
        temp_str = f"{temp}" if temp is not None else "(默认)"
        print(f"{name:<22} {model:<28} {temp_str:<12} {other}")
    print("=" * 60)
    print("说明: 未配置 API Key 时对应节点会使用 MockLLM，模型名为 mock。")
    print("      模型与 base_url 统一在 config/llm_models.yaml 修改；API Key 在 .env。")
    print("=" * 60)


if __name__ == "__main__":
    main()
