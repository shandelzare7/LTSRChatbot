"""
测试当前配置的 API Key 是否可用。
会加载 .env 并使用与主流程相同的 get_llm(role="main") 发一条简单请求。

运行：cd EmotionalChatBot_V5 && python -m devtools.test_api_key
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

def main():
    key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_OPENAI") or "").strip()
    base = (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "").strip() or "默认(OpenAI)"
    model = (os.getenv("OPENAI_MODEL") or os.getenv("LTSR_LLM_MAIN_MODEL") or "").strip() or "gpt-4o"

    print("========== API Key 测试 ==========")
    print(f"  OPENAI_API_KEY: {'已设置 (' + key[:8] + '...)' if key else '未设置'}")
    print(f"  BASE_URL: {base}")
    print(f"  MODEL: {model or '(使用 get_llm 默认)'}")
    print()

    if not key:
        print("结果: 未配置 API Key，请设置 OPENAI_API_KEY 后重试。")
        return 1

    from langchain_core.messages import HumanMessage
    from app.services.llm import get_llm

    llm = get_llm(role="main")
    if getattr(llm, "_inner", None) is not None:
        inner = getattr(llm, "_inner", llm)
    else:
        inner = llm
    if type(inner).__name__ == "MockLLM":
        print("结果: 当前返回的是 MockLLM，未使用真实 API。请确认 .env 中 OPENAI_API_KEY 已正确加载。")
        return 1

    print("发送一条测试请求...")
    try:
        resp = llm.invoke([HumanMessage(content="回复一个字：好")])
        content = getattr(resp, "content", str(resp)).strip()
        print(f"结果: 成功")
        print(f"回复: {content[:200]}")
        return 0
    except Exception as e:
        err_type = type(e).__name__
        print(f"结果: 失败")
        print(f"异常: {err_type}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
