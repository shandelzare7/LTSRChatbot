"""验证 with_structured_output 的 invoke kwargs 是否真的传到 API。"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
try:
    from utils.env_loader import load_project_env
    load_project_env(Path(os.path.join(os.path.dirname(__file__), "..")))
except Exception:
    pass

import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

class Reply(BaseModel):
    reply: str = Field(description="回复")

qwen_key = os.environ.get("QWEN_API_KEY", "")
llm = ChatOpenAI(
    model="qwen-plus-latest",
    api_key=qwen_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
)

msgs = [SystemMessage(content="你好"), HumanMessage(content="测试")]

print("=" * 60)
print("TEST 1: structured.invoke 传 temperature=1.1")
print("=" * 60)
structured = llm.with_structured_output(Reply)
result = structured.invoke(msgs, temperature=1.1, top_p=0.95, frequency_penalty=0.4)
print(f"Result: {result}")

print()
print("=" * 60)
print("TEST 2: 直接 llm.invoke 传 temperature=1.1")
print("=" * 60)
result2 = llm.invoke(msgs, temperature=1.1, top_p=0.95, frequency_penalty=0.4)
print(f"Result: {result2.content[:100]}")
