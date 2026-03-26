from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from pydantic import SecretStr
import os, sys
# os.environ["OPENAI_API_KEY"] = "sk-0447fbe1a6fb43d38cb83123bf7a0fd0"

# 实例化一个对话模型
llm = ChatOpenAI(
    model = "qwen3-max",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = SecretStr("sk-0447fbe1a6fb43d38cb83123bf7a0fd0"),
)

# 调用本地大模型
llm_local = ChatOllama(model="qwen3.5:0.8b")