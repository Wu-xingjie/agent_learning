from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 实例化一个对话模型
llm = ChatOpenAI(
    model = "qwen3-max",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = SecretStr("sk-0447fbe1a6fb43d38cb83123bf7a0fd0"),
)
