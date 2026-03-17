import getpass
import os, sys
from fastapi import FastAPI
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
lib_path = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
print(lib_path)
sys.path.append(lib_path)

import common
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langserve import add_routes

# message = [
    # SystemMessage(content="你是一个数学计算专家"),
#     HumanMessage(content="你好")
# ]

# resp = common.llm.invoke(message)
# llm运行后返回的是Aimessage对象，包含字符串信息和其他的一些元信息。
# 下方直接将所有的信息进行打印
# print(resp.content)
# for chunk in resp:
#     print(chunk.content, end="")

# 定义一个输出解析器
# print("============================================")
parser = StrOutputParser()
# 下方直接用一个输出解析器打印llm响应
# print(parser.invoke(resp))

# 下方将解析器和llm进行链接
# print("============================================")
# chain = common.llm | parser
# chain_resp = chain.invoke(input=message)
# print(chain_resp)

# 提示词设计
system_prompt = "请将下文翻译为{lang}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_prompt),
     ("user","{question}")]
)
prompt = prompt_template.invoke({"lang":"英语", "question":"我真的服了。。。"})
# print(prompt.to_messages)

total_chain = prompt_template | common.llm | parser

# response = total_chain.invoke(input={"lang":"英语", "question":"我真的服了。。。"})
# print(response)

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    total_chain,
    path="/chain",
)

if __file__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)



