import os, sys
lib_path = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(lib_path)
import common

from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage,SystemMessage

chat_prompt_temp = ChatPromptTemplate.from_messages([
    SystemMessage("你是一个炒股大神，能够帮我准确地分析股市的行情和变化"),
    HumanMessage("{question}")
    ])


# 创建代理
from langchain.agents import create_agent
agent = create_agent(
    model=common.llm,
    tools=[search],
    system_prompt=SystemMessage("你是一个炒股大神，能够帮我准确地分析股市的行情和变化")
)

# 调用代理
response = agent.stream(
    {"messages":[HumanMessage("帮我分析一下A股机器人板块的未来走势")]}
)
for chunk in response:
    print(chunk)


