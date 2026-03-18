import os, sys
lib_path = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(lib_path)
import common

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 构建LLM输出解析器，并链接到LLM
str_parser = StrOutputParser()
parser_chain = common.llm | str_parser

# 直接调用模型进行问答
# resp1 = parser_chain.invoke([HumanMessage(content="你好,我是wxj")])
# print(resp1)
# print("----------")
# resp2 = parser_chain.invoke([HumanMessage(content="我是谁？")])
# print(resp2)
# 从上方代码运行结果可以发现，llm不具备记忆功能，进行问题回答时不能够记忆，或者说检索之前的对话内容
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

id2Histroy = {}
def get_session_history(session_id : str)->BaseChatMessageHistory:
    if session_id not in id2Histroy:
        id2Histroy[session_id] = InMemoryChatMessageHistory()
    return id2Histroy[session_id]


chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个外语，用{language}跟我交流"),
    MessagesPlaceholder(variable_name="massage")  
])
chain = chat_prompt | common.llm
# 将链chain加工为一个支持对话历史记忆的链
history_chain = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="massage")
# 确定要调用的历史消息容器
history_id = {"configurable": {"session_id": "abc2"}}
response = history_chain.invoke(
    {"massage":[HumanMessage(content="你好, 我是wxj")],"language":"英语"},
    config = history_id
)
print(response.content)
print("---")
response1 = history_chain.invoke(
    {"massage":[HumanMessage(content="我是谁")],"language":"英语"},
    config = history_id
)
print(response1.content)

