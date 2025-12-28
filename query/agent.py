import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_MODEL_NAME
from query.rag_query_engine import query_engine


@tool
def rag_query(query: str):
    """Query the knowledge base with a user question."""
    return query_engine.query(query)


def build_agent():
    """Build a ReAct agent with access to the query tool."""
    model = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=1,
    )
    tools = [rag_query]
    return create_agent(model, tools)


def build_input(user_query: str) -> dict:
    return {"messages": [("user", user_query)]}


def extract_answer(response: dict) -> str:
    messages = response["messages"]
    final_answer = ""
    for i, m in enumerate(messages):
        match m:
            case ToolMessage():
                print(i, "ToolMessage", m.name, m.content)
            case AIMessage():
                print(i, "AIMessage", m.response_metadata, m.content)
                final_answer = m.content
            case HumanMessage():
                print(i, "HumanMessage", m.content)
            case _:
                print(i, "Unknown message type", m)

    return final_answer


# global agent instance
agent = build_agent()


if __name__ == "__main__":
    user_querys = [
        "What is the responsibility of CNCF Governing Board?",
        # "What is marketing committee?",
    ]
    for user_query in user_querys:
        inputs = build_input(user_query)
        response = agent.invoke(inputs)
        print("-------------------")
        # HumanMessage(...),
        # AIMessage(...),      # tool call
        # ToolMessage(...),    # tool result
        # AIMessage(...)       # final answer
        print(user_query)
        print(extract_answer(response))
        print("-------------------")
