import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient as MCP_Client
from langchain_core.messages import HumanMessage, BaseMessage
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

async def main():
    client = MCP_Client(
        {
            "math": {
                "transport": "stdio",
                "command": "uv",
                "args": ["run", "10_MCP_Workflows/custom_calculator_mcp.py"],
            }
        }
    )

    tools = await client.get_tools()
    print(f"Dependencies installed? Tools loaded: {[t.name for t in tools]}")
    
   
    llm_with_tools_model = model.bind_tools(tools)


    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    graph = StateGraph(State)

    async def call_model_node(state: State) -> State:
        response = await llm_with_tools_model.ainvoke(state['messages'])
        return {'messages': [response]}

    graph.add_node('call_model', call_model_node)
    graph.add_node('tools', ToolNode(tools))

    graph.add_edge(START, 'call_model')
    graph.add_conditional_edges('call_model', tools_condition)
    graph.add_edge('tools', 'call_model')

    workflow = graph.compile()


    output = await workflow.ainvoke({'messages': [HumanMessage(content="What is 2 + 2?")]})

    print(f"Question: {output['messages'][0].content} \nAnswer: {output['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())