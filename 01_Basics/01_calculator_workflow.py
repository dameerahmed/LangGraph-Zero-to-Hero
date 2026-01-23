from langgraph.graph import StateGraph, START, END
from typing import  TypedDict


class Calculator(TypedDict):
    num1: int
    num2: int
    op: str
    result: float

def add(state: Calculator) -> Calculator:
    state['result'] =state['num1'] + state['num2']
    return state

def subtract(state: Calculator) -> Calculator:
    state['result'] =state['num1'] - state['num2']
    return state

def multiply(state: Calculator) -> Calculator:
    state['result'] =state['num1'] * state['num2']
    return state

def divide(state: Calculator) -> Calculator:
    state['result'] =state['num1'] / state['num2']
    return state

def condition(state: Calculator) -> Calculator:
    if state['op'] == '+':
        return 'add'
    elif state['op'] == '-':
        return 'subtract'
    elif state['op'] == '*':
        return 'multiply'
    elif state['op'] == '/':
        return 'divide'

graph = StateGraph(Calculator)

graph.add_node('add', add)
graph.add_node('subtract', subtract)
graph.add_node('multiply', multiply)
graph.add_node('divide', divide)


graph.add_conditional_edges(
    START, 
    condition,
)

graph.add_edge('add', END)
graph.add_edge('subtract', END)
graph.add_edge('multiply', END)
graph.add_edge('divide', END)

workflow = graph.compile()

output = workflow.invoke({
    'num1': 1,
    'num2': 2,
    'op': '-'})

print(f"{output['num1']} {output['op']} {output['num2']} = {output['result']}")
