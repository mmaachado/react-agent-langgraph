import operator
from dataclasses import dataclass
from typing import Annotated, Literal, Literals

from langgraph.constants import END, START
from langgraph.graph import StateGraph


@dataclass
class State:
	nodes_path: Annotated[list[str], operator.add]
	current_number: int = 0


def node_a(state: State) -> State:

	output_state: State = State(nodes_path=['A'], current_number=state.current_number)
	print('> node_a', f'{state=}', f'{output_state=}')
	return output_state


def node_b(state: State) -> State:

	output_state: State = State(nodes_path=['B'], current_number=state.current_number)
	print('> node_b', f'{state=}', f'{output_state=}')
	return output_state


def node_c(state: State) -> State:

	output_state: State = State(nodes_path=['C'], current_number=state.current_number)
	print('> node_c', f'{state=}', f'{output_state=}')
	return output_state


def conditional(state: State) -> Literal['B', 'C']:
	if state.current_number >= 50:
		return 'C'
	return 'B'


builder = StateGraph(State)

builder.add_node('A', node_a)
builder.add_node('B', node_b)
builder.add_node('C', node_c)


builder.add_edge(START, 'A')
builder.add_conditional_edges('A', conditional, ['B', 'C'])
builder.add_edge('B', END)
builder.add_edge('C', END)

graph = builder.compile()

response = graph.invoke(State(nodes_path=[]))

print()
print(f'{response=}')
print()
