# Deliberative LLM Solver

Planner → Solver multi-agent architecture using:

- LangGraph
- LangChain
- AWS Bedrock Claude 3
- Streaming output

## How it works

1. Planner generates reasoning steps
2. Solver follows plan and outputs final answer
3. Streaming enabled for both nodes

## Run

```bash
python main.py