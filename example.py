from graph_rag_agent import agent_executor

# Example usage
response = agent_executor.invoke({
    "input": "Create a person named 'Alice' and another person named 'Bob', and then create a 'FRIENDS' relationship between them."
})
print(response)

response = agent_executor.invoke({
    "input": "Find the person named 'Alice'."
})
print(response)
