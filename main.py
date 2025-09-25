from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import app as query_agent
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Query Decomposition and Refinement Agent",
    description="An agent that decomposes user questions, generates a search query, and refines it through critique.",
    version="1.0.0",
)

# Pydantic model for the request body
class QueryRequest(BaseModel):
    query: str

# Define the API endpoint
@app.post("/process-query/")
async def process_query(request: QueryRequest):
    """
    Receives a user query, processes it through the LangGraph agent,
    and returns the final, refined search query.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Prepare the initial state for the graph
    initial_state = {
        "original_query": request.query,
        "revision_count": 0, # Initialize revision count
    }

    try:
        # Run the agent
        final_state = query_agent.invoke(initial_state)

        # Check for errors during the process
        if final_state.get("error"):
             raise HTTPException(status_code=500, detail=f"Agent error: {final_state['error']}")

        # Return the final search query
        return {
            "original_query": final_state["original_query"],
            "final_search_query": final_state["search_query"],
            "decomposed_details": final_state["decomposed_query"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Add a root endpoint for health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Query Agent is running"}

# To run this app, use the command: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)