from fastapi import FastAPI
from fastapi.responses import JSONResponse

from query.agent import agent, build_input, extract_answer

app = FastAPI()


@app.get("/")
def root():
    return {"Hello": "World"}


@app.get("/query/{user_query}")
async def query_knowledge_base(user_query: str):
    print("Received query:", user_query)
    input = build_input(user_query)
    try:
        response = await agent.ainvoke(input)
        return JSONResponse(
            status_code=200,
            content={
                "user_query": user_query,
                "final_answer": extract_answer(response),
            },
        )
    except Exception as e:
        print(f"Error processing query: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to process the query", "details": str(e)
            },
        )
