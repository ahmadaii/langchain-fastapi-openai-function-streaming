import os
import asyncio

import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent,load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from callback_streaming import MyCallbackHandler

app = FastAPI()

queue = asyncio.Queue()
stream_it = MyCallbackHandler(queue)

# initialize the agent (we need to do this for the callbacks)
llm = ChatOpenAI(
    openai_api_key="YOUR_API_KEY",
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    callbacks=[stream_it]  # ! important (but we will add them later)
)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True,
    output_key="output"
)

@tool
def choose_animal():
    """Chooses a random animal."""
    return "Penguin"

@tool 
def name_pet(animal: str):
    """Generates a name for a pet of the type of animal passed."""
    return 'Mike'

#tools = load_tools(["llm-math"], llm=llm)
tools = [choose_animal, name_pet]

agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    return_intermediate_steps=False
)


async def run_call(query: str):
    # now query
    await agent.acall(inputs={"input": query})

# request input format
class Query(BaseModel):
    text: str

async def create_gen(query: str, stream_it: MyCallbackHandler):
    print("hello?")
    task = asyncio.create_task(run_call(query))
    while True:
        token =  await queue.get()
        print(f't: {token}')
        print("********")
        if token == 'END':
            break;
        yield token
    await task

@app.get("/chat")
async def chat(
    query: str,
):
    gen = create_gen(query, stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")

@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}
    

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="localhost",
        port=8000,
        reload=True
    )
