from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler


class MyCallbackHandler(AsyncCallbackHandler):
    
    def __init__(self, queue):
            self.queue = queue
            self.start = False
            self.buffer =''

    async def on_llm_new_token(self, token, **kwargs) -> None:
        if token is not None:
            #print(token)
            self.buffer += token  # Append the new token to the buffer
            if len(self.buffer) >= 30:
                self.start = True
                await self.queue.put(self.buffer)
                self.buffer = ''  # Clear the buffer

    async def on_llm_end(self, response, **kwargs) -> None:
        if self.start == True:
            if self.buffer:  # If there's anything left in the buffer, put it in the queue
                await self.queue.put(self.buffer)
            await self.queue.put("END")


