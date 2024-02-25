import argparse
import asyncio
import websockets

from chatbot import AIChatBot

def main(args):
    # Initialize your chatbot
    e2e_chatbot = AIChatBot(args)
    e2e_chatbot.build_bot()

    # Define the WebSocket server
    async def chat_handler(websocket, path):
        async for message in websocket:
            query = message.strip()
            print("Received query:", query)
            # Process the query using your chatbot
            response = e2e_chatbot.process_query(query)
            # Ensure response is a string
            response = str(response)
            # Send the response back to the client
            await websocket.send(response)

    # Start the WebSocket server
    start_server = websockets.serve(chat_handler, "localhost", 8765)

    # Run the WebSocket server
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

system_prompt = """
    You are an AI chatbot that is designed to answer questions related to E2E Networks. 
    You are provided with a context and a question. You need to answer the question based on the context provided. 
    If the context is not helpful, do not answer based on prior knowledge, instead, redirect the user to the E2E Networks Support team. 
    You should also provide links that you got from context that are relevant to the answer. 
    You are allowed to answer in first person only, like I/Us/Our; It should feel like a human is answering the question. 
    You should only provide the links and not like [E2E Networks Official Website](https://www.e2enetworks.com/)
    You're not allowed to say something like "Based on the context, I think the answer is...", instead, you should directly answer the question.
    When in confusion, you can ask for more information from the user.

    Here is an example of how you should answer:

    Question: What is the pricing for E2E Networks?
    Context: E2E Networks is a cloud computing company that provides cloud infrastructure and cloud services to businesses and startups.
    Unacceptable Answer: Based on the context, I think the pricing for E2E Networks is...
    Acceptable Answer: The pricing for E2E Networks is...
"""

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help="Model name to use for the LLM")
argparser.add_argument("--embedding_model_name", type=str, default="BAAI/bge-large-en-v1.5", help="Model name to use for the embedding model")
argparser.add_argument("--context_window", type=int, default=4096, help="Context window for the LLM")
argparser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for the LLM")
argparser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size in which to process the input")
argparser.add_argument("--chunk_overlap", type=int, default=256, help="Chunk overlap in which to process the input")
argparser.add_argument("--device", type=str, default="auto", help="Device to use for the LLM")
argparser.add_argument("--database_connection_string", type=str, default="postgresql://postgres:test123@localhost:5432", help="Database connection string")
argparser.add_argument("--database_name", type=str, default="chatbotdb", help="Database name")
argparser.add_argument("--table_name", type=str, default="companyDocEmbeddings", help="Table name")
argparser.add_argument("--system_prompt", type=str, default=system_prompt, help="System prompt to use for the LLM")
argparser.add_argument("--top_k_index_to_return", type=int, default=16, help="Top k index to return for the LLM")
args = argparser.parse_args()

if __name__ == "__main__":
    main(args)











