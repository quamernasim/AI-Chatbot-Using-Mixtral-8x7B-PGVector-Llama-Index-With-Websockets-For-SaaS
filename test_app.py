import argparse
import asyncio
import websockets

# Function to send a message and receive a response from the WebSocket server
async def send_message(url, message):
    async with websockets.connect(url) as websocket:
        await websocket.send(message)
        response = await websocket.recv()
        return response 

def main(args):
    # Send a message to the WebSocket server and receive a response
    response = asyncio.get_event_loop().run_until_complete(send_message(args.url, args.message))
    print("Response from server:", response)

argparser = argparse.ArgumentParser()
argparser.add_argument("--url", type=str, default="ws://localhost:8765", help="URL to connect to the WebSocket server")
argparser.add_argument("--message", type=str, default="Hello, world!", help="Message to send to the server")
args = argparser.parse_args()

if __name__ == "__main__":
    main(args)
