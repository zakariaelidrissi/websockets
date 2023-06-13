import asyncio
import pathlib
import ssl, cv2
import websockets, base64

# IMAGE_PATH = "images/left_01062023_144237.jpg" 
# IMAGE_PATH = "images/stop_01062023_144448.jpg"
IMAGE_PATH = "images/forward_01062023_145103.jpg"

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
# localhost_pem = pathlib.Path(__file__).with_name("localhost.pem")
# ssl_context.load_verify_locations(localhost_pem)
uri = "ws://192.168.3.54:8765"

async def hello():
    # uri = "wss://localhost:8765"
    async with websockets.connect(uri) as websocket:
        name = input("What's your name? ")

        await websocket.send(name)
        print(f">>> {name}")

        greeting = await websocket.recv()
        print(f"<<< {greeting}")

# send image to the server
async def predict():
    global uri
    # Read the image file    
    image = cv2.imread(IMAGE_PATH)
    retval, buffer = cv2.imencode('.jpg', image)
    # Encode the image data to base64
    base64_data = base64.b64encode(buffer).decode("utf-8")
    
    # Connect to the WebSocket server
    async with websockets.connect(uri) as websocket:
        # Send the encoded image data
        await websocket.send(base64_data)

        pred_class_name = await websocket.recv()
        print(f"<<< {pred_class_name}")


if __name__ == "__main__":
    asyncio.run(predict())