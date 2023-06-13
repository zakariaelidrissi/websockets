import asyncio
import pathlib
import ssl
import websockets
import base64, cv2, numpy as np
from ultralytics import YOLO

MODEL_PATH = 'model/best.pt'

model = YOLO(MODEL_PATH)  # load a custom model

async def hello(websocket):
    name = await websocket.recv()
    print(f"<<< {name}")

    greeting = f"Hello {name}!"

    await websocket.send(greeting)
    print(f">>> {greeting}")

# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# localhost_pem = pathlib.Path(__file__).with_name("localhost.pem")
# ssl_context.load_cert_chain(localhost_pem)


async def receive_image(websocket):
    async for message in websocket:
        # Decode the base64 image data
        image_data = base64.b64decode(message)        
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Do further processing with the image as needed
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # predict an image
        results = model(img)
        names_dict = results[0].names
        probs = results[0].probs.tolist()
        pred_class_name = names_dict[np.argmax(probs)]

        await websocket.send(pred_class_name)
        print(f">>> {pred_class_name}")


async def main():
    async with websockets.serve(receive_image, "192.168.3.54", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())


# run the server with watchmedo - mode : auto-restart
# watchmedo auto-restart --pattern "*.py" --recursive --signal SIGTERM python server.py