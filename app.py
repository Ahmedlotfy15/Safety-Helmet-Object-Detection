from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from infer import predict_image

app = FastAPI()

@app.post("/predict")
async def predicted(file: UploadFile = File(...)):
    image = Image.open(file.file)
    predicted_image = predict_image(image)

    buffer = io.BytesIO()
    predicted_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")