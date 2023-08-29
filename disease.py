from fastapi import FastAPI, UploadFile, File
import PIL
import cv2
import numpy as np
import joblib


app = FastAPI()


model = joblib.load("rice.pkl")

@app.get("/")
def index():
    return {"message":"hello world from aakash"}

@app.post("/predict_image")
async def predict_image(file: UploadFile):
    img_pil = PIL.Image.open(file.file)
    
    # Convert the image bytes to a PIL Image
    # img_pil = PIL.Image.open(io.BytesIO(img_pil))

    # Convert the PIL Image to a NumPy array
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_cv2, (180, 180))

    # Preprocess the image for prediction
    test = np.expand_dims(img_resized, axis=0)
    test = test / 255.0

    # Make a prediction using the model
    prediction = model.predict(test)
    predicted_class = np.argmax(prediction, axis=1)

    return {"predicted_class": int(predicted_class)}
    # return {"predicted_class": int(predicted_class)}



