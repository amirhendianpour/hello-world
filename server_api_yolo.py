from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image # type: ignore
import io
from ultralytics import YOLO # type: ignore
import uvicorn

app = FastAPI()

model_ob = YOLO('best_ob.pt')
model_cls = YOLO("best_cls.pt")

class DetectionResultOb(BaseModel):
    class_name: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class DetectionResultCls(BaseModel):
    class_name: str
    confidence: float

@app.get("/hello/")
async def hello():
    return "hello"

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))

    results = model_ob(image,conf=0.1)
    results2 = model_cls(image)

    predictions_ob = []
    predictions_cls = []
    for result in results:
        for box in result.boxes.data:
            x_min, y_min, x_max, y_max, confidence, class_id = box[:6]
            class_name = model_ob.names[int(class_id)]

            predictions_ob.append(DetectionResultOb(
                class_name=class_name,
                confidence=float(confidence),
                x_min=int(x_min),
                y_min=int(y_min),
                x_max=int(x_max),
                y_max=int(y_max)
            ))
    for result in results2:
        top5 = result.probs.top5
        conf5 =result.probs.top5conf    
        for i in range(5):
            predictions_cls.append(DetectionResultCls(
                class_name=model_cls.names[int(top5[i])],
                confidence=float(conf5[i])
            ))

    return {"predictions_ob": [prediction.dict() for prediction in predictions_ob],
             "predictions_cls": [prediction.dict() for prediction in predictions_cls]}

# if __name__ == "__main__":

