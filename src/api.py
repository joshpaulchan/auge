import pathlib
from typing import Sequence, Iterable, Mapping
from io import BytesIO

from fastapi import FastAPI, File
from pydantic import BaseModel

from . import domain

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


class DetectedObject(BaseModel):
    clas: str
    confidence: float
    translations: Mapping

    @classmethod
    def from_model(cls, obj: domain.DetectedObject):
        return cls(
            clas=obj.clas, confidence=obj.confidence, translations=obj.translations
        )


class DetectionResponse(BaseModel):
    objects: Sequence[DetectedObject]


detector = domain.ObjectDetector(pathlib.Path("./ml/imagenet-34.pkl"))
translator = domain.Translator()


@app.post("/detect", response_model=DetectionResponse)
async def detect(output: str, image: bytes = File(None)):
    objects = []
    for obj in detector.detect(BytesIO(image)):
        domain.attach_translation_to_obj(obj, translator=translator, output_lang=output)
        objects.append(DetectedObject.from_model(obj))

    return {"objects": objects}
