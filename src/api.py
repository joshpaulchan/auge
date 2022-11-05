import pathlib
from typing import Sequence, Iterable, Mapping
from io import BytesIO

import tempfile

from fastapi import FastAPI, UploadFile
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


detector = domain.PytorchObjectDetector(pathlib.Path("./ml/imagenet-34.pkl"))
translator = domain.Translator()


@app.post("/detect", response_model=DetectionResponse)
async def detect(output: str, image: UploadFile):
    objects = []

    with tempfile.NamedTemporaryFile(delete=False) as temp_image:
        temp_image.write(await image.read())

        objects = detector.detect(temp_image.name)
        
        for obj in objects:
            domain.attach_translation_to_obj(obj, translator=translator, output_lang=output)
            objects.append(DetectedObject.from_model(obj))

    return {"objects": objects}
