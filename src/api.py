import functools
from typing import Sequence, Iterable, Mapping

from fastapi import FastAPI
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


detector = domain.ObjectDetector()
translator = domain.Translator()


@app.get("/detect", response_model=DetectionResponse)
async def detect(output: str):
    # get image + output languages from input
    image = None
    output_lang = output

    objects = []
    for obj in detector.detect(image):
        domain.attach_translation_to_obj(
            obj, translator=translator, output_lang=output_lang
        )
        objects.append(DetectedObject.from_model(obj))

    return {"objects": objects}
