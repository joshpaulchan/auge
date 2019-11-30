import dataclasses
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
    clas: str = dataclasses.field(default=None)
    translations: Mapping = dataclasses.field(default_factory=dict)
    confidence: float = dataclasses.field(default_factory=0.0)


class DetectionResponse(BaseModel):
    objects: Sequence[DetectedObject]


detector = domain.ObjectDetector()
translator = domain.Translator()


@app.get("/detect", response_model=DetectionResponse)
async def detect(output: str):
    # get image + output languages from input
    image = None

    objects = detector.detect(image)
    attach_translation = functools.partial(
        domain.attach_translation_to_obj, translator=translator, output=output
    )
    for obj in objects:
        attach_translation(obj)

    return {"objects": objects}
