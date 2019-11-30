import dataclasses
import functools
from typing import Sequence, Iterable, Mapping
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@dataclasses.dataclass
class DetectedObject:
    clas: str = dataclasses.field(default=None)
    translations: Mapping = dataclasses.field(default_factory=dict)
    confidence: float = dataclasses.field(default_factory=0.0)


class ObjectDetector:
    def detect(self, image) -> Iterable[DetectedObject]:
        return []


class Translator:
    def translate(
        self, text: str, output_language: str, input_language: str = "english"
    ) -> str:
        return text


detector = ObjectDetector()
translator = Translator()


def attach_translation_to_obj(output_lang: str, obj: DetectedObject):
    obj.translations.update({output_lang: translator.translate(obj.clas, output_lang)})
    return obj


@app.get("/detect")
async def detect():
    # get image + output languages from input
    image = None
    output = "german"

    objects = detector.detect(image)
    attach_translation = functools.partial(attach_translation_to_obj, output=output)
    translated_objects = map(attach_translation, objects)

    return {"objects": objects}
