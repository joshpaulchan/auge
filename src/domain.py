import dataclasses
import functools
import pathlib

from fastai import vision

from typing import Sequence, Iterable, Mapping


@dataclasses.dataclass
class DetectedObject:
    clas: str = dataclasses.field(default=None)
    confidence: float = dataclasses.field(default=0.0)
    translations: Mapping = dataclasses.field(default_factory=dict)

    @classmethod
    def from_prediction(cls, clas, confidence):
        return cls(clas, confidence)


class ObjectDetector:
    def __init__(self, path_to_model: pathlib.Path):
        self.model = self._init_model(path_to_model)

    def detect(self, image: bytes) -> Iterable[DetectedObject]:
        if not image:
            return []
        return []

    def _init_model(self, path_to_model: pathlib.Path):
        return vision.load_learner(path_to_model.parent, path_to_model.name)


class Translator:
    def translate(
        self, text: str, output_language: str, input_language: str = "english"
    ) -> str:
        return text


def attach_translation_to_obj(
    obj: DetectedObject, translator: Translator, output_lang: str
):
    obj.translations.update({output_lang: translator.translate(obj.clas, output_lang)})
    return obj
