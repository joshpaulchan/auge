import dataclasses
import functools
import pathlib

from fastai import vision

from typing import Sequence, Iterable, Mapping
import operator


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

    def detect(self, image: bytes, min_conf: float = 0.85) -> Iterable[DetectedObject]:
        if not image:
            return []

        _, _, losses = self.model.predict(vision.open_image(image))

        meets_min_confidence = functools.partial(operator.le, min_conf)
        predictions = sorted(
            filter(
                lambda p: meets_min_confidence(p[1]),
                zip(self.model.data.classes, map(float, losses)),
            ),
            key=lambda p: p[1],
            reverse=True,
        )

        return [
            DetectedObject.from_prediction(cls, confidence)
            for cls, confidence in predictions
        ]

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
