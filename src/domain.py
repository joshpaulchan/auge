import dataclasses
import functools

from typing import Sequence, Iterable, Mapping


@dataclasses.dataclass
class DetectedObject:
    clas: str = dataclasses.field(default=None)
    confidence: float = dataclasses.field(default=0.0)
    translations: Mapping = dataclasses.field(default_factory=dict)


class ObjectDetector:
    def detect(self, image: bytes) -> Iterable[DetectedObject]:
        if not image:
            return []
        return []


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
