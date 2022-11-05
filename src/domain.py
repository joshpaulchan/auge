import dataclasses
import functools
import pathlib

from fastai import vision
import torchvision

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

    def detect(self, image_path: str, min_conf: float = 0.85) -> Iterable[DetectedObject]:
        if not image_path:
            return []

        _, _, losses = self.model.predict(vision.open_image(image_path))

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

class PytorchObjectDetector:
    def __init__(self, path_to_model: pathlib.Path):
        self.model, self.weights = self._init_model(path_to_model)

    def detect(self, image_path: str, min_conf: float = 0.85) -> Iterable[DetectedObject]:
        if not image_path:
            return []

        from torchvision.io import read_image
        from torchvision.models import resnet50, ResNet50_Weights

        img = read_image(image_path)

        # Step 1: Initialize model with the best available weights
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()

        # Step 2: Initialize the inference transforms
        preprocess = weights.transforms()

        # Step 3: Apply inference preprocessing transforms
        batch = preprocess(img).unsqueeze(0)

        # Step 4: Use the model and print the predicted category
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        print(f"{category_name}: {100 * score:.1f}%")

        return [
            DetectedObject.from_prediction(category_name, score)
        ]
    
    def _init_model(self, path_to_model: pathlib.Path):
        # return vision.load_learner(path_to_model.parent, path_to_model.name)
        return [None, None]


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
