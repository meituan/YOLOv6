from pathlib import Path
import wandb
from pathlib import Path


class WandbInferenceLogger:
    def __init__(self) -> None:
        self._label_dictionary = {}
        self.table = wandb.Table(
            columns=[
                "Image-File",
                "Predictions",
                "Number-of-Objects",
                "Prediction-Confidence",
            ]
        )

    @property
    def label_dictionary(self):
        return self._label_dictionary

    @label_dictionary.setter
    def label_dictionary(self, new_dict):
        self._label_dictionary = new_dict

    def in_infer(self, image, image_file, detection_results):
        bbox_data, confidences = [], []
        height, width, _ = image.shape
        for idx, (*xyxy, confidence, class_id) in enumerate(detection_results):
            confidences.append(float(confidence))
            xyxy = [int(coord) for coord in xyxy]
            bbox_data.append(
                {
                    "position": {
                        "minX": xyxy[0] / width,
                        "maxX": xyxy[2] / width,
                        "minY": xyxy[1] / height,
                        "maxY": xyxy[3] / height,
                    },
                    "class_id": int(class_id),
                    "box_caption": f"Key {idx}: {self.label_dictionary[int(class_id)]} {float(confidence)}",
                    "scores": {"confidence": float(confidence)},
                }
            )
        self.table.add_data(
            Path(image_file).stem,
            wandb.Image(
                image_file,
                boxes={
                    "predictions": {
                        "box_data": bbox_data,
                        "class_labels": self.label_dictionary,
                    }
                },
            ),
            len(detection_results),
            confidences,
        )

    def on_infer_end(self):
        wandb.log({"Inference": self.table})
