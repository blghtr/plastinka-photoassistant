from collections import OrderedDict
from os import PathLike
from typing import Union, Dict
import ultralytics
from numpy import ndarray
import numpy as np
from torch import as_tensor
from .base_module import BaseModule


class Segmenter(BaseModule):
    """YOLOv8-based image segmentation.

    Produces original image, mask, bbox, predicted class, and optional intermediate visualizations.
    """
    def __init__(
            self,
            model: Union[str, PathLike],
            conf_threshold: float = 0.6,
            device: str = 'cpu',
            save_intermediate_outputs: bool = True,
            **kwargs
    ):
        self.model = ultralytics.YOLO(model)

        super().__init__(
            model=model,
            conf_threshold=conf_threshold,
            device=device,
            save_intermediate_outputs=save_intermediate_outputs,
            **kwargs
        )

    # LLM:METADATA
    # :hierarchy: [PhotoAssist | Modules | Segmenter]
    # :relates-to: calls: "segment"
    # :rationale: "Delegate to specialized segmentation logic, bypassing standard BaseModule flow."
    # :contract: pre: "input_data valid", post: "returns segmentation result"
    # LLM:END
    def __call__(self, input_data: Dict):
        """Return segmentation result (overrides base behavior)."""
        return self.segment(input_data)

    # LLM:METADATA
    # :hierarchy: [PhotoAssist | Modules | Segmenter]
    # :relates-to: uses: "ultralytics.YOLO.predict", calls: "cut_mask"
    # :rationale: "Execute object detection and segmentation to isolate the subject."
    # :contract: pre: "image in input_data", post: "returns dict with mask, box, class"
    # LLM:END
    def segment(self, input_data: Dict) -> Dict:
        """Run model inference and construct the result dict."""
        image = input_data['image']
        self.logger.info("Starting model inference...")
        results = self.model.predict(source=image, max_det=1, device=self.args['device'], retina_masks=True)
        self.logger.info("Model inference finished.")

        # Gracefully handle cases where no objects are detected
        if not results or results[0].masks is None or results[0].boxes is None or len(results[0].boxes) == 0:
            image_name = input_data.get('name', 'Unknown')
            self.logger.warning(f"Segmentation failed: No objects detected in image '{image_name}'.")
            raise ValueError(f"No objects detected in '{image_name}'")

        result = results[0]
        save_intermediate_output = self.args['save_intermediate_outputs']

        result = cut_mask(result)
        pred_class, pred_conf = result.names[(int(result.boxes.cls.item()))], result.boxes.conf.item()
        self.logger.info(f"Segmentation result: class='{pred_class}', confidence={pred_conf:.4f}")

        return {
            'image': result.orig_img,
            'segments': result.masks.cpu().xy[0],
            'mask': result.masks.data.numpy().squeeze().astype(np.uint8),
            'box': result.boxes.cpu().xyxy.numpy()[0],
            'class': (pred_class, pred_conf),
            'name': input_data['name'],
            'intermediate_outputs': OrderedDict(
                [(self.__class__.__name__, self._apply_transform(result))]
            ) if save_intermediate_output else None
        }

    # LLM:METADATA
    # :hierarchy: [PhotoAssist | Modules | Segmenter]
    # :relates-to: uses: "ultralytics.engine.results.Results.plot"
    # :rationale: "Generate visual debugging output highlighting detections."
    # :contract: pre: "result object valid", post: "returns annotated image"
    # LLM:END
    def _apply_transform(self, result) -> ndarray:
        """Return visualization: for 'apple' draw bbox; otherwise draw masks."""
        if result.names[(int(result.boxes.cls.item()))] == 'apple':
            return result.plot(masks=False)
        return result.plot(boxes=False, line_width=5)


# LLM:METADATA
# :hierarchy: [PhotoAssist | Modules | Segmenter | Utils]
# :rationale: "Refine segmentation mask by clipping it to the bounding box."
# :contract: pre: "result has masks and boxes", post: "returns updated result object"
# LLM:END
def cut_mask(result):
    """Crop mask by predicted bbox to keep the region of interest only."""
    mask_obj = result.masks.cpu()
    mask = mask_obj.data.numpy().transpose((1, 2, 0))
    new_mask = np.zeros_like(mask)
    h, w, _ = new_mask.shape
    bbx = result.boxes.xyxyn.numpy()[0]
    x_min, y_min, x_max, y_max = (bbx * np.array([w, h, w, h])).astype(np.int32)
    new_mask[y_min:y_max, x_min:x_max] = mask[y_min:y_max, x_min:x_max]
    result.update(masks=as_tensor(new_mask.transpose((2, 0, 1))))

    del mask, new_mask
    return result