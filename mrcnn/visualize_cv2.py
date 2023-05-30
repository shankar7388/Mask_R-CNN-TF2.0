import cv2
import numpy as np
import os
import sys
# import coco
# import utils
# import model as modellib
from mrcnn import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs23/marble_cfg_coco20230225T1820")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_marble_cfg_coco_0015.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "marble_cfg_coco"
    # number of classes (background + Blue Marbles + Non Blue marbles)
    NUM_CLASSES = 1 + 5
    # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    BATCH_SIZE = 1
    IMAGES_PER_GPU = 1
# class InferenceConfig(Config):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

config = PredictionConfig()
config.display()
# config = InferenceConfig()
# config.display()

# model = modellib.MaskRCNN(
#     mode="inference", model_dir=MODEL_DIR, config=config
# )
model = MaskRCNN(mode='inference', model_dir='logs', config=config)

model.load_weights("logs23/marble_cfg_coco20230225T1820/mask_rcnn_marble_cfg_coco_0015.h5", by_name=True)

class_names = ['BG', 'Missing Rail Clip', 'Missing Bolt','Crack Detected','Defective Rail','Weed/Grass']


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


if __name__ == '__main__':
    """
        test everything
    """

    capture = cv2.VideoCapture(0)

    # these 2 lines can be removed if you dont have a 1080p camera.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
