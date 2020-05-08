import numpy as np
from model.yolo import YoloV3
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from model.utils import draw_outputs, load_classes, load_anchors

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("classes")
    parser.add_argument("anchor")
    parser.add_argument("weight")

    args = parser.parse_args()

    classes_names = load_classes(args.classes)
    anchors, masks = load_anchors(args.anchor)
    anchors = anchors/416

    yolo = YoloV3(anchors, masks, len(classes_names))
    yolo.build((1, 416, 416, 3))
    img = np.zeros((1, 416, 416, 3))
    output = yolo(img)
    yolo.load_weights(args.weight)

    base_img = Image.open("street.jpg")
    img = np.array(base_img)
    img = cv2.resize(img, (416, 416))/255

    boxes, scores, classes, nums = yolo(img.reshape(1, 416, 416, 3))

    img = np.array(base_img)
    img = draw_outputs(img, (boxes, scores, classes, nums), classes_names)
    plt.imshow(img)
    plt.show()



    
