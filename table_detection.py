import itertools
import json
import os
import random

import cv2

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import detectron2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


def cv2_imshow(img):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis("off")  # Turn off axis numbers and ticks
    plt.show()


def initialize_detector_model(model_path, config_path, scr_thresh=0.15, nms_thresh=0.5):

    # create detectron config
    cfg = get_cfg()

    # set yaml
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(scr_thresh)  # 0.05 range(0,1)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(nms_thresh)  # 0.5
    # set model weights
    cfg.MODEL.WEIGHTS = str(model_path)  # Set path model .pth

    predictor = DefaultPredictor(cfg)
    return predictor


# Function to update predictor thresholds
def update_predictor_thresholds(predictor, scr_thresh, nms_thresh):
    cfg = predictor.cfg.clone()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(scr_thresh)
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(nms_thresh)
    return DefaultPredictor(cfg)


def plot_prediction(img, predictor):

    outputs = predictor(img)

    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    for x1, y1, x2, y2 in (
        outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()
    ):
        # Start coordinate
        # represents the top left corner of rectangle
        start_point = int(x1), int(y1)

        # Ending coordinate
        # represents the bottom right corner of rectangle
        end_point = int(x2), int(y2)

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img = cv2.rectangle(
            np.array(img, copy=True), start_point, end_point, color, thickness
        )

    # Displaying the image
    # print("TABLE DETECTION:")
    # cv2_imshow(img)
    return img


def make_prediction(img, predictor):

    # img = cv2.imread(img_path)
    outputs = predictor(img)

    table_list = []
    table_coords = []
    imgs = []

    for i, box in enumerate(
        outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()
    ):
        x1, y1, x2, y2 = box
        table_list.append(
            np.array(img[int(y1) : int(y2), int(x1) : int(x2)], copy=True)
        )
        table_coords.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        imgs.append(img[int(y1) : int(y2), int(x1) : int(x2)])
        # print("TABLE", i, ":")
        cv2_imshow(img[int(y1) : int(y2), int(x1) : int(x2)])

    return table_list, table_coords, imgs


def recognize_structure(img, show_contours=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    # Thresholding to a binary image
    img_bin = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    img_bin = 255 - img_bin  # Inverting the image

    # Vertical and horizontal line detection
    kernel_len_ver = img_height // 50
    kernel_len_hor = img_width // 50
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    vertical_lines = cv2.dilate(
        cv2.erode(img_bin, ver_kernel, iterations=3), ver_kernel, iterations=4
    )
    horizontal_lines = cv2.dilate(
        cv2.erode(img_bin, hor_kernel, iterations=3), hor_kernel, iterations=4
    )

    # Combine the detected lines
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)

    # Detect contours
    contours, _ = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("contours", contours)
    # print(len(contours))

    # Draw contours on the original image
    if show_contours:
        img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Contours image", img_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if len(contours) < 60:
        # If there are fewer than 90 contours, it is unlikely to be a table
        return None, None

    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        cnts, boundingBoxes = zip(
            *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse)
        )
        return cnts, boundingBoxes

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)

    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 0.9 * img_width and h < 0.9 * img_height:
            box.append([x, y, w, h])

    if not box:
        return None, None

    row = []
    column = []
    for i in range(len(box)):
        if i == 0:
            column.append(box[i])
            previous = box[i]
        else:
            if box[i][1] <= previous[1] + mean / 2:
                column.append(box[i])
                previous = box[i]
                if i == len(box) - 1:
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    countcol = max(len(r) for r in row)
    center = np.array([int(r[0] + r[2] / 2) for r in row[0]])
    center.sort()

    finalboxes = []
    for r in row:
        lis = [[] for _ in range(countcol)]
        for box in r:
            diff = abs(center - (box[0] + box[2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(box)
        finalboxes.append(lis)

    return finalboxes, img_bin


def filter_images_with_tables(images):
    table_images = []
    for img in images:
        # img = cv2.imread(img)
        if img is None:
            print(f"Error loading image: {img}")
            continue
        # print(f"type of image{type(img)}{img.shape}")
        finalboxes, img_bin = recognize_structure(img)
        if finalboxes:
            table_images.append(img)

    # img = cv2.imread(img)
    # if img is None:
    #     print(f"Error loading image: {img}")

    # print(f"type of image{type(img)}{img.shape}")
    # finalboxes, img_bin = recognize_structure(img, True)
    # if finalboxes:
    #     table_images.append(img)
    return table_images
