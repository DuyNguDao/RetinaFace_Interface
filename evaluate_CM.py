"""
MODULE EVALUATE OBJECT DETECTION INCLUDE: CONFUSION MATRIX, ACCURACY, PRECISION, RECALL, F1-SCORE
Developer: Duy Ngu Dao
Organization: Da Nang University of science and technology
Time: 14/09/2022
"""

import os
import numpy as np
import cv2
from pathlib import Path
import torch
import argparse
from sklearn import metrics
from utils.plot import plot_cm
from retinaface.detector import FaceDetection
from tqdm import tqdm


# --------------------- CONFIG MODEL -----------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# ----------------STRUCTURE DATA WIDER-FACE -----------------------
'''
line 1: 'path image'
line 2: 'faces number'
line 3-vv...: 'bbox'
'''


def get_gt_boxes_from_txt(gt_path):
    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '#' in line:
            # get path of image
            state = 2
            current_name = line
            continue
        # if state == 1:
        #     # get faces number
        #     state = 2
        #     continue

        if state == 2 and '#' in line:
            # check bbox and next path of name
            # state = 1
            boxes[current_name.split(' ')[-1]] = np.array(current_boxes).astype('float32')
            # assign name of next image
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            # get bbox and processing x, y, w, h
            box = [float(x) for x in line.strip().split(' ')]
            current_boxes.append(box)
            continue
    print('Total face: ', len(lines)-len(boxes))
    return boxes


def IOU(gt, pred):
    """
    Develop by: Duy Ngu Dao - Da Nang university of science and technology
    Function compute IOU with array
    Args:
        gt: [[x1, y1, x2, y2], [...], ...]
        pred: [x1, y2, x2, y2]

    Returns: max iou, idx of gt
    Examples
    --------
    gt = np.array([[50, 50, 100, 100], [100, 100, 200, 200]], dtype='float')
    pred = np.array([75, 75, 150, 150], dtype='float')
    IOU(gt, pred)
    0.19047619047619047 1
    """
    point_xy1 = np.maximum(gt[:, :2], pred[:2])
    point_xy2 = np.minimum(gt[:, 2:4], pred[2:4])
    wh = point_xy2 - point_xy1
    result = np.where(wh < 0, 0.0, wh)
    area_overlap = result[:, 0] * result[:, 1]
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
    area_combined = area_pred + area_gt - area_overlap
    iou = np.amax(area_overlap / area_combined)
    idx = np.argmax(area_overlap / area_combined)
    return iou, idx


def compute_iou(gt, pred, gt_label, pred_label, num_class, iou_thresh=0.45):
    gt_iou = np.ones(len(gt)).astype('int32')
    truth_label = []
    predict_label = []
    for id, bbox in enumerate(pred):
        iou, idx = IOU(gt, bbox)
        if iou >= iou_thresh:
            truth_label.append(gt_label[idx])
            predict_label.append(pred_label[id])
            gt_iou[idx] = 0
        else:
            predict_label.append(pred_label[id])
            truth_label.append(num_class)
    idx_true = (gt_iou == 1)
    gt_label = gt_label[idx_true].tolist()
    pred_iou = [num_class]*len(gt_label)
    truth_label.extend(gt_label)
    predict_label.extend(pred_iou)
    return truth_label, predict_label


def evaluate_obj(model, path_data, path_label, name_class: tuple):
    print('Start Evaluate Object detection ...')
    num_class = len(name_class)
    data = get_gt_boxes_from_txt(path_label)
    retina_model = model
    count_face = 0
    CM = np.zeros((num_class + 1, num_class + 1)).astype('int32')
    class_name = [i for i in range(num_class + 1)]
    pred = []
    truth = []
    import time
    start = time.time()
    pbar_eval = tqdm(data.keys(), desc=f'Evaluate: ', unit='image')
    for name in pbar_eval:
        gt = data[name]
        gt_label = np.zeros(len(gt)).astype('int32')
        count_face += len(gt)
        # Format of dataset WIDER-FACE: x, y, w, h ------ FDDB: xmin, ymin, xmax, ymax
        # gt[:, 2] = gt[:, 2] + gt[:, 0]  # xmax = xmin + w
        # gt[:, 3] = gt[:, 3] + gt[:, 1]  # ymax = ymin + h
        path_image = os.path.join(path_data, name)
        image = cv2.imread(path_image)
        bbox, score, landmark = retina_model(image, confidence_threshold=0.35, nms_threshold=0.45)
        label_id = [0]*len(bbox)
        bbox, pred_label = np.array(bbox), np.array(label_id)
        truth_label, predict_label = compute_iou(gt, bbox, gt_label, pred_label, num_class=num_class)
        truth.extend(truth_label)
        pred.extend(predict_label)
        _CM = metrics.confusion_matrix(truth_label, predict_label, labels=class_name).T
        CM += _CM
    fps = (time.time()-start)/len(data.keys())
    precision = metrics.precision_score(truth, pred, average=None)
    recall = metrics.recall_score(truth, pred, average=None)
    accuracy = metrics.accuracy_score(truth, pred, normalize=True)
    f1_score = metrics.f1_score(truth, pred, average=None)
    print('Accuracy: ', accuracy)
    for i in range(num_class):
        print('****Precision-Recall-F1-Score of class {}****'.format(name_class[i]))
        print('Precision: ', precision[i])
        print('Recall: ', recall[i])
        print('F1-score', f1_score[i])
    with open('image_info/info.txt', 'w') as file:
        file.write('{} {} {} {}'.format(round(precision[0], 2), round(recall[0], 2), round(f1_score[0], 2), fps))
    plot_cm(CM, normalize=False, save_dir='image_info', names_x=name_class + ('Background FP',),
            names_y=name_class + ('Background FN',), show=False)
    print('Finishing!.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face On Video')
    parser.add_argument("-md", "--name_model", help="name of model: mobilenet or resnet", default='mobilenet', type=str)
    parser.add_argument("-da", "--data", help="path of data contain image", type=str)
    parser.add_argument("-la", "--label", help="path of file label", type=str)
    parser.add_argument("-na", "--name_class", default=('1',), help="Name of class", type=tuple)
    args = parser.parse_args()

    # retina_model = FaceDetection(net=args.name_model).detect_faces
    # evaluate_obj(retina_model, args.data, args.label, args.name_class)

    # ***************** TEST *******************
    retina_model = FaceDetection(net='mobilenet').detect_faces

    # FDDB
    # path label
    path = '/home/duyngu/Downloads/Dataset_Face_Detection/Dataset_FDDB/label.txt'
    # path image
    path_folder = '/home/duyngu/Downloads/Dataset_Face_Detection/Dataset_FDDB/images'
    # # WIDER FACE
    # # path label
    # path = '/home/duyngu/Downloads/Dataset_Face_Detection/widerface/val/label.txt'
    # # path image
    # path_folder = '/home/duyngu/Downloads/Dataset_Face_Detection/widerface/val/images'
    evaluate_obj(retina_model, path_folder, path, name_class=('Face',))
