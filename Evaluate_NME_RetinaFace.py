"""
MODULE EVALUATE LANDMARK WITH NME (Normalize Mean Error)
Developer: Duy Ngu Dao
Organization: Da Nang University of science and technology
Time: 20/09/2022
"""

import os
import numpy as np
import cv2
from pathlib import Path
import torch
import argparse
from glob import glob
from retinaface.detector import FaceDetection
from tqdm import tqdm


# --------------------- CONFIG MODEL -----------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def load_gt(path_txt):
    try:
        with open(path_txt, 'r') as file:
            data = file.readlines()
        data = [list(map(int, i.strip().split(' '))) for i in data]
        return data
    except:
        return []


def NME(lm_gt, lm_pred, d):
    error = lm_gt - lm_pred
    error = np.sqrt(np.sum(error**2, axis=1))/d
    return error


def evaluate_lm(model, path_data):
    print('Start Evaluate Landmark with NME loss ...')
    retina_model = model
    list_image = glob(path_data + '/*.jpg')
    total_error = []
    pbar_eval = tqdm(list_image, desc=f'Evaluate NME landmark: ', unit='image')
    for path_img in pbar_eval:
        image = cv2.imread(path_img)
        lm_gt = np.array(load_gt(path_img.split('.')[0] + '.txt'), dtype='float32')
        bbox, score, landmark = retina_model(image, confidence_threshold=0.5)
        if len(bbox) == 0:
            continue
        lm_pred = landmark.astype('float32')
        t_error = []
        for idx, box in enumerate(bbox):
            h_bbox, w_bbox = box[3] - box[1], box[2] - box[0]
            d = np.sqrt(w_bbox*h_bbox)
            error = NME(lm_gt, lm_pred[idx], d)
            t_error.append(error)
        id_min = np.argmin(np.sum(t_error, axis=1))
        total_error.append(t_error[id_min])

    error_NME = np.sum(total_error, axis=0)/len(total_error)
    nme = sum(error_NME)*100/5
    with open('image_info/loss_NME_RetinaFace.txt', 'w') as file:
        file.write('{} {} {} {} {} {}'.format(error_NME[0], error_NME[1], error_NME[2], error_NME[3], error_NME[4], nme))
    print('Normalize Mean Error (NME) is: {}% '.format(round(nme, 2)))
    print('finish!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Landmark')
    parser.add_argument("-md", "--name_model", help="name of model: mobilenet or resnet", default='mobilenet', type=str)
    parser.add_argument("-da", "--data", help="path of data contain image", type=str)
    args = parser.parse_args()

    # retina_model = FaceDetection(net=args.name_model).detect_faces
    # evaluate_lm(retina_model, args.data)

    # ***************** TEST *******************
    retina_model = FaceDetection(net='mobilenet').detect_faces
    # path image
    path_folder = '/home/duyngu/Downloads/Dataset_LFPW'
    evaluate_lm(retina_model, path_folder)
