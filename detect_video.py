import cv2
import time
import numpy as np
import math
from numpy import random
from pathlib import Path
import torch
import argparse
from retinaface.detector import FaceDetection
import cv2
import copy


def draw_result(image, boxes, scores, landmarks):
    color = (0, 255, 0)
    xmin, ymin, xmax, ymax = boxes
    h, w, c = image.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=tl, lineType=cv2.LINE_AA)
    tf = max(tl - 1, 1)  # font thickness
    label = str(scores)[:5]
    cv2.putText(image, label, (xmin, ymin - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i, kps in enumerate(landmarks):
        point_x = kps[0]
        point_y = kps[1]
        cv2.circle(image, (point_x, point_y), tl+1, clors[i], -1)
    return image


def detect_video(url_video=None, flag_save=False, fps=None, name_video='video.avi'):

    # ******************************** LOAD MODEL *************************************************
    retina_model = FaceDetection(net='mobilenet').detect_faces
    # ********************************** READ VIDEO **********************************************
    if url_video == '':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(url_video)

    # get size
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if frame_height > 1080 and frame_width > 1920:
        frame_width = 1920
        frame_height = 1080
    # get fps of camera
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    # save video
    if flag_save is True:
        video_writer = cv2.VideoWriter(name_video,
                                       cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # ******************************** REAL TIME *********************************************
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) == ord('q'):
            break
        h, w, _ = frame.shape
        if h > 1080 and w > 1920:
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            h, w, _ = frame.shape

        # ************************ DETECT YOLOv7 ***************************************)
        # h0, w0 = frame.shape[:2]  # orig hw
        # r = 640 / max(h0, w0)  # resize image to img_size
        # if r != 1:  # always resize down, only resize up if training with augmentation
        #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        #     frame = cv2.resize(frame, (int(w0 * r), int(h0 * r)), interpolation=interp)
        # ------------------------ FIX SIZE IMAGE -------------------------------------
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # -----------------------------------------------------------------------------
        bbox, score, landmark = retina_model(frame, confidence_threshold=0.7, resize=1)
        for idx, box in enumerate(bbox):
            draw_result(frame, box, score[idx], landmark[idx])

        # ******************************************** SHOW *******************************************
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        fps = int(1 / (time.time() - start))
        cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('video', frame)
        cv2.waitKey(1)

        if flag_save is True:
            video_writer.write(frame)

    cap.release()
    if flag_save is True:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Face On Video')
    parser.add_argument("-fn", "--file_name", help="video file name or rtsp", default='', type=str)
    parser.add_argument("-op", "--option", help="if save video then choice option = 1", default=False, type=bool)
    parser.add_argument("-o", "--output", help="path to output video file", default='face_recording.avi', type=str)
    parser.add_argument("-f", "--fps", default=20, help="FPS of output video", type=int)
    args = parser.parse_args()

    # PATH VIDEO    url = '/home/duyngu/Downloads/video_test/demo_demnguoi.flv'
    url = ''
    source = args.file_name
    cv2.namedWindow('video')
    # if run  as terminal, replace url = source
    detect_video(url_video=url, flag_save=args.option, fps=args.fps, name_video=args.output)
