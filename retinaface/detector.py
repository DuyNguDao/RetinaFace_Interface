from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from .data import cfg_re50, cfg_mnet
from .layers.functions.prior_box import PriorBox
from .loader import load_model
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


class FaceDetection:
    def __init__(self, net='mobilenet'):
        cudnn.benchmark = True
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            self.model = load_model(self.net).to(self.device)

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.model.eval()
            print("Number parameters: ", count_parameters(self.model))

    def detect_faces(self, img_raw, confidence_threshold=0.45, top_k=5000, nms_threshold=0.4, keep_top_k=750, resize=1):
        img = np.float32(img_raw)
        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        # tic = time.time()
        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass
            # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        landms = landms.reshape((-1, 5, 2))
        # landms = landms.transpose((0, 2, 1))
        # landms = landms.reshape(-1, 10, )
        bbox = dets[:, :4].astype('int32')
        score = dets[:, 4]
        landmark = landms.astype('int32')

        return bbox, score, landmark

