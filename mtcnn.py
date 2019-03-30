#coding:utf-8
import sys
sys.path.append('.')
sys.path.append('/home/wangchao/caffe/python')

import caffe
import cv2
import os
import numpy as np


def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T
    return rectangles


def NMS(rectangles,threshold,type):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)

    x1 = boxes[:,0] # 左上x
    y1 = boxes[:,1] # 左上y
    x2 = boxes[:,2] # 右下x
    y2 = boxes[:,3] # 右下y
    s  = boxes[:,4] # 置信度score
    area = np.multiply(x2-x1+1, y2-y1+1) # boxes 的面积
    I = np.array(s.argsort()) # Returns the indices that would sort an array. 即返回置信度从小到大的索引

    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) # I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        #
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'iom':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1]) # 先找到一个概率最高的框，肯定是自己需要的，然后再找与这个！！框交并比比较小的框再筛选，因为重叠了没用啊
        I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

def calculateScales(img):
        caffe_img = img.copy()
        # pr_scale = 1.0
        pr_scale = 0.7
        h, w, ch = caffe_img.shape
        # multi-scale
        scales = []
        factor = 0.709  # 0.709
        factor_count = 0
        minl = min(h, w) * pr_scale
        while 1:
            if minl >= 12:
                scales.append(pr_scale * pow(factor, factor_count))
                factor_count += 1
                minl *= factor
            else:
                break
        return scales

class mtcnn:
    def __init__(self, crop_face_size=160, threshold=[0.6, 0.6, 0.7]):
        self.crop_face_size = crop_face_size
        self.threshold = threshold
        self.net_12 = caffe.Net('model/det1.prototxt', 'model/det1.caffemodel', caffe.TEST)

        self.net_24 = caffe.Net('model/det2.prototxt', 'model/det2.caffemodel', caffe.TEST)

        self.net_48 = caffe.Net('model/det3.prototxt', 'model/det3.caffemodel', caffe.TEST)

    def detect_face_12net(self, cls_prob, roi, out_side, scale, width, height, threshold):
        in_side = 2 * out_side + 11
        stride = 2.0
        (x, y) = np.where(cls_prob >= threshold)
        boundingbox = np.array([x, y]).T
        bb1 = np.fix((stride * (boundingbox) + 0) * scale)
        bb2 = np.fix((stride * (boundingbox) + 11) * scale)

        boundingbox = np.concatenate((bb1, bb2), axis=1)

        score = np.array([cls_prob[x, y]]).T
        rectangles = np.concatenate((boundingbox, score), axis=1)

        dx1 = roi[0][x, y]
        dx2 = roi[1][x, y]
        dx3 = roi[2][x, y]
        dx4 = roi[3][x, y]
        offset = np.array([dx1, dx2, dx3, dx4]).T
        rectangles = np.concatenate((rectangles, offset), axis=1)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0, rectangles[i][0]))
            y1 = int(max(0, rectangles[i][1]))
            x2 = int(min(width, rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            sc = rectangles[i][4]
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, sc])
            pick.append([x1, y1, x2, y2, sc])
        return pick

    def filter_face_24net(self, cls_prob, roi, rectangles, width, height, threshold):
        prob = cls_prob[:, 1]
        pick = np.where(prob >= threshold)
        rectangles = np.array(rectangles)
        x1 = rectangles[pick, 0]
        y1 = rectangles[pick, 1]
        x2 = rectangles[pick, 2]
        y2 = rectangles[pick, 3]
        sc = np.array([prob[pick]]).T
        dx1 = roi[pick, 0]
        dx2 = roi[pick, 1]
        dx3 = roi[pick, 2]
        dx4 = roi[pick, 3]
        w = x2 - x1
        h = y2 - y1
        x1 = np.array([(x1 + dx1 * w)[0]]).T
        y1 = np.array([(y1 + dx2 * h)[0]]).T
        x2 = np.array([(x2 + dx3 * w)[0]]).T
        y2 = np.array([(y2 + dx4 * h)[0]]).T
        rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
        rectangles = rect2square(rectangles)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0, rectangles[i][0]))
            y1 = int(max(0, rectangles[i][1]))
            x2 = int(min(width, rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            sc = rectangles[i][4]
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, sc])
        return NMS(pick, 0.7, 'iou')

    def filter_face_48net(self, cls_prob, roi, pts, rectangles, width, height, threshold):
        prob = cls_prob[:, 1]
        pick = np.where(prob >= threshold)
        rectangles = np.array(rectangles)
        x1 = rectangles[pick, 0]
        y1 = rectangles[pick, 1]
        x2 = rectangles[pick, 2]
        y2 = rectangles[pick, 3]
        sc = np.array([prob[pick]]).T
        dx1 = roi[pick, 0]
        dx2 = roi[pick, 1]
        dx3 = roi[pick, 2]
        dx4 = roi[pick, 3]
        w = x2 - x1
        h = y2 - y1
        pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
        pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
        pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
        pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
        pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
        pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
        pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
        pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
        pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
        pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T
        x1 = np.array([(x1 + dx1 * w)[0]]).T
        y1 = np.array([(y1 + dx2 * h)[0]]).T
        x2 = np.array([(x2 + dx3 * w)[0]]).T
        y2 = np.array([(y2 + dx4 * h)[0]]).T
        rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                    axis=1)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0, rectangles[i][0]))
            y1 = int(max(0, rectangles[i][1]))
            x2 = int(min(width, rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, rectangles[i][4],
                             rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                             rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13],
                             rectangles[i][14]])
        return NMS(pick, 0.7, 'iom')

    def detectFace(self, img):
        threshold = self.threshold
        if img is None:
            return [], []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        caffe_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, ch = caffe_img.shape
        scales = calculateScales(img)
        out = []
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(caffe_img, (ws, hs))
            scale_img = np.swapaxes(scale_img, 0, 2)
            self.net_12.blobs['data'].reshape(1, 3, ws, hs)
            self.net_12.blobs['data'].data[...] = scale_img
            #caffe.set_device(0)
            #caffe.set_mode_gpu()
            #caffe.set_mode_cpu()


            out_ = self.net_12.forward()
            out.append(out_)
        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            cls_prob = out[i]['prob1'][0][1]  # NCHW,第二个通道是人脸概率
            roi = out[i]['conv4-2'][0]
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            rectangle = self.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)
        rectangles = NMS(rectangles, 0.7, 'iou')
        draw = img.copy()
        for rectangle in rectangles:
            cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 1)

        if len(rectangles) == 0:
            return [], rectangles
        self.net_24.blobs['data'].reshape(len(rectangles), 3, 24, 24)
        crop_number = 0
        for rectangle in rectangles:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            scale_img = np.swapaxes(scale_img, 0, 2)
            self.net_24.blobs['data'].data[crop_number] = scale_img
            crop_number += 1
        out = self.net_24.forward()
        cls_prob = out['prob1']
        roi_prob = out['conv5-2']
        rectangles = self.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

        draw_ = img.copy()
        for rectangle in rectangles:
            cv2.rectangle(draw_, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 1)

        if len(rectangles) == 0:
           return [], rectangles
        self.net_48.blobs['data'].reshape(len(rectangles), 3, 48, 48)
        crop_number = 0
        for rectangle in rectangles:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            scale_img = np.swapaxes(scale_img, 0, 2)
            self.net_48.blobs['data'].data[crop_number] = scale_img
            crop_number += 1
        out = self.net_48.forward()
        cls_prob = out['prob1']
        roi_prob = out['conv6-2']
        pts_prob = out['conv6-3']
        rectangles = self.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        imgs = []
        for rectangle in rectangles:
            img_ = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            imm = cv2.resize(img_, (self.crop_face_size, self.crop_face_size), interpolation=cv2.INTER_CUBIC)
            imgs.append(imm)
        return imgs, rectangles




