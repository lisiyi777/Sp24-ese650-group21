# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from osgeo import gdal
from sklearn.metrics import confusion_matrix

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)        
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())    
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou) 
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, fwavacc

def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset 
    return im_proj,im_geotrans,im_width, im_height,im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape 

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset


def eval_re(label_path, predict_path, eval_path):
    pres = os.listdir(predict_path)
    labels = []
    predicts = []
    for im in pres:
        if im[-4:] == '.tif':
            label_name = im.split('.')[0] + '.tif'
            lab_path = os.path.join(label_path, label_name)
            pre_path = os.path.join(predict_path, im)
            im_proj,im_geotrans,im_width, im_height, label = read_img(lab_path)
            im_proj,im_geotrans,im_width, im_height, pre = read_img(pre_path)
            # label = cv2.imread(lab_path,0)
            # pre = cv2.imread(pre_path,0)
            label[label>0] = 1
            pre[pre>0] = 1
            label = np.uint8(label)
            pre = np.uint8(pre)
            labels.append(label)
            predicts.append(pre)
    el = IOUMetric(2)
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)

    pres = os.listdir(predict_path)
    init = np.zeros((2,2))
    for im in pres:
        lb_path = os.path.join(label_path, im)
        pre_path = os.path.join(predict_path, im)
        # lb = cv2.imread(lb_path,0)
        # pre = cv2.imread(pre_path,0)
        im_proj,im_geotrans,im_width, im_height, lb = read_img(lb_path)
        im_proj,im_geotrans,im_width, im_height, pre = read_img(pre_path)
        lb[lb>0] = 1
        pre[pre>0] = 1
        lb = np.uint8(lb)
        pre = np.uint8(pre)
        lb = lb.flatten()
        pre = pre.flatten()
        confuse = confusion_matrix(lb, pre)
        init += confuse

    precision = init[1][1]/(init[0][1] + init[1][1]) 
    recall = init[1][1]/(init[1][0] + init[1][1])
    accuracy = (init[0][0] + init[1][1])/init.sum()
    f1_score = 2*precision*recall/(precision + recall)

    with open(eval_path, 'a') as f:
        f.write('accuracy: ' + str(accuracy) + '\n')
        f.write('recal: ' + str(recall) + '\n')
        f.write('miou: ' + str(miou))


def eval_func(label_path, predict_path):
    pres = os.listdir(predict_path)
    labels = []
    predicts = []
    for im in pres:
        if im[-4:] == '.png':
            label_name = im.split('.')[0] + '.png'
            lab_path = os.path.join(label_path, label_name)
            pre_path = os.path.join(predict_path, im)
            label = cv2.imread(lab_path,0)
            pre = cv2.imread(pre_path,0)
            label[label>0] = 1
            pre[pre>0] = 1
            label = np.uint8(label)
            pre = np.uint8(pre)
            labels.append(label)
            predicts.append(pre)
    el = IOUMetric(2)
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts,labels)
    print('acc: ',acc)
    print('acc_cls: ',acc_cls)
    print('iou: ',iou)
    print('miou: ',miou)
    print('fwavacc: ',fwavacc)

    pres = os.listdir(predict_path)
    init = np.zeros((2,2))
    for im in pres:
        lb_path = os.path.join(label_path, im)
        pre_path = os.path.join(predict_path, im)
        lb = cv2.imread(lb_path,0)
        pre = cv2.imread(pre_path,0)
        lb[lb>0] = 1
        pre[pre>0] = 1
        lb = np.uint8(lb)
        pre = np.uint8(pre)
        lb = lb.flatten()
        pre = pre.flatten()
        confuse = confusion_matrix(lb, pre)
        init += confuse

    precision = init[1][1]/(init[0][1] + init[1][1]) 
    recall = init[1][1]/(init[1][0] + init[1][1])
    accuracy = (init[0][0] + init[1][1])/init.sum()
    f1_score = 2*precision*recall/(precision + recall)
    print('class_accuracy: ', precision)
    print('class_recall: ', recall)
    print('accuracy: ', accuracy)
    print('f1_score: ', f1_score)


def eval_new(label_list, pre_list):
    el = IOUMetric(2)
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(pre_list, label_list)
    print('acc: ',acc)
    # print('acc_cls: ',acc_cls)
    print('iou: ',iou)
    print('miou: ',miou)
    print('fwavacc: ',fwavacc)

    init = np.zeros((2,2))
    for i in range(len(label_list)):
        lab = label_list[i].flatten()
        pre = pre_list[i].flatten()
        confuse = confusion_matrix(lab, pre)
        init += confuse

    precision = init[1][1]/(init[0][1] + init[1][1]) 
    recall = init[1][1]/(init[1][0] + init[1][1])
    accuracy = (init[0][0] + init[1][1])/init.sum()
    f1_score = 2*precision*recall/(precision + recall)
    print('class_accuracy: ', precision)
    print('class_recall: ', recall)
    # print('accuracy: ', accuracy)
    print('f1_score: ', f1_score)


if __name__ == "__main__":
    label_path = './data/build/test/labels/'
    predict_path = './data/build/test/re/'

    eval_func(label_path, predict_path)

