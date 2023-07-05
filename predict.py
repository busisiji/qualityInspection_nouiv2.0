"""
****************** 质量检测识别案例 ********************
# @Time      : 2019/12/31
# @Author    : Dobot
# @File name : predict.py
# Demo说明   :
*******************************************************
"""
# -*- coding: utf-8 -*-
import datetime
import os
import time
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from utils.utils import *
from PIL import Image

class Model():

    # 加载模型
    def modelinit(self,modelname='./quality_testing.pkl'):
        self.classes = read_lables()
        self.model_eval = models.resnet18(pretrained=False)
        num_ftrs = self.model_eval.fc.in_features
        self.model_eval.fc = nn.Linear(num_ftrs, len(self.classes))
        self.device = torch.device('cuda')
        # device = torch.device('cpu')
        self.model_eval.load_state_dict(torch.load(modelname, map_location=self.device))
        self.model_eval.to(self.device)
        self.model_eval.eval()

    # 模型预测
    def modelpred(self,crop_img):
        img = Image.fromarray(crop_img)
        # img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        img.save('model.bmp')
        tsfrm = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        img = tsfrm(img)

        img = img.unsqueeze(0)
        img = img.to(self.device)
        output = self.model_eval(img)
        # prob是2个分类的概率
        prob = F.softmax(output, dim=1)
        # print(prob)
        value, predicted = torch.max(output.data, 1)
        label = predicted.cpu().numpy()[0]
        pred_class = self.classes[predicted.item()]
        # 打印分类名称
        # print(pred_class)
        return pred_class

# 运行主函数

if __name__ == "__main__":
    # camera()
    model = Model()
    model.modelinit()
    image_source = cv2.imread('./image/white (1).bmp')
    model.modelpred(image_source)
    label = model.modelpred(image_source)
    print(label)



