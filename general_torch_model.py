# coding:utf-8
import torch
import numpy as np
import torch.nn as nn
from torchvision.transforms import Compose, RandomResizedCrop, Pad, RandomCrop
import matplotlib.pyplot as plt
import torch

class GeneralTorchModel(nn.Module):
    def __init__(self, model, args, n_class=10, im_mean=None, im_std=None):
        super(GeneralTorchModel, self).__init__()
        self.model = model
        self.model.eval()
        self.num_queries = 0
        self.im_mean = im_mean
        self.im_std = im_std
        self.n_class = n_class
        self.args = args

    def forward(self, image):
        if len(image.size()) == 3:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self.model(image)
        return logits

    def preprocess(self, images):
        if isinstance(images, np.ndarray):
            processed = torch.from_numpy(images).type(torch.FloatTensor)
        else:
            processed = images

        if self.im_mean is not None and self.im_std is not None:
            im_mean = torch.tensor(self.im_mean).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            im_std = torch.tensor(self.im_std).cuda().view(1, processed.shape[1], 1, 1).repeat(
                processed.shape[0], 1, 1, 1)
            processed = (processed - im_mean) / im_std
        return processed

    def predict_prob(self, images):
        with torch.no_grad():
            if len(images.size()) == 3:
                images = images.unsqueeze(0)
            images = self.preprocess(images)
            logits = self.model(images)
            self.num_queries += images.size(0)

            if isinstance(logits, (dict, tuple)):
                logits = logits["logits"] if "logits" in logits else logits[0]
            elif hasattr(logits, 'logits'):
                logits = logits.logits
        return logits

    def predict_label(self, images):
        images = torch.clamp(images, 0.0, 1.0)
        logits = self.predict_prob(images)
        _, predict = torch.max(logits, 1)
        return predict