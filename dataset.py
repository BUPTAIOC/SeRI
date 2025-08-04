# coding:utf-8
import random
import numpy as np
import torch
from PIL import Image
import os
import csv
import statistics
import torchvision
import torchvision.datasets as dsets
from torchvision import models
import torchvision.transforms as transforms
from general_torch_model import GeneralTorchModel
from arch import mnist_model, Fashion_mnist_model, cifar10_model
from model import *
import timm
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModelForImageClassification, AutoFeatureExtractor
from transformers import AutoModel, AutoConfig
import core

class OutResult:
    def __init__(self, args, Folder_name):
        self.args = args
        self.ImgNum_origin_right = 0
        self.ImgNum_total_tested = 0
        self.ImgNum_ATK_success = 0
        self.AccQuery = []
        self.AccQuery_avg = 0
        self.AccQuery_mid = 0
        self.AccIter = []
        self.AccIter_avg = 0
        self.AccQdI = 0
        self.EndADB = []
        self.EndADB_avg = 0
        self.EndADB_mid = 0
        self.EndL2 = []
        self.Endl2_avg = 0
        self.Endl2_mid = 0
        self.Endlinf = []
        self.Endlinf_avg = 0
        self.L2_lines = []
        self.Linf_lines = []
        self.Kbridge = []
        self.Folder_name = Folder_name
        self.NATURAL_ACCURACY_RATE = 0
        self.ROBUST_ACCURACY_RATE = 0
        self.ATTACK_SUCCESS_RATE = 0

        self.L2_LINE_sum = None
        self.Linf_LINE_sum = None
        self.AUC_l2 = 0
        self.AUC_linf = 0
    def add1Result(self, ATKer):
        self.ImgNum_origin_right = self.ImgNum_origin_right + 1
        self.EndADB.append(ATKer.old_best_adv.ADBmax)
        self.EndADB_avg = statistics.mean(self.EndADB)
        self.EndADB_mid = statistics.median(self.EndADB)
        self.EndL2.append(ATKer.old_best_adv.RealL2)
        try:
            self.Kbridge.extend(ATKer.total_Kbridge)
        except AttributeError:
            self.Kbridge.extend([])
        self.Endl2_avg = statistics.mean(self.EndL2)
        self.Endl2_mid = statistics.median(self.EndL2)
        self.Endlinf.append(ATKer.old_best_adv.RealLinf)
        self.Endlinf_avg = statistics.mean(self.Endlinf)
        ATKer.L2_line.append([self.args.budget + 1, ATKer.old_best_adv.RealL2])
        self.L2_lines.append(ATKer.L2_line)
        ATKer.Linf_line.append([self.args.budget + 1, ATKer.old_best_adv.RealLinf])
        self.Linf_lines.append(ATKer.Linf_line)

        if ATKer.success == 1:
            self.ImgNum_ATK_success = self.ImgNum_ATK_success + 1
            self.AccQuery.append(ATKer.ACCquery)
            self.AccIter.append(ATKer.ACCiter_n)
        if self.ImgNum_ATK_success != 0:
            self.AccQuery_avg = statistics.mean(self.AccQuery)
            self.AccQuery_mid = statistics.median(self.AccQuery)
            self.AccIter_avg = statistics.mean(self.AccIter)
            self.AccQdI = self.AccQuery_avg / max(self.AccIter_avg, 1)

        self.NATURAL_ACCURACY_RATE = self.ImgNum_origin_right / self.ImgNum_total_tested
        self.ROBUST_ACCURACY_RATE = (self.ImgNum_origin_right - self.ImgNum_ATK_success) / self.ImgNum_total_tested
        self.ATTACK_SUCCESS_RATE = self.ImgNum_ATK_success / self.ImgNum_origin_right

        with open("results_record/" + self.Folder_name + "/" + self.Folder_name + "_ResultInfo.csv", mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ImgI", self.ImgNum_total_tested-1,
                             "SUCC", ATKer.success,
                             "AccQ", ATKer.ACCquery,
                             "NATURAL_ACC", self.NATURAL_ACCURACY_RATE,
                             "ROBUST_ACC", self.ROBUST_ACCURACY_RATE,
                             "ATTACK_SUC", self.ATTACK_SUCCESS_RATE,
                             "ADB_AVG", self.EndADB_avg,
                             "ADB_MID", self.EndADB_mid,
                             "AccI", ATKer.ACCiter_n,
                             "EndADB", ATKer.old_best_adv.ADBmax,
                             "Endl2", ATKer.old_best_adv.RealL2,
                             "Endlinf", ATKer.old_best_adv.RealLinf,
                             "AccQ_avg", self.AccQuery_avg,
                             "AccQ_mid", self.AccQuery_mid,
                             "AccI_avg", self.AccIter_avg,
                             "AccQ/I", self.AccQdI,
                             "AUC_l2", self.AUC_l2
                             ])

    def Summary(self):
        self.L2_LINE_sum = torch.zeros(self.ImgNum_origin_right, self.args.budget + 1).cuda()
        self.Linf_LINE_sum = torch.zeros(self.ImgNum_origin_right, self.args.budget + 1).cuda()
        self.AUC_l2=0
        self.AUC_linf=0
        for ImgI in range(self.ImgNum_origin_right):
            for pair in range(len(self.L2_lines[ImgI]) - 1):
                l,r = self.L2_lines[ImgI][pair][0], min(self.L2_lines[ImgI][pair + 1][0], self.args.budget + 1)
                self.L2_LINE_sum[ImgI,l:r] = self.L2_lines[ImgI][pair][1]
                self.Linf_LINE_sum[ImgI,l:r] = self.Linf_lines[ImgI][pair][1]

        with open("results_record/" + self.Folder_name + "/" + self.Folder_name + "_AtkSucRate.csv", encoding='utf-8', mode='w', newline='') as file:
            writer = csv.writer(file)
            for Qbudget_limit_x in range(1, self.args.budget+1):
                AccImgN = 0
                for querys in self.AccQuery:
                    if querys <= Qbudget_limit_x:
                        AccImgN = AccImgN + 1
                if Qbudget_limit_x % 100 == 0:
                    writer.writerow(["Q|AccRate:", Qbudget_limit_x, AccImgN / self.ImgNum_origin_right
                                 ])

        with open("results_record/" + self.Folder_name + "/" + self.Folder_name + "_DistVsQuery.csv", encoding='utf-8', mode='w', newline='') as file:
            writer = csv.writer(file)
            JUMPnum = 100
            for Qbudget_limit_x in range(1, self.args.budget+1):
                if Qbudget_limit_x % JUMPnum == 0:
                    L2_avg = torch.mean(self.L2_LINE_sum[:, Qbudget_limit_x]).round(decimals=4).item()
                    L2_mid = torch.median(self.L2_LINE_sum[:, Qbudget_limit_x]).round(decimals=4).item()
                    Linf_avg = torch.mean(self.Linf_LINE_sum[:, Qbudget_limit_x]).round(decimals=4).item()
                    Linf_mid = torch.median(self.Linf_LINE_sum[:, Qbudget_limit_x]).round(decimals=4).item()
                    self.AUC_l2 += L2_mid * JUMPnum
                    self.AUC_linf += Linf_mid * JUMPnum
                    writer.writerow(["QBudget:", Qbudget_limit_x,
                                     "L2_avg(mid):",L2_avg,
                                                    L2_mid,
                                     "Linf_avg(mid):",Linf_avg,
                                                    Linf_mid,
                                     ])
        return
        ##################################################################################################


def load_mnist_test_data(test_batch_size=1):
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    # MNIST Dataset
    test_dataset = dsets.MNIST(root='./data/mnist', download=True, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
    return test_loader

def load_Fashion_mnist_test_data(test_batch_size=1):
    test_dataset = dsets.FashionMNIST(root='./data/Fashion_mnist', download=True, train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
    return test_loader

def load_cifar10_test_data(resize_to=32, test_batch_size=1):
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
    ])
    test_dataset = dsets.CIFAR10('./data/cifar10-py', download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
    return test_loader

def load_cifar100_test_data(resize_to=224, test_batch_size=1):
    transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
    ])
    test_dataset = dsets.CIFAR100('./data/cifar100-py', download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)
    return test_loader

def load_imagenet_test_data(test_batch_size=1, folder="./data/imagenet/archive/imagenet-mini/val"):
    val_dataset = dsets.ImageFolder(folder,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))
    """"""
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=True)
    return val_loader


def load_TinyImagenet_val(val_batch_size=1):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    val_file = os.path.join("./data/tiny-imagenet", 'val.npz')
    loaded_npz = np.load(val_file)
    data = loaded_npz['image']
    targets = loaded_npz["label"].tolist()
    images = [Image.fromarray(img) for img in data]
    labels = targets

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __getitem__(self, index):
            img, label = self.images[index], int(self.labels[index])
            if self.transform:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.images)
    val_dataset = SimpleDataset(images=images, labels=labels, transform=test_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=4)
    return val_loader

def load_dataset_model(datasetName,args):
    rand_seed = 42
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.deterministic = True
    if datasetName == 'mnist-cnn':
        model = mnist_model.MNIST().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/mnist_gpu.pt'))
        test_loader = load_mnist_test_data()
        torch_model = GeneralTorchModel(model,args, n_class=10, im_mean=None, im_std=None)
    elif datasetName == 'fashionmnist-cnn':
        model = Fashion_mnist_model.Network().cuda()
        #model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/Fashion_mnist_model.pt'))
        test_loader = load_Fashion_mnist_test_data(1)
        torch_model = GeneralTorchModel(model,args, n_class=10, im_mean=None, im_std=None)
    elif datasetName == 'cifar10-cnn':
        model = cifar10_model.CIFAR10().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        model.load_state_dict(torch.load('model/cifar10_gpu.pt'))
        test_loader = load_cifar10_test_data()
        torch_model = GeneralTorchModel(model,args, n_class=10, im_mean=None, im_std=None)
    elif datasetName == 'cifar10-resnet50':
        #proxies = {'http': 'http://127.0.0.1:17891','https': 'http://127.0.0.1:17891'}
        model_dir = "model/ResNet50-cifar10"
        model = timm.create_model("resnet50_cifar10", pretrained=True).cuda()
        test_loader = load_cifar10_test_data()
        torch_model = GeneralTorchModel(model,args, n_class=10, im_mean=None, im_std=None)
    elif datasetName == 'cifar10-vit':
        model_dir = "model/cifar10-vit"
        processor = AutoImageProcessor.from_pretrained(model_dir)
        model = AutoModelForImageClassification.from_pretrained(model_dir).cuda()
        test_loader = load_cifar10_test_data(224,1)
        torch_model = GeneralTorchModel(model,args, n_class=10, im_mean=None, im_std=None)
    elif datasetName == 'cifar10-wrn':
        model = torch.load("model/cifar10_WRN_70_16.pth").cuda()
        test_loader = load_cifar10_test_data()
        torch_model = GeneralTorchModel(model,args, n_class=10, im_mean=None, im_std=None)
    elif datasetName == 'cifar100-vit':
        model_dir = "model/Vit_cifar100"
        model = AutoModelForImageClassification.from_pretrained(model_dir).cuda()
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        #model = torch.nn.DataParallel(model, device_ids=[0])
        #model.load_state_dict(torch.load('model/resnet18-f37072fd.pth'))
        test_loader = load_cifar100_test_data(224, 1)
        torch_model = GeneralTorchModel(model,args, n_class=100, im_mean=None, im_std=None)
    elif datasetName == 'cifar100-wrn':
        model = torch.load("model/cifar100_WRN-70-16.pth").cuda()
        test_loader = load_cifar100_test_data(32, 1)
        torch_model = GeneralTorchModel(model,args, n_class=100, im_mean=None, im_std=None)
    elif datasetName == 'tinyimagenet-wrn':
        model = torch.load("model/TinyImageNet_WRN-28-10.pth").cuda()
        test_loader = load_TinyImagenet_val()
        torch_model = GeneralTorchModel(model,args, n_class=200, im_mean=None, im_std=None)
    elif datasetName == 'imagenet-vgg':
        model = models.vgg19().cuda()
        #weight = models.vgg19(models.VGG19_Weights.DEFAULT)
        weight = torch.load("model/vgg19-dcbb9e9d.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data()
        torch_model = GeneralTorchModel(model,args, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif datasetName == 'imagenet-vggbn':
        model = models.vgg19_bn().cuda()
        #weight = models.vgg19_bn(models.VGG19_BN_Weights.DEFAULT)
        weight = torch.load("model/vgg19_bn-c79401a0.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data()
        torch_model = GeneralTorchModel(model,args, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif datasetName == 'imagenet-resnet50':
        model = models.resnet50().cuda()
        weight = torch.load("model/resnet50-11ad3fa6.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data()
        torch_model = GeneralTorchModel(model,args, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif datasetName == 'imagenet-inceptionv3':
        model = models.inception_v3().cuda()
        # weight = models.inception_v3(models.Inception_V3_Weights.DEFAULT)
        weight = torch.load("model/inception_v3_google-0cc3c7bd.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data()
        torch_model = GeneralTorchModel(model,args, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif datasetName == 'imagenet-vit':
        model = models.vit_b_32().cuda()
        # weight = models.vit_b_32(models.ViT_B_32_Weights.DEFAULT)
        weight = torch.load("model/vit_b_32-d86f8d99.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data()
        torch_model = GeneralTorchModel(model, args,n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif datasetName == 'imagenet-efficient':
        model = models.efficientnet_b0().cuda()
        # weight = torchvision.models.efficientnet_b0(models.EfficientNet_B0_Weights.DEFAULT)
        weight = torch.load("model/efficientnet_b0_rwightman-7f5810bc.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data()
        torch_model = GeneralTorchModel(model,args, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif datasetName == 'imagenet-densenet':
        model = models.densenet161().cuda()
        #weight = torchvision.models.densenet161(models.DenseNet161_Weights.DEFAULT)
        weight = torch.load("model/densenet161-8d451a50.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data()
        torch_model = GeneralTorchModel(model, args, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])

    else:
        print("Invalid dataset")
        exit(1)
    return test_loader, torch_model.cuda()