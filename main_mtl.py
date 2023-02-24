import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import random, os
import argparse

from collections import OrderedDict
from scipy.special import softmax

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from mtl_pytorch.trainer import Trainer
from mobilenetv2 import mobilenet_v2
from head import ClassificationHead
from utils import *

# -----------------------------------

def main(args):

    print("---------------")
    print(args)
    print("---------------")

    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 2, 'pin_memory': False}

    ### load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                            download=True, transform=transform_test)
        
    classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    x_train, x_test = trainset.data, testset.data
    y_train, y_test = trainset.targets, testset.targets

    animal_trainset = DatasetMaker([get_class_i(x_train, y_train, [2,3,4,5,6,7]),
                                    get_class_i(x_train, y_train, [0,1,8,9])], 
                                   transform_with_aug)
    animal_testset = DatasetMaker([get_class_i(x_test, y_test, [2,3,4,5,6,7]),
                                   get_class_i(x_test, y_test, [0,1,8,9])], 
                                  transform_no_aug)

    tasks = ['multiclass', 'binary']
    task_cls_num = {'multiclass': 10, 'binary': 2}
    train_dataset = {"multiclass": trainset, 
                     "binary": animal_trainset}
    val_dataset = {"multiclass": testset, 
                   "binary": animal_testset}
    criterion = nn.CrossEntropyLoss()

    headsDict = nn.ModuleDict()
    trainDataloaderDict = {task: [] for task in tasks}
    valDataloaderDict = {}
    criterionDict = {}
    metricDict = {}

    for task in tasks:
        headsDict[task] = ClassificationHead(task_cls_num[task])

        trainDataloaderDict[task] = DataLoader(train_dataset[task], 
                                                batch_size=args.batch_size, 
                                                shuffle=True, **kwargs)
        valDataloaderDict[task] = DataLoader(val_dataset[task], batch_size=100, 
                                                shuffle=False, **kwargs)
        criterionDict[task] = criterion
        metricDict[task] = []
    
    ### Define MTL model
    mtlmodel = mobilenet_v2(False, heads_dict=headsDict)
    mtlmodel = mtlmodel.to(device)

    ### Define training framework
    trainer = Trainer(mtlmodel, 
                        trainDataloaderDict, valDataloaderDict, 
                        criterionDict, metricDict, 
                        print_iters=10, val_iters=100, 
                        save_iters=100, save_num=1, 
                        policy_update_iters=100)

    # ==================================
    ### Train
    checkpoint = 'checkpoint/'
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    savepath = checkpoint+args.save_dir+"/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    print(f"All ckpts save to {savepath}")

    # ----------------
    ### Step 1: pre-train
    if args.pretrain:
        print(">>>>>>>> pre-train <<<<<<<<<<")
        trainer.pre_train(iters=args.pretrain_iters, lr=args.lr, 
                          savePath=savepath, writerPath=savepath)

    # ----------------
    ### Step 2: alter-train
    if args.alter_train:
        print(">>>>>>>> alter-train <<<<<<<<<<")
        loss_lambda = {'multiclass': 1, 'binary': 1, 'policy': 0.0005}
        trainer.alter_train_with_reg(iters=args.alter_iters, 
                                     policy_network_iters=(50,200), 
                                     policy_lr=0.01, network_lr=0.0001,
                                     loss_lambda=loss_lambda,
                                     savePath=savepath, writerPath=savepath)

    # ----------------
    if args.post_train:
        ### Step 3: sample policy from trained policy distribution and save
        print(">>>>>>>> Sample Policy <<<<<<<<<<")
        policy_list = {'multiclass': [], 'binary': []}
        name_list = {'multiclass': [], 'binary': []}

        for name, param in mtlmodel.named_parameters():
            if 'policy' in name :
                print(name)
                if 'multiclass' in name:
                    policy_list['multiclass'].append(param.data.cpu().detach().numpy())
                    name_list['multiclass'].append(name)
                elif 'binary' in name:
                    policy_list['binary'].append(param.data.cpu().detach().numpy())
                    name_list['binary'].append(name)


        shared = 5
        sample_policy_dict = OrderedDict()
        for task in tasks:
            count = 0
            for name, policy in zip(name_list[task], policy_list[task]):
                if count < shared:
                    sample_policy_dict[name] = torch.tensor([1.0, 0.0, 0.0]).cuda()
                else:
                    distribution = softmax(policy, axis=-1)
                    distribution /= sum(distribution)
                    choice = np.random.choice((0, 1, 2), p=distribution)
                    if choice == 0:
                        sample_policy_dict[name] = torch.tensor([1.0, 0.0, 0.0]).cuda()
                    elif choice == 1:
                        sample_policy_dict[name] = torch.tensor([0.0, 1.0, 0.0]).cuda()
                    elif choice == 2:
                        sample_policy_dict[name] = torch.tensor([0.0, 0.0, 1.0]).cuda()
                count += 1

        sample_path = savepath
        sample_state = {'state_dict': sample_policy_dict}
        torch.save(sample_state, sample_path + 'sample_policy.model')

        # ----------------
        ### Step 4: post train from scratch
        print(">>>>>>>> Post-train <<<<<<<<<<")
        loss_lambda = {'multiclass': 10, 'binary': 1}
        trainer.post_train(iters=args.post_iters, lr=args.post_lr,
                            decay_lr_freq=args.decay_lr_freq, decay_lr_rate=0.5,
                            loss_lambda=loss_lambda,
                            savePath=savepath, writerPath=savepath,
                            reload='sample_policy.model',
                            ext=args.ext)

# --------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ### DONN parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=128, 
                        help="cifar10: train: 50k; test: 10k")
    parser.add_argument('--pretrain-iters', type=int, default=8000, 
                        help='#iterations for pre-training, default: [8k: bz=128]')
    parser.add_argument('--alter-iters', type=int, default=6000, 
                        help='#iterations for alter-train, default: 20000')
    parser.add_argument('--post-iters', type=int, default=30000, 
                        help='#iterations for post-train, default: 30000')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='pre-train learning rate')
    parser.add_argument('--post-lr', type=float, default=0.01, 
                        help='post-train learning rate')
    parser.add_argument('--decay-lr-freq', type=float, default=2000, 
                        help='post-train learning rate decay frequency')

    parser.add_argument('--save-dir', type=str, default='avg', help="save the model")
    parser.add_argument('--evaluate', type=str, help="Model path for evaulation")

    parser.add_argument('--pretrain', action='store_true', default=False, 
                        help='whether to run pre-train part')
    parser.add_argument('--alter-train', action='store_true', default=False, 
                        help='whether to run alter-trian part')
    parser.add_argument('--post-train', action='store_true', default=False, 
                        help='whether to run post-train part')

    parser.add_argument('--ext', type=str, default='', 
                        help="extension for save the model")

    args_ = parser.parse_args()

    main(args_)