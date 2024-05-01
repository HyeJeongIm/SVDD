# we used the precomputed min_max values from the original implementation: 
# https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/1901612d595e23675fb75c4ebb563dd0ffebc21e/src/datasets/mnist.py

import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image

from utils.utils import global_contrast_normalization
from torch.utils.data import ConcatDataset


class MNIST_loader(data.Dataset):
    """This class is needed to processing batches for the dataloader."""
    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        """return transformed items."""
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y

    def __len__(self):
        """number of samples."""
        return len(self.data)
    
def get_mnist_combined_test_dataloader(dataloader_test_normal, dataloader_test_abnormal, batch_size):
    """Combine two dataloaders into one."""
    
    combined_test_dataset = ConcatDataset([dataloader_test_normal.dataset, dataloader_test_abnormal.dataset])
    dataloader_test_combined = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return dataloader_test_combined

def get_mnist(args, data_dir='./data/mnist/'):
    """get dataloders"""
    # min, max values for each class after applying GCN (as the original implementation)
    min_max = [(-0.8826567065619495, 9.001545489292527),
                (-0.6661464580883915, 20.108062262467364),
                (-0.7820454743183202, 11.665100841080346),
                (-0.7645772083211267, 12.895051191467457),
                (-0.7253923114302238, 12.683235701611533),
                (-0.7698501867861425, 13.103278415430502),
                (-0.778418217980696, 10.457837397569108),
                (-0.7129780970522351, 12.057777597673047),
                (-0.8280402650205075, 10.581538445782988),
                (-0.7369959242164307, 10.697039838804978)]
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x)),
                                    transforms.Normalize([min_max[args.normal_class][0]],
                                                         [min_max[args.normal_class][1] \
                                                         -min_max[args.normal_class][0]])])
    
    # 데이터 로딩 및 변환
    all_train = datasets.MNIST(root=data_dir, train=True, download=True)
    all_test = datasets.MNIST(root=data_dir, train=False, download=True)

    '''
        train dataset
    '''

    # 정상(normal) 클래스 필터링
    normal_indices = all_train.targets == args.normal_class
    x_train_normal = all_train.data[normal_indices]
    y_train_normal = all_train.targets[normal_indices]
    y_train_normal = torch.zeros_like(y_train_normal) 

    '''
        test dataset
    '''

    # 정상(normal) 클래스 - 테스트 데이터
    x_test_normal = all_test.data[all_test.targets == args.normal_class]
    y_test_normal = all_test.targets[all_test.targets == args.normal_class]
    y_test_normal = torch.zeros_like(y_test_normal) 

    # 비정상(abnormal) 클래스 필터링 - 테스트 데이터에서만
    abnormal_indices = all_test.targets == args.abnormal_class
    x_test_abnormal = all_test.data[abnormal_indices]
    y_test_abnormal = all_test.targets[abnormal_indices]
    y_test_abnormal = torch.ones_like(y_test_abnormal) 

    # 데이터셋 생성
    train_dataset = MNIST_loader(x_train_normal, y_train_normal, transform)
    test_normal_dataset = MNIST_loader(x_test_normal, y_test_normal, transform)
    test_abnormal_dataset = MNIST_loader(x_test_abnormal, y_test_abnormal, transform)

    # 데이터로더 생성
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dataloader_test_normal = DataLoader(test_normal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataloader_test_abnormal = DataLoader(test_abnormal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # combined test dataset
    dataloader_test = get_mnist_combined_test_dataloader(dataloader_test_normal, dataloader_test_abnormal, args.batch_size)
    
    return dataloader_train, dataloader_test