import torch
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def global_contrast_normalization(x):
    """Apply global contrast normalization to tensor. """

    # 이미지의 모든 pixel 값에 대한 평균을 계산
    # 평균이 0이 되도록 함
    # 이미지의 밝기 차이를 제거하여 대비에만 집중할 수 있게 함 
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    # 전체 이미지 pixel이 평균으로부터 얼마나 떨어져 있는지를 나타내는 척도 
    x_scale = torch.mean(torch.abs(x))
    # 데이터의 스케일을 조정하여 대비를 표준화하는 과정
    x /= x_scale

    return x