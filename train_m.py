import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar

from model_m import autoencoder, network
from utils.utils import weights_init_normal
import os

import ipdb
class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device

        # Create the directory path for the weights, including the parent directory.
        self.weights_path = f'/home/cal-05/hj/SVDD/weights/mnist/{self.args.normal_class}'
        self.pretrained_weights_path = f'{self.weights_path}/pretrained_parameters.pth'

        # Ensure the directory for the weights exists.
        self.ensure_directory_exists(self.weights_path)

    def ensure_directory_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    def pretrain(self):
        """Pretraining the weights for the deep SVDD network using autoencoder"""
        ae = autoencoder(self.args.latent_dim).to(self.device)

        # 학습 초기 가중치 setting 
        ae.apply(weights_init_normal)

        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae, weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)
        
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0

            # train_loader에서 batch를 순차적으로 불러옴
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                # 153.6943359375
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(epoch, total_loss/len(self.train_loader)))
        self.save_weights_for_DeepSVDD(ae, self.train_loader) 
    
    def save_weights_for_DeepSVDD(self, model, dataloader):

        """Initialize Deep SVDD weights using the encoder weights of the pretrained autoencoder."""
        c = self.set_c(model, dataloader)
        
        net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(), 'net_dict': net.state_dict()}, self.pretrained_weights_path)

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""

        # 일부 layer가 train mode와 다르게 동작 
        # model이 기존에 학습된 가중치를 바탕으로 데이터 처리
        model.eval()

        z_ = []
        # gradient 계산 비활성화 -> 메모리 사용을 줄이고 연산 속도를 향상시킴 
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())

        # len(z_): 5,923
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)

        # c값이 너무 작으면 -> 모델의 성능에 영향을 줄 수 있음 -> eps로 대체 
        # 값들을 안정적인 범위 내로 제한하기 위함 
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        print(c)
        return c


    def train(self):
        """Training the Deep SVDD model"""
        net = network(self.args.latent_dim).to(self.device)
        
        if self.args.pretrain==True:
            state_dict = torch.load(self.pretrained_weights_path)
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)
        
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        '''
            초구의 부피를 줄임
        '''
        net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        self.net = net
        self.c = c
                

        

