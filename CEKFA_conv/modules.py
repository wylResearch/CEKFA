#### Import all the supporting classes
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import *


class BottleneckBlock(nn.Module):
    """
    Bert-ResNet 的版本,不会减少feature map的长宽,只会改变通道数目
    """
    def __init__(self, args, input_dim, int_dim, output_dim, stride=1):
        super(BottleneckBlock, self).__init__()
        # print('--', input_dim, int_dim, output_dim)
        self.conv1 = nn.Conv2d(input_dim, int_dim, (1, 1), stride=stride)
        self.conv2 = nn.Conv2d(int_dim, int_dim, (3, 3), padding=1, padding_mode='circular')
        self.conv3 = nn.Conv2d(int_dim, output_dim, (1, 1))

        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        self.bn1 = torch.nn.BatchNorm2d(input_dim)
        self.bn2 = torch.nn.BatchNorm2d(int_dim)
        self.bn3 = torch.nn.BatchNorm2d(int_dim)

        if input_dim != output_dim:
            self.proj_shortcut = nn.Conv2d(
                input_dim, output_dim, (1, 1), stride=stride)
        else:
            self.proj_shortcut = None
        

    # def init(self):
    #     nn.init.xavier_normal_(self.rel_embedding.weight.data)

    def forward(self, features):
        # print('---', features.shape)
        x = self.bn1(features)      # bs,input_dim,5,5                  # bs,input_cha,32,48
        x = F.relu(x)
        if self.proj_shortcut:
            features = self.proj_shortcut(features)
        x = self.feature_map_drop(x)
        x = self.conv1(x)           #  bs,input_dim//4,5,5              # bs,output_cha//4,32,48
        x = self.bn2(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = self.conv3(x)           # bs,output_dim,5,5                  # bs,output_cha,32,48
        # print(x.shape)
        
        x += features

        return x


class ModelBottleneckBlocks(nn.Module):
    """
    将之前的ResNet部分的 bottlenecks 封装起来
    """
    def __init__(self, args):
        super(ModelBottleneckBlocks, self).__init__()
        self.args = args
        input_dim = args.bert_dim
        output_dim = input_dim
        bottlenecks = []
        for i in range(args.resnet_num_blocks): # 2
            # print('-----------')
            if i == 0:
                bottlenecks.extend([BottleneckBlock(
                    args, input_dim, output_dim//4, output_dim) for _ in range(args.resnet_block_depth)])   # resnet_block_depth: 3
            else:
                bottlenecks.append(BottleneckBlock(
                    args, input_dim, output_dim//4, output_dim, stride=1))
                input_dim = output_dim
                bottlenecks.extend([BottleneckBlock(
                    args, input_dim, output_dim//4, output_dim) for _ in range(min(args.resnet_block_depth, 2)-1)])
            output_dim *= 2
        self.output_dim = output_dim//2
        self.bottlenecks = nn.Sequential(*bottlenecks)

    def forward(self, x):
        x = self.bottlenecks(x)
        return x


class ConvEntAndRel(nn.Module):
    """
    将之前的ResNet部分编码ent与rel的部分封装起来,其实就是把 query_emb 放在class外面了
    """
    def __init__(self, args, input_c=2):
        super(ConvEntAndRel, self).__init__()
        assert input_c in [2,3]
        self.reshape_len = args.reshape_len
        self.bn0 = torch.nn.BatchNorm1d(input_c)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.conv1 = nn.Conv1d(input_c, self.reshape_len**2, kernel_size=1)

        self.bottlenecks = ModelBottleneckBlocks(args)
        self.output_dim = self.bottlenecks.output_dim
        
        self.bn1 = torch.nn.BatchNorm2d(self.output_dim)
        self.hidden_drop = torch.nn.Dropout(args.resnet_dropout)
        self.fc = nn.Linear(self.output_dim, args.bert_dim)
        self.prelu = torch.nn.PReLU(args.bert_dim)
        self.args = args
        self.input_c = input_c
    
    def forward(self, ent_embedded, rel_embedded, ent_canonical=None):
        bs = ent_embedded.shape[0]
        if self.input_c == 2:
            stacked_inputs = torch.cat([ent_embedded.unsqueeze(1), rel_embedded.unsqueeze(1)], 1)
        else:
            stacked_inputs = torch.cat([ent_embedded.unsqueeze(1), ent_canonical.unsqueeze(1), rel_embedded.unsqueeze(1)], 1)
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)        # (bs,2,768)
        x = self.conv1(x)           # (bs,self.reshape_len**2,768)
        x = torch.transpose(x, 1, 2).view(bs, self.args.bert_dim, self.reshape_len, self.reshape_len).contiguous()
        x = self.bottlenecks(x)     # (bs,768x2, 5, 5)
        x = self.bn1(x)
        x = F.relu(x)
        x = torch.mean(x.view(bs, self.output_dim, -1), dim=2)  # (bs,1536)
        x = self.hidden_drop(x)
        x = self.fc(x)              # (bs,768)
        x = self.prelu(x)
        x = self.hidden_drop(x)
        return x





