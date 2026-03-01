import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .common import MLPBlock, LayerNorm2d, Conv, GradientLoss, Dense, UpSampling, ContrastiveLoss


class Losses(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mse = nn.MSELoss()
        self.grad = GradientLoss()
        self.contrast = ContrastiveLoss()

    '''
    output 是学生的输出 (output_student)
    ref 是教师的输出 (output_teacher)
    '''
    def cal(self, output, y, ir, ref, y_mask, ir_mask):
        # 使学生网络的融合结果在像素级别上接近教师网络
        loss_fuse = 3*self.mse(output , y ) + 2*self.mse(output , ir ) + 4*self.mse(output, ref)  # (1) 输出对齐 (像素级)
        # loss_fuse = torch.tensor(0.0)       完全禁用像素级别的融合损失。模型将不会直接从像素值上学习如何融合来自 y (可见光亮度)、ir (红外) 或 ref (教师输出) 的信息
        # loss_fuse = 3*self.mse(output, ref)
        loss_grad = self.grad(output, y, ir) * 9 + self.grad(output, ref, ref) * 3   # (2) 梯度对齐
        # loss_grad = 6*self.grad(output, ref, ref)
        # loss_grad =torch.tensor(0.0) 完全禁用梯度损失  做消融实验
        # 计算学生输出和教师输出在特征空间的相似度。这也是一种输出对齐，但更侧重于高层语义和感知质量
        loss_contrast, DH_value = self.contrast(output, y, ir, ref, y_mask, ir_mask)  # (3) 感知/对比损失
        # loss_contrast, DH_value = torch.tensor(0.0), [torch.tensor(0.0)] * 5

        loss_contrast /= 3000
        # 最终的损失
        loss = 3*loss_fuse + 3*loss_grad + loss_contrast
        # loss = loss_contrast
        return loss, loss_fuse, loss_grad, loss_contrast, DH_value


