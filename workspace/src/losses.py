"""
Loss Functions module

ファイル説明：損失関数の定義
"""
import torch
import torch.nn.functional as F

# Discriminator
# ==========================================
def loss_dis_hinge(pred_real, pred_fake):
  loss_real = torch.mean(F.relu(1. - pred_real))
  loss_fake = torch.mean(F.relu(1. + pred_fake))
  return loss_real, loss_fake

def calc_dis_loss(pred_real, pred_fake):
  loss_real, loss_fake = loss_dis_hinge(pred_real, pred_fake)
  return loss_real, loss_fake


# Generator
# ==========================================
def loss_gen_con(imgs_real, imgs_fake):
  loss = 100 * torch.mean(torch.abs(imgs_real - imgs_fake))
  return loss

def loss_gen_adv(pred_fake):
  loss = -torch.mean(F.relu(pred_fake))
  return loss

def calc_gen_loss(imgs_real, imgs_fake, pred_fake):
  loss_con = loss_gen_con(imgs_real, imgs_fake)
  loss_adv = loss_gen_adv(pred_fake)
  return loss_con, loss_adv