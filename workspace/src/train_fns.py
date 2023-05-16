"""
Training Function module

ファイル説明：ネットワークの学習を行う関数を定義
"""
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses


# ネットワークの学習関数を返す関数（クロージャ関数）
# 説明：引数で渡したネットワークで、内部関数のネットワークを固定
# =====================================================================
def training_function(G, D, GD, G_enc_ema, G_dec_ema, state_dict, config):
  def train(x, y):
    # 各ネットワークの勾配をリセット
    G.Enc.optim.zero_grad()
    G.Dec.optim.zero_grad()
    D.Enc.optim.zero_grad()
    
    # 勾配を累積させるために、バッチサイズの塊にデータを分割
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0

    # Discriminatorの更新
    # -------------------------------------------
    # Discriminatorの「requires_grad」のみON
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
    
    for step_index in range(config['num_D_steps']):
      # 勾配のリセット
      D.Enc.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        imgs_real, class_labels = x[counter], y[counter]
        D_pred_real, D_pred_fake = GD(imgs_real, class_labels, 
                                      train_G=False, train_D=True, get_only_preds=True)
        # 損失関数の計算
        D_loss_real, D_loss_fake = losses.calc_dis_loss(D_pred_real, D_pred_fake)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        # 勾配計算
        D_loss.backward()
        counter += 1

      # Discriminatorパラメータの更新
      D.Enc.optim.step()

    # Generatorの更新
    # -------------------------------------------
    # Generatorの「requires_grad」のみON
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)

    G.Enc.optim.zero_grad()
    G.Dec.optim.zero_grad()
    counter = 0

    for accumulation_index in range(config['num_G_accumulations']):
      imgs_real, class_labels = x[counter], y[counter]
      imgs_fake, D_pred_fake = GD(imgs_real, class_labels, train_G=True)
      # 損失関数の計算
      G_loss_con, G_loss_adv = losses.calc_gen_loss(imgs_real, imgs_fake, D_pred_fake)
      G_loss = (G_loss_con + G_loss_adv) / float(config['num_G_accumulations'])
      # 勾配計算
      G_loss.backward()
      counter += 1
    
    # Generatorパラメータの更新
    G.Enc.optim.step()
    G.Dec.optim.step()

    # EMAの更新
    if config['ema']:
      G_enc_ema.update(state_dict['itr'])
      G_dec_ema.update(state_dict['itr'])

    # Lossの可視化のために辞書を作成
    out = {'G_loss_con': float(G_loss_con.detach().clone().cpu()), 
           'G_loss_adv': float(G_loss_adv.detach().clone().cpu()),
           'D_loss_real': float(D_loss_real.detach().clone().cpu()), 
           'D_loss_fake': float(D_loss_fake.detach().clone().cpu()),}

    return out
  return train