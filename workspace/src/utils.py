#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities module

ファイル説明：汎用的に使用可能な変数・関数・クラスをまとめたファイル
"""

# 一般的なモジュールのインポート
import sys
import os
import numpy as np
import time
import datetime
from argparse import ArgumentParser
import json
from typing import Sequence, Union
from matplotlib import pyplot as plt

# Pytorch関連モジュールのインポート
import torch
from torch import Tensor
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Datasetクラスモジュールのインポート
import datasets as dset


# コマンドライン引数の設定
# =====================================================================
def prepareParser():
  usage = '実験条件指定用のコマンドライン引数'
  parser = ArgumentParser(description=usage)

  ### Dataset, Dataloaderに関する引数 ###
  parser.add_argument(
    '--dataset', type=str, default='CT256',
    help='使用するデータセットの指定')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='dataloaderのワーク数の指定')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='データをシャッフルするか否かの指定')
  parser.add_argument(
    '--data_resize', action='store_true', default=False,
    help='データをリサイズするか否かの指定')
  parser.add_argument(
    '--use_multiepoch_sampler', action='store_true', default=False, 
    help='')

  #### モデル構築に関する引数 ###
  parser.add_argument(
    '--G_param', type=str, default='SN', 
    help='Generatorのバッチ正規化層にSpectral Normalizationを使用するか否かの指定')
  parser.add_argument(
    '--D_param', type=str, default='SN', 
    help='Discriminatorのバッチ正規化層にSpectral Normalizationを使用するか否かの指定')
  parser.add_argument(
    '--G_ch', type=int, default=64, 
    help='Generatorのチャネル数の指定')
  parser.add_argument(
    '--D_ch', type=int, default=64, 
    help='Discriminatorのチャネル数の指定')
  parser.add_argument(
    '--G_depth', type=int, default=1, 
    help='GeneratorのResBlock 1つの深さ指定')
  parser.add_argument(
    '--D_depth', type=int, default=1, 
    help='DiscriminatorのResBlock 1つの深さ指定')
  parser.add_argument(
    '--G_shared', action='store_true', default=False,
    help='Generatorの共有埋め込みを使用するか否か指定')
  parser.add_argument(
    '--shared_dim', type=int, default=0,
    help='Generatorの共有埋め込みにおける次元数の指定')
  parser.add_argument(
    '--hier', action='store_true', default=False, 
    help='Generatorで階層的ノイズ分割を使用するか否か指定')
  parser.add_argument(
    '--Genc_nl', type=str, default='relu',
    help='Encoder(Generator)の活性化関数の指定')
  parser.add_argument(
    '--Gdec_nl', type=str, default='relu',
    help='Decoder(Generator)の活性化関数の指定')
  parser.add_argument(
    '--Denc_nl', type=str, default='relu',
    help='Encoder(Discriminator)の活性化関数の指定')
  parser.add_argument(
    '--G_attn', type=str, default='64',
    help='GeneratorでSelf Attentionを導入する解像度の指定')
  parser.add_argument(
    '--D_attn', type=str, default='64',
    help='DiscriminatorでSelf Attentionを導入する解像度の指定')
  parser.add_argument(
    '--G_ccbn_norm', action='store_true', default=False,
    help='GeneratorのCCBNで潜在変数・クラスベクトルを正規化するかの指定')
  parser.add_argument(
    '--D_ccbn_norm', action='store_true', default=False,
    help='GeneratorのCCBNで潜在変数・クラスベクトルを正規化するかの指定')
  parser.add_argument(
    '--norm_style', type=str, default='bn',
    help='正規化層の種類の指定（bn [batchnorm], in [instancenorm], ln [layernorm], gn [groupnorm]')
  parser.add_argument(
    '--latent_dim', type=int, default=2000,
    help='潜在変数の次元数の指定')

  #### モデル初期化に関する引数 ###
  parser.add_argument(
    '--seed', type=int, default=0,
    help='初期化に使用する乱数シードの指定')
  parser.add_argument(
    '--G_init', type=str, default='ortho',
    help='Generatorの初期化メソッドの指定')
  parser.add_argument(
    '--D_init', type=str, default='ortho',
    help='Discriminatorの初期化メソッドの指定')
  parser.add_argument(
    '--skip_init', action='store_true', default=False,
    help='ネットワークの初期化をスキップするか否か指定（検証・評価時に使用）')
  
  #### 最適化アルゴリズムに関する引数 ###
  #### 基本的には「Adam」を使用 ###
  parser.add_argument(
    '--G_lr', type=float, default=5e-5,
    help='Generatorの初期学習率の指定')
  parser.add_argument(
    '--D_lr', type=float, default=5e-5,
    help='Discriminatorの初期学習率の指定')
  parser.add_argument(
    '--G_B1', type=float, default=0.0, 
    help='Generator パラメータ勾配の移動平均（Beta1）')
  parser.add_argument(
    '--D_B1', type=float, default=0.0, 
    help='Discriminator パラメータ勾配の移動平均（Beta1）')
  parser.add_argument(
    '--G_B2', type=float, default=0.0, 
    help='Generator 二乗パラメータ勾配の移動平均（Beta2）')
  parser.add_argument(
    '--D_B2', type=float, default=0.0, 
    help='Discriminator 二乗パラメータ勾配の移動平均（Beta2）')

  #### 学習の実行に関する引数 ###
  parser.add_argument(
    '--num_epochs', type=int, default=100,
    help='Number of epochs to train for (default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='デフォルトバッチサイズの指定')
  parser.add_argument(
    '--G_batch_size', type=int, default=0,
    help='Generatorのバッチサイズの指定')
  parser.add_argument(
    '--num_G_accumulations', type=int, default=1,
    help='Generatorの勾配の累積回数を指定')  
  parser.add_argument(
    '--num_D_steps', type=int, default=2,
    help='Generator学習1回に対する, Discriminator学習回数')
  parser.add_argument(
    '--num_D_accumulations', type=int, default=1,
    help='Discriminatorの勾配の累積回数を指定')
  parser.add_argument(
    '--pbar', type=str, default='tqdm',
    help='学習状況の詳細表示関数の指定')
  
  #### 検証・評価の実行に関する引数 ###
  parser.add_argument(
    '--G_eval_mode', action='store_true', default=False,
    help='Generatorを検証モードで動かすか否か指定')
  parser.add_argument(
    '--save_frequency', type=int, default=2000,
    help='画像保存を行うイテレーションの指定')
  parser.add_argument(
    '--num_save_copies', type=int, default=2,
    help='コピーの保存枚数の指定')
  parser.add_argument(
    '--num_best_copies', type=int, default=2,
    help='保存する最適なチェックポイントの数の指定')
  parser.add_argument(
    '--test_frequency', type=int, default=2000,
    help='検証・評価を行うイテレーションの指定')
  
  #### データの保存場所に関する引数 ###
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='学習データの保存場所のルート指定')
  parser.add_argument(
    '--save_root', type=str, default='saves',
    help='全ての保存データのルート指定')
  parser.add_argument(
    '--weights_folder', type=str, default='weights',
    help='重みの保存場所のルート指定')
  parser.add_argument(
    '--images_folder', type=str, default='images',
    help='生成画像などの保存場所のルート指定')
  parser.add_argument(
    '--option_folder', type=str, default='option',
    help='実験オプションの保存場所のルート指定')
  parser.add_argument(
    '--trainCurve_folder', type=str, default='trainCurve',
    help='学習曲線の保存場所のルート指定')
  parser.add_argument(
    '--structure_folder', type=str, default='structure',
    help='ネットワーク構成の保存場所のルート指定')

  #### EMA（指数平滑移動平均）に関する引数 ###
  parser.add_argument(
    '--ema', action='store_true', default=False,
    help='GeneratorにEMAを適用する指定')
  parser.add_argument(
    '--ema_decay', type=float, default=0.9999,
    help='EMA decay rate (default: %(default)s)')
  parser.add_argument(
    '--use_ema', action='store_true', default=False,
    help='検証・評価にEMAパラメータを使用するか否か指定')
  parser.add_argument(
    '--ema_start', type=int, default=0,
    help='EMAアップデートを開始するイテレーションの指定')

  #### Epcilon（計算不可能を回避する微小数）に関する引数 ###
  parser.add_argument(
    '--adam_eps', type=float, default=1e-8,
    help='epsilon value to use for Adam (default: %(default)s)')
  parser.add_argument(
    '--BN_eps', type=float, default=1e-5,
    help='epsilon value to use for BatchNorm (default: %(default)s)')
  parser.add_argument(
    '--SN_eps', type=float, default=1e-8,
    help='epsilon value to use for Spectral Norm(default: %(default)s)')

  #### Ortho 初期化に関する引数 ###
  parser.add_argument(
    '--G_ortho', type=float, default=0.0,
    help='Ortho初期化の係数（Generator）指定')
  parser.add_argument(
    '--D_ortho', type=float, default=0.0,
    help='Ortho初期化の係数（Discriminator）指定')
  parser.add_argument(
    '--toggle_grads', action='store_true', default=True,
    help='Generator or Discriminatorの自動勾配をON or OFF'
         ' (default: %(default)s)')
  
  #### Resume training stuff ###
  parser.add_argument(
    '--load_weights', type=str, default='',
    help='Suffix for which weights to load (e.g. best0, copy0) '
         '(default: %(default)s)')
  parser.add_argument(
    '--resume', action='store_true', default=False,
    help='Resume training? (default: %(default)s)')
  
  return parser


# 汎用的な辞書の宣言 (SB: Split Body -> 体軸方向の分割，ID割り当てによる3次元情報の埋め込み)
# =====================================================================================
dset_dict = {'CT256': dset.DicomDatasets, 'CT128': dset.DicomDatasets, 
             'CT256SB-class1': dset.DicomDatasets, 'CT128SB-class1': dset.DicomDatasets,
             'CT256SB-class4': dset.DicomDatasets, 'CT128SB-class4': dset.DicomDatasets, 
             'CT256SB-class8': dset.DicomDatasets, 'CT128SB-class8': dset.DicomDatasets,} 
sample_dset_dict = {'CT256': dset.DicomSampleDatasets, 'CT128': dset.DicomSampleDatasets, 
             'CT256SB-class1': dset.DicomSampleDatasets, 'CT128SB-class1': dset.DicomSampleDatasets,
             'CT256SB-class4': dset.DicomSampleDatasets, 'CT128SB-class4': dset.DicomSampleDatasets,
             'CT256SB-class8': dset.DicomSampleDatasets, 'CT128SB-class8': dset.DicomSampleDatasets,} 
imsize_dict = {'CT256': 256, 'CT128': 128,
               'CT256SB-class1': 256, 'CT128SB-class1': 128,
               'CT256SB-class4': 256, 'CT128SB-class4': 128,
               'CT256SB-class8': 256, 'CT128SB-class8': 128,}
root_dict = {'CT256': 'ct', 'CT128': 'ct',
             'CT256SB-class1': 'ct-split-class1', 'CT128SB-class1': 'ct-split-class1',
             'CT256SB-class4': 'ct-split-class4', 'CT128S4-class4': 'ct-split-class4',
             'CT256SB-class8': 'ct-split-class8', 'CT128S8-class8': 'ct-split-class8'}
sample_root_dict = {'CT256': 'sample', 'CT128': 'sample', 
                    'CT256SB-class1': 'sample-split-class1', 'CT128SB-class1': 'sample-split-class1',
                    'CT256SB-class4': 'sample-split-class4', 'CT128SB-class4': 'sample-split-class4',
                    'CT256SB-class8': 'sample-split-class8', 'CT128SB-class8': 'sample-split-class8'}
nclass_dict = {'CT256': 200, 'CT128': 200,
               'CT256SB-class1': 1, 'CT128SB-class1': 1,
               'CT256SB-class4': 4, 'CT128SB-class4': 4,
               'CT256SB-class8': 8, 'CT128SB-class8': 8}
activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True)}


# 汎用的な補助関数
# =====================================================================
# 初期化
def initialize_all(config):
  config['save_date'] = str(datetime.datetime.today().date())
  
  # 各データの保存先作成
  prepare_folder(config)

  # 実験オプションの保存
  save_option(config)

  # Configのアップデート
  config['resolution'] = imsize_dict[config['dataset']]
  config['n_classes'] = nclass_dict[config['dataset']]
  config['Genc_activation'] = activation_dict[config['Genc_nl']]
  config['Ddec_activation'] = activation_dict[config['Gdec_nl']]
  config['Denc_activation'] = activation_dict[config['Denc_nl']]

  # 乱数シードの初期化
  seed_rng(config['seed'])
  # GPU使用率の上限解放 ⇒ GPuを使用した処理の高速化
  torch.backends.cudnn.benchmark = True
  # Cudaデバイスの指定
  if torch.cuda.is_available:
    config['device'] = 'cuda'
  else:
    config['device'] = 'cpu'

  return config

# 乱数シードを設定
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

# 結果の保存先フォルダを作成
def prepare_folder(config):
  for key in ['weights_folder', 'images_folder', 'option_folder', 'trainCurve_folder', 'structure_folder']:
    dir_path = os.path.join(config['save_root'], config['save_date'], config[key])
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

# パラメータの自動勾配計算をON・OFF
def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off
    

# 任意のメッセージを表示
# 仕様：  ======================================
#         Message
#         =====================================
def show_message(message):
  print('')
  print("=========================================================================")
  print('{}'.format(message))

# フォルダの作成
def make_folder(path_folder):
  if not os.path.exists(path_folder):
    os.makedirs(path_folder)

# ネットワークの入出力テスト
def test_net(config, net, loader):
  loader = iter(loader)
  x, y = next(loader)
  x, y = x.to(config['device']), y.to(config['device'])
  imgs_fake, pred_fake = net(x, y)
  return imgs_fake, pred_fake

# テンソルの正規化（範囲：0 ~ 1）
def normalize_tensor(x: Tensor, axis: Union[int, Sequence[int], None]=None):
  min = tensor_min(x, axis)
  max = tensor_max(x, axis)
  tensor_normalized = (x - min) / (max - min)
  tensor_scaled = tensor_normalized * 1
  return tensor_scaled

# テンソルの最大値取得
def tensor_max(x: Tensor, axis: Union[int, Sequence[int], None]=None, keepdims: bool=False) -> Tensor:
  if axis is None:
    axis = range(x.ndim)
  elif isinstance(axis, int):
    axis = [axis]
  else:
    axis = sorted(axis)
  
  for ax in axis[::-1]:
    x = x.max(dim=ax, keepdim=keepdims)[0]
  
  return x

# テンソルの最小値取得
def tensor_min(x: Tensor, axis: Union[int, Sequence[int], None]=None, keepdims: bool=False) -> Tensor:
  if axis is None:
    axis = range(x.ndim)
  elif isinstance(axis, int):
    axis = [axis]
  else:
    axis = sorted(axis)
  
  for ax in axis[::-1]:
    x = x.min(dim=ax, keepdim=keepdims)[0]
  
  return x

# Generatorネットワークの詳細を保存
def save_gen_detail(config, net):
  from torchinfo import summary
  from torchvision.models.feature_extraction import get_graph_node_names
  
  # Layerの名前を取得
  enc_node_names = get_graph_node_names(net.Enc)
  dec_node_names = get_graph_node_names(net.Dec)
  
  # ネットワークの構成を取得
  model_stats = summary(net, [(1, 1, imsize_dict[config['dataset']], imsize_dict[config['dataset']]), (1,)], device='cpu', dtypes=[torch.float, torch.long], depth=5)
  
  # Textファイルとして保存
  root = os.path.join(config['save_root'], config['save_date'], config['structure_folder'])
  with open(os.path.join(root, 'layerName_Genc.txt'), 'w') as f:
    f.write(str(enc_node_names))
  with open(os.path.join(root, 'layerName_Gdec.txt'), 'w') as f:
    f.write(str(dec_node_names))
  with open(os.path.join(root, 'Generator.txt'), 'w') as f:
    f.write(str(model_stats))



# DataLoaderに渡すSamplerクラス
# =====================================================================
# multi-epoch Dataset sampler to avoid memory leakage and enable resumption of
# training from the same sample regardless of if we stop mid-epoch
class MultiEpochSampler(torch.utils.data.Sampler):
  """Samples elements randomly over multiple epochs

  Arguments:
      data_source (Dataset): dataset to sample from
      num_epochs (int) : Number of times to loop over the dataset
      start_itr (int) : which iteration to begin from
  """

  def __init__(self, data_source, num_epochs, start_itr=0, batch_size=128):
    self.data_source = data_source
    self.num_samples = len(self.data_source)
    self.num_epochs = num_epochs
    self.start_itr = start_itr
    self.batch_size = batch_size

    if not isinstance(self.num_samples, int) or self.num_samples <= 0:
      raise ValueError("num_samples should be a positive integeral "
                       "value, but got num_samples={}".format(self.num_samples))

  def __iter__(self):
    n = len(self.data_source)
    # Determine number of epochs
    num_epochs = int(np.ceil((n * self.num_epochs 
                              - (self.start_itr * self.batch_size)) / float(n)))
    
    out = [torch.randperm(n) for epoch in range(self.num_epochs)][-num_epochs:]
    # Ignore the first start_itr % n indices of the first epoch
    out[0] = out[0][(self.start_itr * self.batch_size % n):]
    output = torch.cat(out).tolist()
    return iter(output)

  def __len__(self):
    return len(self.data_source) * self.num_epochs - self.start_itr * self.batch_size



# DataLoaderの作成関数
# =====================================================================
def getDataLoader(dataset, data_root=None, data_resize=False, batch_size=64,
                  num_workers=8, shuffle=True, pin_memory=True, drop_last=True,
                  num_epochs=500, start_itr=0, use_multiepoch_sampler=False,
                  sample=False, sample_type='normal', **kwargs):
  
  # データパスの作成
  if sample:
    data_root = os.path.join(data_root, sample_root_dict[dataset], sample_type)
    dataset_fn = sample_dset_dict[dataset]
  else:
    data_root = os.path.join(data_root, root_dict[dataset])
    dataset_fn = dset_dict[dataset]
  show_message('Using dataset root location {}'.format(data_root))

  norm_mean = 0.5
  norm_std = 0.5
  image_size = imsize_dict[dataset]

  # Transform 設定
  # 正規化を実施 ⇒ 以下はDatasets作成時にDICOM画像に対して適用 ⇒ 数値範囲：[0 1]
  # ・負の値の削除
  # ・CT値の最大値4095で除算
  train_transforms = []
  if data_resize:
    train_transforms = [T.Resize(image_size)]
  train_transforms = T.Compose(train_transforms + 
                               [T.ToTensor(),
                                T.Normalize(norm_mean, norm_std)])
  
  # Datasetの作成
  train_dataset = dataset_fn(root=data_root, transform=train_transforms, **kwargs)

  # Dataloaderの作成
  if use_multiepoch_sampler:
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    sampler = MultiEpochSampler(train_dataset, num_epochs, start_itr, batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=sampler, **loader_kwargs)
  else:
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                    'drop_last': drop_last}
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, **loader_kwargs)

  return train_dataloader



# 学習状況の詳細を表示する関数
# =====================================================================
def progress(items, desc='', total=None, min_delay=0.1, displaytype='s1k'):
  """
  Returns a generator over `items`, printing the number and percentage of
  items processed and the estimated remaining processing time before yielding
  the next item. `total` gives the total number of items (required if `items`
  has no length), and `min_delay` gives the minimum time in seconds between
  subsequent prints. `desc` gives an optional prefix text (end with a space).
  """
  total = total or len(items)
  t_start = time.time()
  t_last = 0
  for n, item in enumerate(items):
    t_now = time.time()
    if t_now - t_last > min_delay:
      print("\r%s%d/%d (%6.2f%%)" % (
              desc, n+1, total, n / float(total) * 100), end=" ")
      if n > 0:
        
        if displaytype == 's1k': # minutes/seconds for 1000 iters
          next_1000 = n + (1000 - n%1000)
          t_done = t_now - t_start
          t_1k = t_done / n * next_1000
          outlist = list(divmod(t_done, 60)) + list(divmod(t_1k - t_done, 60))
          print("(TE/ET1k: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
        else:# displaytype == 'eta':
          t_done = t_now - t_start
          t_total = t_done / n * total
          outlist = list(divmod(t_done, 60)) + list(divmod(t_total - t_done, 60))
          print("(TE/ETA: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
          
      sys.stdout.flush()
      t_last = t_now
    yield item
  t_total = time.time() - t_start
  print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))
  


# 学習過程を記録する関数
# 説明：各ネットワークの重み、生成画像を随時保存
# =====================================================================
def save_progress(G, D, GD, G_ema, normal_loader, abnormal_loader, loss_log, state_dict, config):
  # 重みの保存
  # save_weights(G, D, G_ema, state_dict, config)

  # 画像の保存
  if config['ema'] and config['use_ema']:
    which_G = G_ema
  else:
    which_G = G

  save_imgs(which_G, normal_loader, state_dict, config, 'normal')
  save_imgs(which_G, abnormal_loader, state_dict, config, 'abnormal')

  # 学習曲線の保存
  save_trainCurve(loss_log, state_dict, config)


# 重みの保存
# -------------------------------------------------------------
def save_weights(G, D, G_ema, state_dict, config):
  # 保存ファルダの作成
  path = os.path.join(config['save_root'], config['save_date'],
                      config['weights_folder'], 'iter_%d' % (state_dict['itr']))
  if not os.path.exists(path):
    os.makedirs(path)
  # 重みの保存
  torch.save(G.Enc.state_dict(), '%s/%s.pth' % (path, 'G_enc'))
  torch.save(G.Dec.state_dict(), '%s/%s.pth' % (path, 'G_dec'))
  torch.save(G_ema.Enc.state_dict(), '%s/%s.pth' % (path, 'G_ema_enc'))
  torch.save(G_ema.Dec.state_dict(), '%s/%s.pth' % (path, 'G_ema_dec'))
  torch.save(D.Enc.state_dict(), '%s/%s.pth' % (path, 'D_enc'))


# 画像の保存
# -------------------------------------------------------------
def save_imgs(G, loader, state_dict, config, type):
  # パス設定
  root = os.path.join(config['save_root'], config['save_date'], config['images_folder'])
  iter_str = 'iter_%d' % (state_dict['itr'])
  
  # 画像1枚ずつの保存
  last_patient = ''
  for img_real, class_label, patient in loader:

    # 現在の患者設定
    cur_patient = patient[0]

    if cur_patient != last_patient:
      idx = 0

    if idx == 0:
      # 保存フォルダの作成
      path_real = os.path.join(root, iter_str, type, cur_patient, 'real')
      path_fake = os.path.join(root, iter_str, type, cur_patient, 'fake')
      make_folder(path_real)
      make_folder(path_fake)

    img_real, class_label = img_real.to(config['device']), class_label.to(config['device'])
    img_fake = G(img_real, class_label, train_G=False)
    
    # GPUから抽出
    img_real = img_real.detach().clone().cpu()
    img_fake = img_fake.detach().clone().cpu()

    # 0 ~ 1に正規化
    img_real = normalize_tensor(img_real)
    img_fake = normalize_tensor(img_fake)

    # 保存
    torchvision.utils.save_image(img_real, '%s/img_%03d.png' % (path_real, idx))
    torchvision.utils.save_image(img_fake, '%s/img_%03d.png' % (path_fake, idx))

    # 患者の更新
    last_patient = cur_patient
    idx += 1


# 学習曲線の保存
# -------------------------------------------------------------
def save_trainCurve(loss_log, state_dict, config):
   # 保存ファルダの作成
  path_folder = os.path.join(config['save_root'], config['save_date'],
                      config['trainCurve_folder'], 'iter_%d' % (state_dict['itr']))
  if not os.path.exists(path_folder):
    os.makedirs(path_folder)

  t = [i for i in range(state_dict['itr'])]

  # Figure作成
  fig = plt.figure(figsize=(10, 7))
  fig.subplots_adjust(wspace=0.4, hspace=0.6)

  # Contextual Loss of Generator
  ax1 = fig.add_subplot(2, 2, 1)
  ax1.plot(t, loss_log['G_log_con'])
  ax1.set_title('Contextual Loss of Generator')
  ax1.set_xlabel('Iteration')
  ax1.set_ylabel('Loss')
  ax1.grid()
  
  # Adversarial Loss of Generator
  ax2 = fig.add_subplot(2, 2, 2)
  ax2.plot(t, loss_log['G_log_adv'])
  ax2.set_title('Adversarial Loss of Generator')
  ax2.set_xlabel('Iteration')
  ax2.set_ylabel('Loss')
  ax2.grid()

  # Real Prediction Loss of Discriminator
  ax3 = fig.add_subplot(2, 2, 3)
  ax3.plot(t, loss_log['D_log_real'])
  ax3.set_title('Real Prediction Loss of Discriminator')
  ax3.set_xlabel('Iteration')
  ax3.set_ylabel('Loss')
  ax3.grid()
  
  # Fake Prediction Loss of Discriminator
  ax4 = fig.add_subplot(2, 2, 4)
  ax4.plot(t, loss_log['D_log_fake'])
  ax4.set_title('Fake Prediction Loss of Discriminator')
  ax4.set_xlabel('Iteration')
  ax4.set_ylabel('Loss')
  ax4.grid()

  fig.savefig(os.path.join(path_folder, 'trainCurve.png'))



# 実験オプションの保存
# -------------------------------------------------------------
def save_option(config):
  root = os.path.join(config['save_root'], config['save_date'], config['option_folder'])
  filename = os.path.join(root, 'option.json')

  with open(filename, 'w', encoding='utf-8', newline='\n') as fp:
    json.dump(config, fp, indent=2)



# EMA学習を行うクラス -> Generatorクラスのラッパー
# 説明：EMA学習により，局所解に収束するのを防ぐ
# =====================================================================
class ema(object):
  def __init__(self, source, target, decay=0.9999, start_itr=0):
    self.source = source
    self.target = target
    self.decay = decay
    self.start_itr = start_itr
    # ネットワークの重みの辞書
    self.sourch_dict = self.source.state_dict()
    self.target_dict = self.target.state_dict()

    with torch.no_grad():
      for key in self.sourch_dict:
        self.target_dict[key].data.copy_(self.sourch_dict[key].data)

  def update(self, itr=None):
    if itr and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      for key in self.sourch_dict:
        self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                         + self.sourch_dict[key].data * (1 - decay))