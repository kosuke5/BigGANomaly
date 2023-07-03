"""
Models module

ファイル説明：プロジェクトで使用する深層学習モデルを定義
"""
import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

import utils
import layers

"""
Decoder Network
=========================
説明 : Decoderを定義 ⇒ Decoderは画像生成の際に使用
"""
# Decoderのアーキテクチャ定義
# =================================================================
def Dec_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[512] = {
      'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
      'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
      'upsample': [True] * 7,
      'resolution': [8, 16, 32, 64, 128, 256, 512],
      'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                    for i in range(3, 10)}
  }
  arch[256] = {
      'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2]],
      'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
      'upsample': [True] * 6,
      'resolution': [8, 16, 32, 64, 128, 256],
      'attention': {2**i: (2**i in [int(item) for item in attention.split('_')])
                    for i in range(3, 9)}
  }
  return arch

# Decoderクラス
# =================================================================
class Decoder(nn.Module):
  def __init__(self, Dec_ch=64, latent_dim=128, initial_width=4, resolution=128,
               Dec_kernel_size=3, Dec_attn='64', n_classes=1000,
               num_Dec_SVs=1, num_Dec_SV_itrs=1, Dec_ccbn_norm=False,
               Dec_shared=True, shared_dim=0, hier=False,
               mybn=False, Dec_activation=nn.ReLU(inplace=False),
               Dec_lr=5e-5, Dec_B1=0.0, Dec_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, Dec_init='ortho', skip_init=False, 
               no_optim=False, Dec_param='SN', norm_style='bn',
               **kwargs):
    super(Decoder, self).__init__()

    # 各種パラメータ設定
    # --------------------------------------------------------
    # 最初のチャネル数
    self.ch = Dec_ch
    # 潜在変数（入力）の次元数
    self.latent_dim = latent_dim
    # 最初のアップサンプリング層に入力する画像サイズ（幅）
    self.initial_width = initial_width
    # 生成する画像サイズ
    self.resolution = resolution
    # カーネルサイズ
    self.kernel_size = Dec_kernel_size
    # Slef Attention層を入れる解像度
    self.attention = Dec_attn
    # クラス数 ⇒ クラス情報の付与に使用
    self.n_classes = n_classes
    # TODO: 仕様解明（Use shared embeddings?）
    self.Dec_shared = Dec_shared
    # TODO: 仕様解明（Dimensionality of the shared embedding? Unused if not using Dec_shared）
    self.shared_dim = shared_dim if shared_dim > 0 else latent_dim
    # 階層型の潜在変数を使用？ ⇒ 潜在変数を分割し、各解像度の畳み込み層に使用
    self.hier = hier
    # TODO: 仕様解明（Use my batchnorm?）
    self.mybn = mybn
    # 残差ブロックで使用する活性化関数
    self.activation = Dec_activation
    # 重みの初期化方法
    self.init = Dec_init
    # 重みの正規化手法 ⇒ Spectral Normalizationなどの指定
    self.Dec_param = Dec_param
    # 正規化層の種類（選択肢：Batch Norm, Instance Norm, Layer Norm, Group Norm）
    self.norm_style = norm_style
    # 潜在変数 or クラスベクトルを正規化するか指定
    self.Dec_ccbn_norm = Dec_ccbn_norm
    # バッチ正規化層で使用するEpsilon
    self.BN_eps = BN_eps
    # スペクトラル正規化で使用するEpsilon
    self.SN_eps = SN_eps
    # 初期学習率
    self.lr = Dec_lr
    # Adamのベータ1・ベータ2
    self.B1, self.B2 = Dec_B1, Dec_B2
    # Adamのイプシロン
    self.adam_eps = adam_eps
    # Decoderのアーキテクチャ
    self.arch = Dec_arch(self.ch, self.attention)[resolution]

    # 潜在変数の設定（階層型の潜在変数）
    # --------------------------------------------------------
    if self.hier:
      # 階層の数
      self.num_chunks = len(self.arch['in_channels']) + 1     
      # 各階層に含まれる潜在変数の次元数        
      self.latent_per_chunk = (self.latent_dim // self.num_chunks)  
      # 割り切れる次元数に再設定
      self.latent_dim = self.latent_per_chunk * self.num_chunks
    else:
      self.num_chunks = 1
      self.latent_per_chunk = 0

    # Convolution, Linear層の定義
    # --------------------------------------------------------
    if self.Dec_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_Dec_SVs, num_itrs=num_Dec_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                          num_svs=num_Dec_SVs, num_itrs=num_Dec_SV_itrs,
                          eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear

    # BatchNormalization層の定義
    # --------------------------------------------------------
    self.embedding_layer = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.Dec_shared
                 else self.embedding_layer)
    self.which_bn = functools.partial(layers.ccbn,
                          linear_layer=bn_linear,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.latent_per_chunk if self.Dec_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)
    
    # Decoder モデル構築
    # --------------------------------------------------------
    # 埋め込み層の定義 ⇒ 潜在変数やクラス情報の埋め込みに使用（共有埋め込みを使用？）
    self.shared = (self.embedding_layer(n_classes, self.shared_dim) if Dec_shared
                   else layers.identity())
    # 最初の全結合層
    self.first_linear = self.which_linear(self.latent_dim // self.num_chunks,
                                          self.arch['in_channels'][0] * (self.initial_width ** 2))

    # ResBlockの構築
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DecBlock(in_channels=self.arch['in_channels'][index],
                                  out_channels=self.arch['out_channels'][index],
                                  conv_layer=self.which_conv,
                                  bn_layer=self.which_bn,
                                  activation=self.activation,
                                  upsample=(functools.partial(F.interpolate, scale_factor=2)
                                            if self.arch['upsample'][index] else None) )]]
      
      # Attention機構の導入
      if self.arch['attention'][self.arch['resolution'][index]]:
        utils.show_message('Adding attention layer in Decoder as resolution {}'
              .format(self.arch['resolution'][index]))
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # nn.ModuleListに変換 ⇒ 層のイテレータを作成可能となる
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # 出力層：BatchNorm ⇒ ReLU ⇒ Conv
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                mybn=self.mybn),
                                      self.activation,
                                      self.which_conv(self.arch['out_channels'][-1], 1))

    # 各パラメータの初期化（評価の際はスキップ）
    # --------------------------------------------------------
    if not skip_init:
      self.init_weights()

    # オプティマイザーの設定
    # --------------------------------------------------------
    if no_optim:
      return
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                              betas=(self.B1, self.B2), weight_decay=0,
                              eps=self.adam_eps)

  # パラメータ初期化メソッド
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          utils.show_message('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    utils.show_message('Param count for G''s initialized parameters: %d' % self.param_count)
    
  # Forwardメソッド
  def forward(self, z, y):
    # Latent variables, Class vectorの正規化
    if self.Dec_ccbn_norm:
      mins_z, mins_y = utils.tensor_min(z, 1), utils.tensor_min(y, 1)
      maxs_z, maxs_y = utils.tensor_max(z, 1), utils.tensor_max(y, 1)
      for i in range(z.shape[0]):
        z[i] = (z[i] - mins_z[i]) / (maxs_z[i] - mins_z[i])
        y[i] = (y[i] - mins_y[i]) / (maxs_y[i] - mins_y[i])

    if self.hier:
      zs = torch.split(z, self.latent_per_chunk, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)

    # First linear and reshape
    h = self.first_linear(z)
    h = h.view(h.size(0), -1, self.initial_width, self.initial_width)

    # ResBlock
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h, ys[index])

    return torch.tanh(self.output_layer(h))



"""
Encoder Network
=========================
説明 : Encoder ⇒ Encoderは画像圧縮に使用
"""
# Encoderのアーキテクチャ定義
# =================================================================
def Enc_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [1] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [1] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  return arch

# Encoderクラス
# =================================================================
class Encoder(nn.Module):
  def __init__(self, Enc_ch=64, Enc_wide=True, resolution=128,
               Enc_kernel_size=3, Enc_attn='64', n_classes=200,
               num_Enc_SVs=1, num_Enc_SV_itrs=1, Enc_ccbn_norm=False,
               Enc_activation=nn.ReLU(inplace=False),
               Enc_lr=2e-4, Enc_B1=0.0, Enc_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, Enc_init='ortho', skip_init=False, 
               no_optim=False, Enc_param='SN', out_latent=False,
               **kwargs):
    super(Encoder, self).__init__()

    # 各種パラメータ設定
    # --------------------------------------------------------
    # 最初のチャネル数
    self.ch = Enc_ch
    # TODO: Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.Enc_wide = Enc_wide
    # 生成する画像サイズ
    self.resolution = resolution
    # カーネルサイズ
    self.kernel_size = Enc_kernel_size
    # Slef Attention層を入れる解像度
    self.attention = Enc_attn
    # クラス数 ⇒ クラス情報の付与に使用
    self.n_classes = n_classes
    # 残差ブロックで使用する活性化関数
    self.activation = Enc_activation
    # 重みの初期化方法
    self.init = Enc_init
    # 重みの正規化手法 ⇒ Spectral Normalizationなどの指定
    self.Enc_param = Enc_param
    # クラスベクトルの正規化をするか指定
    self.Enc_ccbn_norm = Enc_ccbn_norm
    # スペクトラル正規化で使用するEpsilon
    self.SN_eps = SN_eps
    # 初期学習率
    self.lr = Enc_lr
    # Adamのベータ1・ベータ2
    self.B1, self.B2 = Enc_B1, Enc_B2
    # Adamのイプシロン
    self.adam_eps = adam_eps
    # 潜在変数として出力（Generatorで使用)
    self.out_latent = out_latent
    # Encoderのアーキテクチャ
    self.arch = Enc_arch(self.ch, self.attention)[resolution]

    # Convolution, Linear, Embedding層の定義
    # --------------------------------------------------------
    if self.Enc_param == 'SN':
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1,
                          num_svs=num_Enc_SVs, num_itrs=num_Enc_SV_itrs,
                          eps=self.SN_eps)
      self.which_linear = functools.partial(layers.SNLinear,
                            num_svs=num_Enc_SVs, num_itrs=num_Enc_SV_itrs,
                            eps=self.SN_eps)
      self.which_embedding = functools.partial(layers.SNEmbedding,
                                num_svs=num_Enc_SVs, num_itrs=num_Enc_SV_itrs,
                                eps=self.SN_eps)
      
    # Encoder モデル構築
    # --------------------------------------------------------
    # ResBlockの構築
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.EncBlock(in_channels=self.arch['in_channels'][index],
                                       out_channels=self.arch['out_channels'][index],
                                       conv_layer=self.which_conv,
                                       wide=self.Enc_wide,
                                       activation=self.activation,
                                       preactivation=(index > 0),
                                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # Attention機構の導入
      if self.arch['attention'][self.arch['resolution'][index]]:
        utils.show_message('Adding attention layer in Decoder as resolution {}'
              .format(self.arch['resolution'][index]))
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
      
    # nn.ModuleListに変換 ⇒ 層のイテレータを作成可能となる
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # 出力層：Linear
    self.linear_layer = self.which_linear(self.arch['out_channels'][-1], output_dim)
    # 出力層：Embedding
    self.embed_layer = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

    # 各パラメータの初期化（評価の際はスキップ）
    # --------------------------------------------------------
    if not skip_init:
      self.init_weights()
    
    # オプティマイザーの設定
    # --------------------------------------------------------
    if no_optim:
      return
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                              betas=(self.B1, self.B2), weight_decay=0,
                              eps=self.adam_eps)
  
  # パラメータ初期化メソッド
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          utils.show_message('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    utils.show_message('Param count for G''s initialized parameters: %d' % self.param_count)

  # Forwardメソッド
  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    h = x
    # Loop over blocks
    for _, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])
    # Get initial class-unconditional output
    out = self.linear_layer(h)
    # Get projection of final featureset onto class vectors and add to evidence
    class_vector = self.embed_layer(y)
    
    # クラスベクトルの正規化
    if self.Enc_ccbn_norm:
      mins_class, maxs_class = utils.tensor_min(class_vector, 1), utils.tensor_max(class_vector, 1)
      for i in range(class_vector.shape[0]):
        class_vector[i] = (class_vector[i] - mins_class[i]) / (maxs_class[i] - mins_class[i])

    if self.out_latent:
      out = out + self.linear_layer(class_vector * h)
    else:
      out = out + torch.sum(class_vector * h, 1, keepdim=True)
    return out



"""
Generator Network
=========================
説明 : Generatorを定義
"""
class Generator(nn.Module):
  def __init__(self, Enc, Dec):
    super(Generator, self).__init__()
    self.Enc = Enc
    self.Dec = Dec

  def forward(self, x, y, train_G=False):
    with torch.set_grad_enabled(train_G):
      # 潜在変数の抽出
      z = self.Enc(x, y)
      # 画像復元
      img_fake = self.Dec(z, self.Dec.shared(y))

      return img_fake


"""
Discriminator Network
=========================
説明 : Discriminatorを定義
"""
class Discriminator(nn.Module):
  def __init__(self, Enc):
    super(Discriminator, self).__init__()
    self.Enc = Enc

  def forward(self, x, y, train_D=False):
    with torch.set_grad_enabled(train_D):
      pred = self.Enc(x, y)
      return pred
    

"""
BigGANomaly Network
=========================
説明 : BigGANomalyを定義
"""
class BigGANomaly(nn.Module):
  def __init__(self, Gen, Dis):
    super().__init__()
    self.Gen = Gen
    self.Dis = Dis
  
  def forward(self, imgs_real, class_labels, train_G=False, train_D=False, 
              get_only_imgs=False, get_only_preds=False):
    # Generatorによる画像復元
    imgs_fake = self.Gen(imgs_real, class_labels, train_G)

    if get_only_imgs:
      return imgs_fake
    
    # Discriminatorによる判定
    pred_real = self.Dis(imgs_real, class_labels, train_D)
    pred_fake = self.Dis(imgs_fake, class_labels, train_D)

    if get_only_preds:
      return pred_real, pred_fake

    return imgs_fake, pred_fake