"""
BigGANomaly
-> BigGAN のプログラムを基にリファクタリング

ファイル説明：メインの学習・検証・評価ファイル
"""
# Import the all modules we need
# ===================================================
# 一般的なモジュール
import subprocess
from tqdm import tqdm
from visdom import Visdom
# numpy
import numpy as np
# 汎用モジュール
import utils
# ネットワーク構築
import models
# 学習関数
import train_fns


# ===================================================
# 実行関数（深層学習の一連の処理を行う）
def run(config):
  
  # 初期化処理
  # ------------------------------------
  config = utils.initialize_all(config)
  vis = Visdom()
  vis.close()


  # DataLoaderの取得
  # ------------------------------------
  batch_size = (config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations'])
  loader = utils.getDataLoader(**{**config, 'batch_size': batch_size})
  sample_normal_loader = utils.getDataLoader(**{**config, 'batch_size': 1,
                                                'sample': True, 'sample_type': 'normal', 'shuffle': False})
  sample_abnormal_loader = utils.getDataLoader(**{**config, 'batch_size': 1,
                                                  'sample': True, 'sample_type': 'abnormal', 'shuffle': False})

  # ネットワークの構築
  # ------------------------------------
  # Generatorの構築
  Gen_encoder = models.Encoder(**{**config, 'out_latent': True, 'output_dim': config['latent_dim']}).to(config['device'])
  Gen_decoder = models.Decoder(**config).to(config['device'])
  Generator = models.Generator(Gen_encoder, Gen_decoder)
  # Discriminatorの構築
  Dis_encoder = models.Encoder(**config).to(config['device'])
  Discriminator = models.Discriminator(Dis_encoder)
  # BigGANomalyの構築
  net = models.BigGANomaly(Generator, Discriminator)
  utils.show_message('Building BigGANomaly has completed!!')
  # utils.test_net(config, net, loader)

  # EMAモデルの構築
  if config['ema']:
    # Generator
    Gen_encoder_ema = models.Encoder(**{**config, 
                                        'out_latent': True, 'output_dim': config['latent_dim'], 
                                        'skip_init': True, 'no_optim': True}).to(config['device'])
    Gen_decoder_ema = models.Decoder(**{**config, 
                                        'skip_init': True, 'no_optim': True}).to(config['device'])
    Generator_ema = models.Generator(Gen_encoder_ema, Gen_decoder_ema)
    # EMAインスタンスの作成
    ema_instance_Gen_encoder = utils.ema(Gen_encoder, Gen_encoder_ema, config['ema_decay'], config['ema_start'])
    ema_instance_Gen_decoder = utils.ema(Gen_decoder, Gen_decoder_ema, config['ema_decay'], config['ema_start'])
  else:
    Gen_encoder_ema, ema_instance_Gen_encoder = None, None
    Gen_decoder_ema, ema_instance_Gen_decoder = None, None

  # 学習状態の初期化
  # ------------------------------------
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
  
  # 学習関数の定義
  # ------------------------------------
  train_fn = train_fns.training_function(Generator, Discriminator, net, 
                                         ema_instance_Gen_encoder, ema_instance_Gen_decoder, 
                                         state_dict, config)

  # 学習ループ
  # ------------------------------------
  utils.show_message('Biginning training')
  # Epochs
  for epoch in range(config['num_epochs']):
    print('Epoch : %d' % (epoch))
    # 詳細表示の設定
    if config['pbar'] == 'tqdm':
      pbar = tqdm(loader)
    elif config['pbar'] == 'mine':
      pbar = utils.progress(loader, displaytype='eta')
    
    # Iterations
    for i, (x, y) in enumerate(pbar):
      # Iteration + 1
      state_dict['itr'] += 1
      # Generator・Discriminatorの学習モード ON
      Generator.Enc.train()
      Generator.Dec.train()
      Discriminator.Enc.train()
      if config['ema']:
        Gen_encoder_ema.train()
        Gen_decoder_ema.train()
      
      # GPUに転送
      x, y = x.to(config['device']), y.to(config['device'])

      # 学習の実行
      metrics = train_fn(x, y)

      # 詳細表示
      if config['pbar'] == 'mine':
        print(', '.join(['itr: %d' % state_dict['itr']] 
                        + ['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]), end=' ')
      
      # Visdomによる学習曲線の表示
      vis.line(X=np.array([state_dict['itr']]), Y=np.array([metrics['G_loss_con']]), 
               win='window_1',
               update='append',
               opts=dict(title='Contextual Loss of Generator', fontsize=8, 
                         xlabel='Iteration', ylabel='Loss'))
      vis.line(X=np.array([state_dict['itr']]), Y=np.array([metrics['G_loss_adv']]),
               win='window_2',
               update='append',
               opts=dict(title='Adversarial Loss of Generator', fontsize=8, 
                         xlabel='Iteration', ylabel='Loss'))
      vis.line(X=np.array([state_dict['itr']]), Y=np.array([metrics['D_loss_real']]),
               win='window_3',
               update='append',
               opts=dict(title='Real prediction Loss of Discriminator', fontsize=8, 
                         xlabel='Iteration', ylabel='Loss'))
      vis.line(X=np.array([state_dict['itr']]), Y=np.array([metrics['D_loss_fake']]),
               win='window_4',
               update='append',
               opts=dict(title='Fake prediction Loss of Discriminator', fontsize=8, 
                         xlabel='Iteration', ylabel='Loss'))

      # 途中経過の保存
      if (state_dict['itr'] % config['save_frequency']) == 0:
        if config['G_eval_mode']:
          print('\nSave training progress in iter : %d' % state_dict['itr'])
          Generator.Enc.eval()
          Generator.Dec.eval()
          if config['ema']:
            Gen_encoder_ema.eval()
            Gen_decoder_ema.eval()
        utils.save_progress(Generator, Discriminator, net, Generator_ema,
                            sample_normal_loader, sample_abnormal_loader,
                            state_dict, config)
    # Epoch + 1
    state_dict['epoch'] += 1


# ===================================================
# main関数
def main():
  # ターミナルの Clear
  subprocess.run('clear')
  # コマンドライン引数を取得
  parser = utils.prepareParser()
  config = vars(parser.parse_args())
  run(config)


# ===================================================
# Pythonスクリプトとして実行された場合に実行
if __name__ == '__main__':
  main()