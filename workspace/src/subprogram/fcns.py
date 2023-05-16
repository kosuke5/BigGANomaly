import os
import math
from PIL import Image
import numpy as np
from pydicom import dcmread
from argparse import ArgumentParser

import torch
import torchvision.transforms as T


### Function
# Description : コマンドライン引数の定義
def prepare_parser():
  usage = 'Parser for scripts called makeBodySplitData.py'
  parser = ArgumentParser(description=usage)
  # Add argument
  parser.add_argument('--load_root', type=str, default='./data/ct')
  parser.add_argument('--save_root', type=str, default='ct-split')
  parser.add_argument('--num_section', type=int, default=4)

  return parser


### Function
# Description : DICOMファイルの読み込みと正規化を行う
def readNormalizedDICOM(path_folder, name_file, changeToTensor=False):
  path = os.path.join(path_folder, name_file)
  d = dcmread(path)
  instanceNumber = d.InstanceNumber
  img = d.pixel_array
  # 正規化
  img[img < 0] = 0
  img = img / 4095
  # PILフォーマットに変更
  img = Image.fromarray(img)

  # Tensorフォーマットに変更
  if changeToTensor:
    transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
    img = transforms(img)
  
  return img, instanceNumber


### Function 
# Description : 体軸方向分割（セクション分割）と画像保存を行う
def splitAndsaveImage(path_folder, list_file, num_section, save_patient_folder):
  for name_file in list_file:
    # DICOMの読み込み
    ds = dcmread(os.path.join(path_folder, name_file))
    instanceNumber = ds.InstanceNumber

    # セクション管理変数の定義
    num_image = len(list_file)
    num_image_per_section = math.ceil(num_image / num_section)

    # 保存先セクションの判定
    section = identifySection(num_section, num_image_per_section, instanceNumber)

    # DICOMの保存
    save_path = os.path.join(
      save_patient_folder, str(section), 'IMG'+format(instanceNumber, '03')+'.dcm')
    ds.save_as(save_path)


### Function
# Description : 保存先とする体軸方向に分割したセクションの判定
def identifySection(num_section, num_image_per_section, instanceNumber):
  for section in range(num_section):
    start = num_image_per_section * section + 1   #InstanceNumberが 1 から始まるため
    stop = start + num_image_per_section
    if start <= instanceNumber and instanceNumber < stop:
      return section


### Function
# Description : 患者毎にセクションに分割し保存先フォルダを作成
def makeSaveFolder(num_section, patient_folder):
  for section in range(num_section):
    path_folder = os.path.join(patient_folder, str(section))
    if not os.path.exists(path_folder):
      os.makedirs(path_folder)
