"""
Datasets module

ファイル説明：データセット作成に関する関数を定義
"""
import os
import os.path
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data
from pydicom import dcmread


IMG_EXTENSIONS = ['.jpg', '.png', '.bmp', '.dcm']

# =====================================================================
# 補助関数
# -------------------------------------------------------------------
# Description：DICOM画像の読み込み
def dicomLoader(path):
  # DICOM画像の読み込み
  d = dcmread(path)
  img = d.pixel_array
  # Pixel値に変換
  if d.RescaleIntercept == -1024:
    img = img
  elif d.RescaleSlope == 0:
    img = img + 1024
  # 正規化
  img[img < 0] = 0
  img = img / 4095
  # PIL image フォーマットに変更：torchvision.transformsに対応
  img = Image.fromarray(img)
  return img

# -------------------------------------------------------------------
# Description：各クラスにIDを付与した辞書を作成
def makeClassDict(root_dir):
  first_patientID = os.listdir(root_dir)[0]
  path = os.path.join(root_dir, first_patientID)
  classes = [section for section in os.listdir(path) if os.path.isdir(os.path.join(path, section))]
  classes.sort()
  class_dict = {classes[i]: i for i in range(len(classes))}
  return classes, class_dict
  
# -------------------------------------------------------------------
# Description：画像とクラスのタプルを作成
def makeDataset(root_dir, class_dict):
  images = []
  root_dir = os.path.expanduser(root_dir)
  for patient_dir in sorted(os.listdir(root_dir)):
    patient_path = os.path.join(root_dir, patient_dir)
    if not os.path.isdir(patient_path):
      continue

    for section_dir in sorted(os.listdir(patient_path)):
      section_path = os.path.join(root_dir, patient_dir, section_dir)
      for fname in sorted(os.listdir(section_path)):
        if isImageFile(fname):
          path = os.path.join(root_dir, patient_dir, section_dir, fname)
          item = (path, class_dict[section_dir])
          images.append(item)
  
  return images

# -------------------------------------------------------------------
# Description：画像とクラスのタプルを作成 -> 生成画像の確認用サンプルとして使用
#              返り値にPatientフォルダの名前を追加
def makeSampleDataset(root_dir, class_dict):
  images = []
  root_dir = os.path.expanduser(root_dir)
  for patient_dir in sorted(os.listdir(root_dir)):
    patient_path = os.path.join(root_dir, patient_dir)
    if not os.path.isdir(patient_path):
      continue

    for section_dir in sorted(os.listdir(patient_path)):
      section_path = os.path.join(root_dir, patient_dir, section_dir)
      for fname in sorted(os.listdir(section_path)):
        if isImageFile(fname):
          path = os.path.join(root_dir, patient_dir, section_dir, fname)
          item = (path, class_dict[section_dir], patient_dir)
          images.append(item)
  
  return images

# -------------------------------------------------------------------
# Description：画像ファイルの拡張子判定
def isImageFile(name):
  name_lower = name.lower()
  return any(name_lower.endswith(ext) for ext in IMG_EXTENSIONS)



# =====================================================================
# DICOM画像のDataset作成関数
class DicomDatasets(data.Dataset):
  """
  DicomDatasetsクラス（自作Datasetsクラス）
  
  想定されているフォルダ構成：
  root/patientID/section0/IMG001.dcm
  root/patientID/section0/IMG002.dcm
  root/patientID/section0/IMG003.dcm
  ...
  root/patientID/section1/IMG001.dcm
  root/patientID/section1/IMG002.dcm
  root/patientID/section1/IMG003.dcm
  ...

  引数：
  root (string): 学習データのルートパス（example: data/ct-split/）
  transform (callable, optional): 学習データのTransform
  loader (callable, optional): 画像読み込みの関数

  """
  def __init__(self, root, transform=None, loader=dicomLoader, **kwargs):
    # パスとクラスのセットを作成
    classes, class_dict = makeClassDict(root)
    imgs = makeDataset(root, class_dict)

    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                         "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    # インスタンス初期化
    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_dict = class_dict
    self.transform = transform
    self.loader = loader

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) DICOM画像とクラスのタプル
    """
    path, target = self.imgs[index]
    img = self.loader(str(path))
    if self.transform is not None:
      img = self.transform(img)

    return img, int(target)

  def __len__(self):
    return len(self.imgs)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '     Number of data: {}\n'.format(self.__len__())
    fmt_str += '     Root Location: {}\n'.format(self.root)
    return fmt_str
  

# =====================================================================
# DICOM画像のDataset作成関数
# Description：生成画像の確認用サンプルとして使用。返り値にPatientフォルダの名前を追加
#              DicomDatasetsクラスとほとんど同じ(返り値の種類が増えるだけ)
class DicomSampleDatasets(DicomDatasets):
  def __init__(self, root, transform=None, loader=dicomLoader, **kwargs):
    super().__init__(root, transform, loader, **kwargs)
    self.imgs = makeSampleDataset(root, self.class_dict)
  
  def __getitem__(self, index):
    path, target, patient_dir = self.imgs[index]
    img = self.loader(str(path))
    if self.transform is not None:
      img = self.transform(img)

    return img, int(target), patient_dir