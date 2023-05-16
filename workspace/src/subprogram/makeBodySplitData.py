import os
import subprocess
import fcns

### main
def main(config):
  path_root = config['load_root']
  list_dir = os.listdir(path_root)

  for dir in list_dir:
      path_folder = os.path.join(path_root, dir)
      list_file = os.listdir(path_folder)
      
      # 保存用フォルダの作成
      save_patient_folder = os.path.join(config['save_root'], dir)
      fcns.makeSaveFolder(config['num_section'], save_patient_folder)

      # 体軸方向を4分割し，各領域毎にフォルダを作成，画像を各フォルダに保存
      print('Target Folder : {}'.format(dir))
      fcns.splitAndsaveImage(path_folder, list_file, config['num_section'], save_patient_folder)

### Python scripts
# Description : Pythonスクリプトとして実行しているか判定
if __name__ == '__main__':
    subprocess.run('clear')
    parser = fcns.prepare_parser()
    config = vars(parser.parse_args())
    main(config)