import mmcv
import os.path as osp
# split train/val set randomly
split_dir = 'splits'
data_root = '/workspace/Mapillary'
ann_dir = '/val/1920_1080/labels'
mmcv.mkdir_or_exist(split_dir)
dir_path = '/workspace/Mapillary/val/1920_1080/labels'
print(dir_path)
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(dir_path, suffix='.png')]
with open(osp.join(split_dir, 'split.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list)*1/4)
  f.writelines(line + '\n' for line in filename_list[:train_length])