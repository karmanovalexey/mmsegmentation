import os
import numpy as np

from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser

PALETTE = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
            [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
            [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
            [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
            [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
            [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
            [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
            [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
            [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
            [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
            [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
            [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
            [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
            [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90],
            [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70],
            [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]]

def main(args):
    assert os.path.exists(args.data_dir), "No dataset found"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    os.chmod(args.save_dir, 0o777)
    
    if args.dataset=='Mapillary':
        filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.data_dir)) for f in fn if f.endswith(".jpg")]
    elif args.dataset=='Kitti':
        filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.data_dir)) for f in fn if f.endswith(".png")]

    print(len(filenames))
    filenames.sort()

    for filename in tqdm(filenames):
        with open(filename, 'rb') as f:
            image = np.array(Image.open(f).convert('RGB'))
        
        arr_3d = np.array(image)
        height = arr_3d.shape[0]
        width = arr_3d.shape[1]
        arr_2d = np.zeros((height, width), dtype=np.uint8)

        for i, pal in enumerate(PALETTE):
            m = np.all(arr_3d == np.array(pal).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i

        save_img = Image.fromarray(arr_2d)
        if args.dataset=='Mapillary':
            save_point = args.save_dir + '/' + filename[37:-4] + '.png'
        elif args.dataset=='Kitti':
            save_point = args.save_dir + '/' + filename[17:-4] + '.png'
        save_img.save(save_point, 'PNG')
        os.chmod(save_point, 0o777)

    return 
    



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['Mapillary', 'Kitti'], help='What Dataset to convert')
    parser.add_argument('--save-dir', help='Where to save new files')
    parser.add_argument('--data-dir', help='Dataset path')
    main(parser.parse_args())
