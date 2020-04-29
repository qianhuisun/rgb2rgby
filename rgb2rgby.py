import os
import sys
import numpy as np
import cv2


def get_all_paths(folder_path):
    all_file_paths = []

    for root, dirs, files in os.walk(folder_path):

        if len(files) > 0:

            file_paths = []
            
            for name in files:

                if name.find('.png') >= 0 or name.find('.jpg')>=0:

                    file_paths.append(os.path.join(root, name))

            all_file_paths += file_paths
        
    return all_file_paths
    


def rgb2rgby():

    assert len(sys.argv) == 3, "python rgb2rgby.py input/ output/"

    assert os.path.exists(sys.argv[1]), "path does not exist"

    file_paths = get_all_paths(sys.argv[1])

    output_path = sys.argv[2]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx, file_path in enumerate(file_paths, 100):

        rgb = cv2.imread(file_path, -1)[:, :, ::-1].astype(np.float32) # convert BGR to RGB(h,w,c)
        rgb = np.clip(rgb / 255, 1e-8, 1)
        rgb_linear = np.power(rgb, 2.2)
        
        h, w, _ = rgb_linear.shape

        rgby_linear = np.zeros((h, w, 4), dtype = np.float32)

        rgby_linear[::, ::, 0] = rgb_linear[::, ::, 0]
        rgby_linear[::, ::, 1] = rgb_linear[::, ::, 1]
        rgby_linear[::, ::, 2] = rgb_linear[::, ::, 2]
        rgby_linear[::, ::, 3] = 0.65 * rgb_linear[::, ::, 0] + 1.003 * rgb_linear[::, ::, 1] - 0.06 * rgb_linear[::, ::, 2]
        
        rgby_linear = np.clip(rgby_linear, 1e-8 ,1)

        np.save(output_path + "/" + str(idx) + ".png", rgby_linear, allow_pickle=True)


if __name__ == "__main__":
    rgb2rgby()
