import os
import sys

# noinspection PyUnresolvedReferences
# from prepare_data import true_wheat_indoor3d_util
from true_wheat_indoor3d_util_normal import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# DATA_PATH = os.path.join(ROOT_DIR, 'data/true_wheat_sample')
DATA_PATH = "D:/a-project-T/Data/true_wheat_data/true_wheat_sample"
# DATA_PATH = "Y:\\LPR_segmentation\\2data\\un_RGB\\true_Wheat_Sample"

path = "D:/a-project-T/Pointnext_1/prepare_data/meta/true_wheat_anno_paths.txt"
anno_paths = [line.rstrip() for line in open(path)]
# anno_paths = "D:/a-project-T/Pointnext_1/prepare_data/meta/wheat_anno_paths.txt"
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

# output_folder = os.path.join(ROOT_DIR, 'data/true_wheat_npy')
output_folder = "C:/Users/tang/Desktop/true_wheat_npy"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# revise_file = os.path.join(DATA_PATH, "Sample_1/Run1_WT1_11/Annotations/leaf_1.txt")
# with open(revise_file, "r") as f:
#     data = f.read()
#     data = data[:5545347] + ' ' + data[5545348:]
#     f.close()
# with open(revise_file, "w") as f:
#     f.write(data)
#     f.close()

for anno_path in anno_paths:
    print(anno_path)
    elements = anno_path.split('/')
    elements_1 = elements[-3].split('\\')
    out_filename = elements_1[-1]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy    Sample_1_Run1_WT1_10.npy
    collect_point_label_1(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    print("*********  完成"+ "   "+ out_filename + " *********")