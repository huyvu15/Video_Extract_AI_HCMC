import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import json

from utils.query_processing import Translation
from utils.faiss import Myfaiss

with open('image_path.json') as json_file:
    json_dict = json.load(json_file)

DictImagePath = {}
for key, value in json_dict.items():
   DictImagePath[int(key)] = value 

LenDictPath = len(DictImagePath)
bin_file='faiss_normal_ViT.bin'
MyFaiss = Myfaiss(bin_file, DictImagePath, 'cpu', Translation(), "ViT-B/32")

def text_search():
    text_query = input("Nhập text cần tìm kiếm: ")
    _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=50)
    print(list_image_paths)

if __name__ == '__main__':
    
    text_search()
