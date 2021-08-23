# -*- coding: utf-8 -*-
# @Time    : 2020/12/25
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : cut_large_into_small_pieces.py
# @Software: PyCharm
import numpy as np


def segment(number, space):
    gap_list = []
    times = number // space
    for t in range(times+1):
        gap_list.append(t*space)
    return gap_list

def cut_img_into_pieces(feature, crop_size):
    height, width, channel = feature.shape
    print('width, height, channel',width, height, channel)
    crop_h, crop_w = crop_size

    feature = feature.transpose((2, 0, 1))
    feature = np.expand_dims(feature, axis=0)
    w_remainder = width%crop_w
    h_remainder = height%crop_h
    delta_w = (crop_size[1] - w_remainder) if w_remainder!=0 else 0
    delta_h = (crop_size[0] - h_remainder) if h_remainder!=0 else 0
    # feature shape 为整数倍数
    feature = np.pad(feature, mode="edge", pad_width=((0, 0), (0, 0), (0, delta_h), (0, delta_w)))
    # 更新shape
    batch, channel, height, width = feature.shape
    h_interval_list = segment(height, crop_h)
    w_interval_list = segment(width, crop_w)
    h_interval_idw_list = [int((i+j)/2) for i,j in zip(h_interval_list,h_interval_list[1:])]
    w_interval_idw_list = [int((i+j)/2) for i,j in zip(w_interval_list,w_interval_list[1:])]
    # 获取图片pieces
    img_pieces, idw_img_pieces = [], []
    coordinates, coordinate1, coordinate2 = [], [], []
    for i, h_interval in enumerate(h_interval_list):
        if i==len(h_interval_list)-1: continue
        for j, w_interval in enumerate(w_interval_list):
            if j==len(w_interval_list)-1: continue
            piece = feature[:,:,h_interval_list[i]:h_interval_list[i+1],w_interval_list[j]:w_interval_list[j+1]]
            coordinate1.append([w_interval_list[j], h_interval_list[i]])
            img_pieces.append(piece)
    for i, h_interval in enumerate(h_interval_idw_list):
        if i==len(h_interval_idw_list)-1: continue
        for j, w_interval in enumerate(w_interval_idw_list):
            if j==len(w_interval_idw_list)-1: continue
            piece = feature[:,:,h_interval_idw_list[i]:h_interval_idw_list[i+1],w_interval_idw_list[j]:w_interval_idw_list[j+1]]
            coordinate2.append([w_interval_idw_list[j], h_interval_idw_list[i]])
            idw_img_pieces.append(piece)
    # 拼接img_pieces和idw_img_pieces
    img_pieces = img_pieces + idw_img_pieces
    coordinates = coordinate1 + coordinate2
    out = np.concatenate(img_pieces, axis=0)
    return out, coordinates
