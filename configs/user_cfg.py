# -*- coding: utf-8 -*-
# @Time    : 2021/5/7
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : user_cfg.py
# @Software: PyCharm

colormap=[[0, 0, 0], [128, 0, 128], [0, 128, 0],[128, 128, 128]]
classes=['background', 'bug_b', 'bug_y_long', 'bug_y_short']
VOC_COLORMAP = [[0, 0, 0], [128, 0, 128], [0, 128, 0], [128, 128, 0]]


# 预测
cfg_path = 'configs/bisenet_optic_disc_512x512_1k.yml'
save_dir = './output/result'
image_path = 'docs/images'
model_path = 'output/iter_10000/model.pdparams'

# pieces size
crop_size = (480, 480)
