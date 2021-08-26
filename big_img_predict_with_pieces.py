# -*- coding: utf-8 -*-
# @Time    : 2021/5/8
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : big_img_predict_with_pieces.py
# @Software: PyCharm
import cv2, os, time, paddle, copy, numpy as np
from paddleseg.utils.FilepathFilenameFileext import filepath_filename_fileext
from paddleseg.utils.cut_large_into_small_pieces import cut_img_into_pieces
from paddleseg.utils import get_sys_env
from paddleseg import utils
from configs.user_cfg import model_path, cfg_path, crop_size
from paddleseg.cvlibs import Config
import paddle.nn.functional as F
from big_img_predict_without_pieces import parse_obj_postion_score, draw_obj_info, predict, normalize


def postprocess(blocks_list, coordinates):
    # 局部坐标转化为全局坐标
    result = []
    for i, blocks in enumerate(blocks_list):
        x_start, y_start = coordinates[i]
        for j, block in enumerate(blocks):
            for k, ele in enumerate(block['position']):
                block['position'][k]['x'] += x_start
                block['position'][k]['y'] += y_start
            result.append(block)
    # 目标去重
    result_copy = copy.deepcopy(result)
    for i, ele1 in enumerate(result_copy):
        pos1 = ele1['position']
        rect1 = [pos1[0]['y'], pos1[0]['x']],[pos1[2]['y'], pos1[2]['x']]
        for j, ele2 in enumerate(result_copy):
            if j<=i: continue
            pos2 = ele2['position']
            rect2 = [pos2[0]['y'], pos2[0]['x']], [pos2[2]['y'], pos2[2]['x']]
            iou = get_iou(rect1, rect2)
            if iou>0.95:
                if ele2 in result:
                    result.remove(ele2)
    return result


def get_iou(points, pos):
    delta_w = min(pos[1][0], points[1][0]) - max(pos[0][0], points[0][0])
    delta_h = min(pos[1][1], points[1][1]) - max(pos[0][1], points[0][1])
    if delta_h < 0 or delta_w < 0: return 0
    denominator = (points[1][0] - points[0][0]) * (points[1][1] - points[0][1])
    if denominator == 0: return 0
    ratio = delta_w * delta_h / denominator
    return ratio


def cal_score_class(net_outs):
    score_maps = F.softmax(net_outs, axis=1)  # 置信度map [n,c, h, w]
    label_maps = paddle.argmax(score_maps, axis=1)  #
    return score_maps, label_maps


# 模型加载
env_info = get_sys_env()
place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else 'cpu'
paddle.set_device(place)
cfg = Config(cfg_path)
val_dataset = cfg.val_dataset
transforms = val_dataset.transforms
model = cfg.model
utils.utils.load_entire_model(model, model_path)
model.eval()


def main(img_path, out_img_path):
    image = cv2.imread(img_path)
    t1 = time.time()
    X, coordinates = cut_img_into_pieces(image, crop_size) # 待优化,耗时0.2s [n, c, h, w]
    print('X', X.shape)
    X, _ = transforms(X)
    print('X',X.shape)
    t2 = time.time()
    print('------Cal cut img timecost: %s s------' % (round(t2 - t1, 3)))
    net_outs = predict(X.astype('float32'), model) # 模型计算 [n, len(class), h, w]
    t3 = time.time()
    print('------Cal predict timecost: %s s------' % (round(t3 - t2, 3)))
    score_maps, class_maps = cal_score_class(net_outs)  # 置信度计算
    class_maps = class_maps.numpy().astype(np.int8)
    t4 = time.time()
    print('------Cal score_class timecost: %s s------' % str(round(t4 - t3, 2)))
    blocks_list = []
    for idx in range(len(class_maps)):
        class_map, score_map = class_maps[idx], score_maps[idx]
        blocks = parse_obj_postion_score(score_map, class_map)
        blocks_list.append(blocks)
    t5 = time.time()
    print('------Cal parse obj postion and score timecost: %s s------' % str(round(t5 - t4, 2)))
    result = postprocess(blocks_list, coordinates) # 映射到实际坐标+目标去重
    filepath, shotname, extension = filepath_filename_fileext(img_path)
    out_img_path = os.path.join(out_img_path, shotname + extension)
    draw_obj_info(cv2.imread(img_path), result, out_img_path)
    t6 = time.time()
    print('------Cal postprocess timecost: %s s------' % str(round(t6 - t5, 2)))
    print('各阶段耗时分析：', round(t2-t1, 3), round(t3-t2, 3), round(t4-t3, 3), round(t5-t4, 3), round(t6-t5, 3))
    return result


if __name__ == '__main__':
    img_path = 'docs/images/440.jpg'
    out_img_path = 'output/img_test_result'
    main(img_path, out_img_path)