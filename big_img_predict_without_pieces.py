# -*- coding: utf-8 -*-
# @Time    : 2021/5/8
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : big_img_predict.py
# @Software: PyCharm
import cv2, os, time, paddle, numpy as np
from paddleseg.utils.FilepathFilenameFileext import filepath_filename_fileext
from configs.user_cfg import classes, VOC_COLORMAP
from paddleseg.core import infer
from paddleseg.utils import get_sys_env
from paddleseg import utils
from configs.user_cfg import model_path, cfg_path
from paddleseg.cvlibs import Config
import paddle.nn.functional as F

cc_area_thresh = 300 # 连通域阈值，过滤面积小于该阈值的目标
score_threshold = 0.99 # 置信度阈值，过滤置信度小于该阈值的目标


def get_obj_score(score_map, cc_labels, cc_label_id, class_id):
    '''
    Args:
        score_map:  经过softmax后得到的置信度矩阵
        cc_labels: 带连通标记的图像
        cc_label_id: 连通域标记
        class_id: 目标类别

    Returns:目标置信度

    '''
    score_map = score_map.numpy()
    points = np.where(cc_labels==cc_label_id)
    points_score = score_map[int(class_id)][points]
    obj_score = sum(points_score)/len(points_score)
    return obj_score


def predict(X, model):
    with paddle.no_grad():
        im = paddle.to_tensor(X)
        net_outs = infer.inference(model, im)
    return net_outs


def cal_score_class(net_out):
    score_map = F.softmax(net_out, axis=0)  # 置信度map [c, h, w]
    label_map = paddle.argmax(score_map, axis=0)  #
    return score_map, label_map


def parse_obj_postion_score(score_map, class_map):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_map)  # 连通域分析
    cc_label_classes_map = dict()
    for idx_label in range(nlabels):
        obj_classes = class_map[int(centroids[idx_label][1])][int(centroids[idx_label][0])]
        cc_label_classes_map[idx_label] = obj_classes
    blocks = []
    for k, stat in enumerate(stats):
        temp = dict()
        [x0, y0, width, height, area] = stat
        x0, y0, width, height = int(x0), int(y0), int(width), int(height)  # 便于json格式转化
        # 删除异常
        cond1 = width / height > 10
        cond2 = height / width > 10
        if cond1 or cond2: continue # 根据领域知识 实际中不可能出现满足cond1和cond2目标
        if cc_label_classes_map[k] == 0: continue # 过滤背景
        if area < cc_area_thresh: continue # 过滤连通区域过小目标
        temp["position"] = [{"x": x0, "y": y0}, {"x": x0 + width, "y": y0}, {"x": x0 + width, "y": y0 + height}, {"x": x0, "y": y0 + height}]
        temp["bug"] = classes[cc_label_classes_map[k]]
        temp["obj_score"] = get_obj_score(score_map, labels, k, cc_label_classes_map[k])
        if temp["obj_score"] <= score_threshold: continue
        blocks.append(temp)
    return blocks


def draw_obj_info(img, blocks, out_img_path):
    font = cv2.FONT_HERSHEY_COMPLEX
    for block in blocks:
        position = block['position']
        bug = block['bug']
        num = classes.index(bug)
        x1, y1 = position[0]['x'], position[0]['y']
        x2, y2 = position[2]['x'], position[2]['y']
        color = VOC_COLORMAP[num]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 标注文本
        cv2.putText(img, bug, (x1, y1), font, 0.4, color, 1)
        cv2.imwrite(out_img_path, img)


# 模型加载
env_info = get_sys_env()
place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else 'cpu'
paddle.set_device(place)
cfg = Config(cfg_path)
model = cfg.model
utils.utils.load_entire_model(model, model_path)
model.eval()


def main(img_path, out_img_path):
    image = cv2.imread(img_path)
    image = np.transpose(image, (2, 0, 1)) # [h, w, c] -> [c, h, w]
    image = image[np.newaxis, :] # [c, h, w] -> [n, c , h, w]
    t1 = time.time()
    print('------input model tensor size, %s ------'% str(image.shape))
    net_outs = predict(image.astype('float32'), model) # 模型计算
    t2 = time.time()
    print('------Cal predict timecost: %s s------' % (round(t2-t1, 3)))
    net_out = paddle.squeeze(net_outs) # [n, c, h, w] -> [c, h, w], squeeze去掉size为1的维度
    score_map, class_map = cal_score_class(net_out) # 置信度计算
    t3 = time.time()
    print('------Cal score_class timecost: %s s------'%str(round(t3-t2,2)))
    class_map = class_map.numpy().astype(np.int8)
    blocks = parse_obj_postion_score(score_map, class_map)
    t4 = time.time()
    print('------Cal parse obj postion and score timecost: %s s------' % str(round(t4 - t3, 2)))
    print('计算各阶段耗时分析：', round(t2-t1, 3), round(t3-t2, 3), round(t4-t3, 3))
    # 画图
    filepath, shotname, extension = filepath_filename_fileext(img_path)
    out_img_path = os.path.join(out_img_path, shotname + extension)
    draw_obj_info(cv2.imread(img_path), blocks, out_img_path)
    return blocks


if __name__ == '__main__':
    img_path = 'docs/images/440.jpg'
    out_img_path = 'output/img_test_result'
    main(img_path, out_img_path)



