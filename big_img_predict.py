# -*- coding: utf-8 -*-
# @Time    : 2021/5/8
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : big_img_predict.py
# @Software: PyCharm
import cv2, os, time, paddle, copy, numpy as np
from paddleseg.utils.FilepathFilenameFileext import filepath_filename_fileext
from paddleseg.utils.cut_large_into_small_pieces import cut_img_into_pieces
from configs.user_cfg import crop_size, classes, VOC_COLORMAP
from paddleseg.core import infer
from paddleseg.utils import get_sys_env
from paddleseg import utils
from configs.user_cfg import model_path, cfg_path
from paddleseg.cvlibs import Config
import paddle.nn.functional as F

def get_iou(points, pos):
    delta_w = min(pos[1][0], points[1][0]) - max(pos[0][0], points[0][0])
    delta_h = min(pos[1][1], points[1][1]) - max(pos[0][1], points[0][1])
    if delta_h < 0 or delta_w < 0: return 0
    denominator = (points[1][0] - points[0][0]) * (points[1][1] - points[0][1])
    if denominator == 0: return 0
    ratio = delta_w * delta_h / denominator
    return ratio

def get_obj_score(softmax_result, obj_bool_pixel):
    filter = obj_bool_pixel.numpy()>=1.0
    results = softmax_result.numpy()[np.where(filter==True)]
    if results.size==0: return 0.0
    try:
        score = results.sum()/results.size
    except:
        score = 0.0
    return score

def object_score_map(net_out, score_threshold):
    out_sigmoid = F.sigmoid(net_out)
    out_softmax = F.softmax(out_sigmoid, axis=0)
    object_score_softmax = copy.deepcopy(out_softmax[3])
    # 将阈值低于score_threshold的置0，反之置为1
    if type(score_threshold) is str: score_threshold = float(score_threshold)
    object_score = object_score_softmax>score_threshold # 元素均为0或者1
    # 获取阈值处理后剩余区域的像素平均置信度，作为目标置信度
    confidence = get_obj_score(object_score_softmax,object_score)
    return object_score, confidence

def predict(X, model):
    with paddle.no_grad():
        im = paddle.to_tensor(X)
        net_outs = infer.inference(model, im)
        net_outs = paddle.squeeze(net_outs)
    return net_outs

def cal_pre_score_conf(net_outs, score_threshold):
        preds, score_maps, confidences = [], [] ,[]
        for net_out in net_outs:
            # 获取目标置信度分布图
            score_map, confidence = object_score_map(net_out, score_threshold)
            pred = paddle.argmax(net_out, axis=0)
            preds.append(pred)
            score_maps.append(score_map)
            confidences.append(confidence)
        return preds, score_maps, confidences


env_info = get_sys_env()
place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else 'cpu'
paddle.set_device(place)
cfg = Config(cfg_path)
model = cfg.model
utils.utils.load_entire_model(model, model_path)
model.eval()


def main(image, score_threshold):
    if type(image) is str: image = cv2.imread(image)
    t1 = time.time()
    X, coordinates = cut_img_into_pieces(image, crop_size) # 待优化,耗时0.2s
    t2 = time.time()
    print('输入tensor: ',X.shape)
    net_outs = predict(X.astype('float32'), model) # 模型计算
    print('out', net_outs.shape)
    # print('输出tensor: ', net_outs[0])
    t3 = time.time()
    outputs, score_maps, confidences = cal_pre_score_conf(net_outs, score_threshold) # 置信度计算
    t4 = time.time()
    blocks_list = []
    for idx in range(len(outputs)):
        output, score_map, confidence = outputs[idx], score_maps[idx], confidences[idx]
        output = output.numpy()
        output_copy = copy.deepcopy(output).astype(np.int8)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(output_copy)
        # 不同联通区域对应的
        cc_label_classes_map = dict()
        for idx_label in range(nlabels):
            obj_classes = output_copy[int(centroids[idx_label][1])][int(centroids[idx_label][0])]
            cc_label_classes_map[idx_label] = obj_classes
        # 过滤连通区域小于300个像素面积区域
        thre_area = 300
        blocks = []
        for k, stat in enumerate(stats):
            temp = dict()
            [x0, y0, width, height, area] = stat
            x0, y0, width, height = int(x0), int(y0), int(width), int(height)  # 便于json格式转化
            # 删除异常
            cond1 = width/height>10
            cond2 = height/width>10
            if cond1 or cond2: continue
            if cc_label_classes_map[k]==0: continue
            if area<thre_area: continue
            temp["position"] = [{"x":x0,"y":y0},{"x":x0+width,"y":y0},{"x":x0+width,"y":y0+height},{"x":x0,"y":y0+height}]
            temp["bug"] = classes[cc_label_classes_map[k]]
            temp["obj_score"] = confidence
            blocks.append(temp)
        blocks_list.append(blocks)
    t5 = time.time()
    print('各阶段耗时分析：',round(t5-t4,3), round(t4-t3,3),round(t3-t2,3),round(t2-t1,3))
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
    # print('result', result)
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


if __name__ == '__main__':
    img_path = 'docs/images/440.jpg'
    img_save_path = 'output/img_test_result'
    img = cv2.imread(img_path)
    filepath, shotname, extension = filepath_filename_fileext(img_path)
    img_save_path = os.path.join(img_save_path, shotname+extension)
    score_threshold = 0.1365
    score_threshold_ = 0.17468
    t1 = time.time()
    blocks = main(img, score_threshold)
    t2 = time.time()
    print('总耗时：', round(t2-t1,3))
    # print(blocks)
    # font = cv2.FONT_HERSHEY_COMPLEX
    # for block in blocks:
    #     position = block['position']
    #     bug = block['bug']
    #     obj_score = block['obj_score']
    #     print('pp', obj_score,score_threshold)
    #     if obj_score<score_threshold_: continue
    #     num = classes.index(bug)
    #     x1, y1 = position[0]['x'], position[0]['y']
    #     x2, y2 = position[2]['x'], position[2]['y']
    #     color = VOC_COLORMAP[num]
    #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    #     # 标注文本
    #     cv2.putText(img, bug, (x1, y1), font, 0.4, color, 1)
    #     cv2.imwrite(img_save_path, img)