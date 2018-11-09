# -*- coding: utf-8 -*-
import random
import colorsys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 【1】图像预处理(pre process前期处理)
def preprocess_image(image,image_size=(416,416)):
    # 复制原图像
    image_cp = np.copy(image).astype(np.float32)

    # resize image
    image_rgb = cv2.cvtColor(image_cp,cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb,image_size)

    # normalize归一化
    image_normalized = image_resized.astype(np.float32) / 255.0

    # 增加一个维度在第0维——batch_size
    image_expanded = np.expand_dims(image_normalized,axis=0)
    return image_expanded

# 【2】筛选解码后的回归边界框——NMS(post process后期处理)
def postprocess(bboxes,obj_probs,class_probs,image_shape=(416,416),threshold=0.5):
    # bboxes表示为：图片中有多少box就多少行；4列分别是box(xmin,ymin,xmax,ymax)
    bboxes = np.reshape(bboxes,[len(bboxes),-1,4])
    # 将所有box还原成图片中真实的位置
    bboxes[:,:,0:1] *= float(image_shape[1]) # xmin*width
    bboxes[:,:,1:2] *= float(image_shape[0]) # ymin*height
    bboxes[:,:,2:3] *= float(image_shape[1]) # xmax*width
    bboxes[:,:,3:4] *= float(image_shape[0]) # ymax*height
    bboxes = bboxes.astype(np.int32)    #转变为int类型

    # (1)cut the box:将边界框超出整张图片(0,0)—(415,415)的部分cut掉
    bbox_min_max = [0,0,image_shape[1]-1,image_shape[0]-1]
    bboxes = bboxes_cut(bbox_min_max,bboxes)

    # 置信度*max类别概率=类别置信度scores
    obj_probs = np.reshape(obj_probs,[len(obj_probs),-1])  #13*13*5=845
    class_probs = np.reshape(class_probs,[len(class_probs),len(obj_probs[0]),-1])  #[13*13*5,80]
    class_max_index = np.argmax(class_probs,axis=2) # 得到max类别概率对应的维度
    class_probs1 = np.ones((len(class_max_index),len(obj_probs[0])))
    for i in range(len(class_max_index)):
        class_probs1[i] = class_probs[i][np.arange(len(obj_probs[0])),class_max_index[i]]#np.arange(len(class_max_index)),np.arange(len(obj_probs[0]))
    scores = obj_probs * class_probs1   #一一对应，得到类别置信度

    idxes = np.argmax(scores,axis=1)
    # 类别置信度scores>threshold的边界框bboxes留下
    keep_index = scores > threshold
    for i in range(len(idxes)):
        keep_index[i,idxes[i]] = True
    class_max_index1 = []
    scores1 = []
    bboxes1 = []
    for i in range(len(class_max_index)):
        class_max_indexi = class_max_index[i][keep_index[i]]
        scoresi = scores[i][keep_index[i]]    #一一对应，得到保留下来的框的类别置信度
        bboxesi = bboxes[i][keep_index[i]]    #一一对应，得到保留下来的框(左上-右下)
    
        # (2)排序top_k(默认为400)
        class_max_indexi,scoresi,bboxesi = bboxes_sort(class_max_indexi,scoresi,bboxesi)  #只保留类别置信度为top k的k个
        # (3)NMS
        class_max_indexi,scoresi,bboxesi = bboxes_nms(class_max_indexi,scoresi,bboxesi)  #关键的一步，非极大值抑制
        class_max_index1.append(class_max_indexi)
        scores1.append(scoresi)
        bboxes1.append(bboxesi)
    return bboxes1,scores1,class_max_index1   #返回保存下来的框

def generate_colors(class_names):  #为每个类别显示不同的颜色，哈哈，颜值
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors


# 【3】绘制筛选后的边界框
def draw_detection(im, bboxes, scores, cls_inds, labels, colors, thr=0.1): #传入的参数分别为原始图片，最终得到的框，类别置信度，类别索引，类别库，各个类别的颜色，thr为类别置信度阈值
    # draw image
    imgcv = np.copy(im)
    _, h, w, _ = imgcv.shape    #得到图片的高度和宽度
    for i, box in enumerate(bboxes):
#        if scores[i] < thr:    #我感觉多此一举，因为前面已经过滤过一次socres了
#            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 1000)
        #cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),(255, 114, 0), thick)
        cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),(74, 120, 237),thick)  #显示框
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)

        cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, colors[cls_indx], thick//3)
        #cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (184, 166, 51), thick)  #显示类别和类别置信度
    return imgcv  #返回图片

## 对应【2】:筛选解码后的回归边界框
# (1)cut the box:将边界框超出整张图片(0,0)—(415,415)的部分cut掉
def bboxes_cut(bbox_min_max,bboxes):
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)    #先转置
    bbox_min_max = np.transpose(bbox_min_max)
    # cut the box
    bboxes[0] = np.maximum(bboxes[0],bbox_min_max[0]) # xmin
    bboxes[1] = np.maximum(bboxes[1],bbox_min_max[1]) # ymin
    bboxes[2] = np.minimum(bboxes[2],bbox_min_max[2]) # xmax
    bboxes[3] = np.minimum(bboxes[3],bbox_min_max[3]) # ymax
    bboxes = np.transpose(bboxes)   #再转置回来
    return bboxes

# (2)按类别置信度scores降序，对边界框进行排序并仅保留top_k
def bboxes_sort(classes,scores,bboxes,top_k=400):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]  #0~79
    scores = scores[index][:top_k]
    bboxes = bboxes[index][:top_k]
    return classes,scores,bboxes   #这三个变量分别表示类别（0，79）针对于coco数据集，scores表示这个框的类别置信度，bboxes表示存储着框的四个值（左上-右下）

# (3)计算IOU+NMS,计算两个box的IOU
def bboxes_iou(bboxes1,bboxes2):   #计算IOU的值
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)

    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax-int_ymin,0.)
    int_w = np.maximum(int_xmax-int_xmin,0.)

    # 计算IOU
    int_vol = int_h * int_w # 交集面积
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1]) # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1]) # bboxes2面积
    IOU = int_vol / (vol1 + vol2 - int_vol) # IOU=交集/并集
    return IOU

# NMS，或者用tf.image.non_max_suppression(boxes, scores,self.max_output_size, self.iou_threshold)
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):   #iou阈值为0.5， 这些值已经按照从高到底进行排序
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])   #逻辑或，IOU没有超过0.5或者是不同的类则保存下来
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)   #逻辑与

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]   #返回保存下来的框
