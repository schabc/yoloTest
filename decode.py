# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:20:08 2018

@author: admin
"""
import tensorflow as tf


def decode(detection_feat, feat_sizes=(26, 26), num_classes=5,anchors=None):
    """decode from the detection feature"""
    '''
     model_output:darknet19网络输出的特征图
     output_sizes:darknet19网络输出的特征图大小，默认是26*26(默认输入416*416，下采样16)
    '''
    H, W = feat_sizes
    num_anchors = len(anchors)
    # H*W*num_anchors*(num_class+5)，第一个维度自适应batchsize
    detetion_results = tf.reshape(detection_feat, [-1, H * W, num_anchors,
                                                   num_classes + 5])
    # darknet19网络输出转化——偏移量、置信度、类别概率
    bbox_xy = tf.nn.sigmoid(detetion_results[:, :, :, 0:2])
    bbox_wh = tf.exp(detetion_results[:, :, :, 2:4])
    
    obj_probs = tf.nn.sigmoid(detetion_results[:, :, :, 4])
    class_probs = tf.nn.softmax(detetion_results[:, :, :, 5:])

    # 将anchors转变成tf格式的常量列表
    anchors = tf.constant(anchors, dtype=tf.float32)
    
    # 构建特征图每个cell的左上角的xy坐标
    height_ind = tf.range(H, dtype=tf.float32)
    width_ind = tf.range(W, dtype=tf.float32)
    # 变成 H*W个cell
    x_offset, y_offset = tf.meshgrid(height_ind, width_ind)
    # 和上面[H*W,num_anchors,num_class+5]对应
    x_offset = tf.reshape(x_offset, [1, -1, 1])
    y_offset = tf.reshape(y_offset, [1, -1, 1])

    # decode
    bbox_x = (bbox_xy[:, :, :, 0]  + x_offset)  / W
    bbox_y = (bbox_xy[:, :, :, 1]  + y_offset)  / H
    bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / W * 0.5#半长
    bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / H * 0.5

    bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h,
                       bbox_x + bbox_w, bbox_y + bbox_h], axis=3)

    return bboxes, obj_probs, class_probs