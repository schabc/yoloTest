# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:12:57 2018

@author: admin
"""
import os
import time
import tensorflow as tf
import cv2
from PIL import Image
from Net import darknet19
from decode import decode
import numpy as np
from utils import preprocess_image, postprocess, draw_detection, generate_colors
import THZData
import matplotlib.pyplot as plt
from loss import my_compute_loss

ANCHORS = [(1.3221, 1.73145)#, (3.19275, 4.00944), (5.05587, 8.09892),
            #(9.47112, 4.84053), (11.2364, 10.0071)
            ]
class_map = {
#    'knife' : 5,
#    'water' : 6,
#    'gun' : 7,
#    'powder' : 8,
#    'warning' : 9,
    0: 'knife' ,
    1: 'water',
    2: 'gun',
    3: 'powder',
    4: 'warning',
    
    }
model_path = os.path.join('models','yolomodel.ckpt-86300')    #加载模型路径
image_shape = (416,416)
image_path = os.path.join('THZDataset','VOC2007','JPEGImages')

def vis_detections(image, bboxes, scores, class_name,thresh=0.1):
    """Draw detected bounding boxes."""

    for j in range(len(image)):#len(image)
        inds = np.where(scores[j] >= thresh)[0]
        if len(inds) == 0:
            return
    
        im = image[j][:, :, (2, 1, 0)]
        #im = np.array(image)#[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for i in inds:
            bbox = bboxes[j][i]
            score = scores[j][i]
    
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_map[class_name[j][i]], score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
    
        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}').format(class_map[class_name[j][i]], class_map[class_name[j][i]],
                                                      thresh),
                     fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()


def detect(Images,sess,input_size=(416,416)):
    
    output_sizes = input_size[0]//16, input_size[1]//16
    tf_image = tf.placeholder(tf.float32,[None,input_size[0],input_size[1],3])  #定义placeholder
    model_output = darknet19(tf_image)
    output_decoded = decode(model_output,output_sizes,len(class_map),ANCHORS)
    
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())   #初始化tensorflow全局变量
#        saver = tf.train.Saver()
#        saver.restore(sess, model_path)  #把模型加载到当前session中
    st = time.time()
    bboxes, obj_probs, class_probs = sess.run(output_decoded, feed_dict={tf_image: Images})  #这个函数返回框的坐标，目标置信度，类别置信度
    print(time.time()-st)
    
    bboxes,scores,class_max_index = postprocess(bboxes,obj_probs,class_probs)   #得到候选框之后的处理，先留下阈值大于0.5的框，然后再放入非极大值抑制中去
    #colors = generate_colors(class_map)
    #img_detection = draw_detection(Image, bboxes, scores, class_max_index, class_map, colors)  #得到图片
    vis_detections(Images[0], bboxes, scores, class_max_index,thresh=0.5)

    
        
def main():
    imagename = "/000002.jpg"
    img = Image.open(image_path + imagename)
    image = img.resize((416,416))
    image = np.array(image)               
    image = image/255 
    image = np.expand_dims(image,axis=0)
    
    labels_path = './THZDataset/'
    traindata,testdata= THZData.read_all_data(labels_path)
    input_size=(416,416)
    
    B = len(ANCHORS)
    C = len(class_map)
    scales = (np.array([5,5,5,5]),1,.5,np.ones(C))
    output_sizes = input_size[0]//16, input_size[1]//16
    tf_image = tf.placeholder(tf.float32,[None,input_size[0],input_size[1],3])  #定义placeholder
    tf_label = tf.placeholder(tf.float32,[None,output_sizes[0],output_sizes[1],B*(5+C)])
    model_output = darknet19(tf_image)
    loss = my_compute_loss(model_output,tf_label,ANCHORS,scales)
    output_decoded = decode(model_output,output_sizes,len(class_map),ANCHORS)
    
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)  #把模型加载到当前session中
        for i in range(5):
            Images,labels = testdata.next_batch(1)
            st = time.time()
            bboxes, obj_probs, class_probs = sess.run(output_decoded, feed_dict={tf_image: Images})  #这个函数返回框的坐标，目标置信度，类别置信度
            print('spendTime: ',time.time()-st)           
            _loss = sess.run(loss,feed_dict={tf_image: Images,tf_label:labels})
            print('loss: ',_loss)
            
            
            bboxes,scores,class_max_index = postprocess(bboxes,obj_probs,class_probs,image_shape)   #得到候选框之后的处理，先留下阈值大于0.5的框，然后再放入非极大值抑制中去
            #colors = generate_colors(class_map)
            #img_detection = draw_detection(Image, bboxes, scores, class_max_index, class_map, colors)  #得到图片
            vis_detections(Images, bboxes, scores, class_max_index,thresh=0.1)
#        img_detection = detect(Images)[0]
    
#        ShowImg = cv2.resize(img_detection,(377,754),interpolation=cv2.INTER_AREA)
#        #cv2.imwrite(image, img_detection)
#        cv2.imshow("detection_results", ShowImg)  #显示图片
#        cv2.waitKey(0)  #等待按任意键退出
    
main()