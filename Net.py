# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:20:49 2018

@author: admin
"""
import os
import numpy as np
import tensorflow as tf
import THZData
from loss import my_compute_loss

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-04
_LEAKY_RELU = 0.1
#_ANCHORS = [(16, 30), (33, 23), (30, 61), (62, 45), (59, 119)]
ANCHORS = [(1.3221, 1.73145)#, (3.19275, 4.00944), (5.05587, 8.09892),
            #(9.47112, 4.84053), (11.2364, 10.0071)
            ]
anchor_num = len(ANCHORS)
class_map = {
    'knife' : 5,
    'water' : 6,
    'gun' : 7,
    'powder' : 8,
    'warning' : 9,
    }


def arg_scope(batch_norm_var_collection='moving_vars',is_training=True,reuse=False):
    batch_norm_params = {  # 定义batch normalization（标准化）的参数字典
      'decay': _BATCH_NORM_DECAY,  # 定义参数衰减系数
      'epsilon': _BATCH_NORM_EPSILON,  
      'scale': True,
      'is_training': is_training,
      'fused': None,  # Use fused batch norm if possible.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
          }
      }
    with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
        with slim.arg_scope([slim.conv2d], 
             normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
             biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)) as sc:
            return sc
            
          

# input_image 448*448
def darknet19(input_image,num_classes=5):
    with slim.arg_scope(arg_scope()):
        net = slim.conv2d(input_image,32,[3,3], scope="conv1")
        net = slim.max_pool2d(net,2,stride=2,scope="max_pool1")
        net = slim.conv2d(net,64,[3,3], scope="conv2")
        net = slim.max_pool2d(net,2,stride=2,scope="max_pool2")
        net = slim.conv2d(net,128,[3,3], scope="conv3_1")
        net = slim.conv2d(net,64,[1,1], scope="conv3_2")
        net = slim.conv2d(net,128,[3,3], scope="conv3_3")
        net = slim.max_pool2d(net,2,stride=2,scope="max_pool3")
        net = slim.conv2d(net,256,[3,3], scope="conv4_1")
        net = slim.conv2d(net,128,[1,1], scope="conv4_2")
        net = slim.conv2d(net,256,[3,3], scope="conv4_3")
        net = slim.max_pool2d(net,2,stride=2,scope="max_pool4")
        net = slim.conv2d(net,512,[3,3], scope="conv5_1")
        net = slim.conv2d(net,256,[1,1], scope="conv5_2")
        net = slim.conv2d(net,512,[3,3], scope="conv5_3")
        net = slim.conv2d(net,256,[1,1], scope="conv5_4")
        net = slim.conv2d(net,512,[3,3], scope="conv5_5")
        # 存储这一层特征图，以便后面passthrough层
        #shortcut = net      #大小为14*14
#        net = slim.max_pool2d(net,2,stride=2,scope="max_pool5")
#        net = slim.conv2d(net,1024,[3,3], scope="conv6_1")
#        net = slim.conv2d(net,512,[1,1], scope="conv6_2")
#        net = slim.conv2d(net,1024,[3,3], scope="conv6_3")
#        net = slim.conv2d(net,512,[1,1], scope="conv6_4")
#        net = slim.conv2d(net,1024,[3,3], scope="conv6_5")
        #net = slim.conv2d(net,1000,[1,1], scope="convful")
        #net = slim.avg_pool2d(net,7,scope="avg_pool")
        
        #upsample
#        net = slim.conv2d(net,1024,[3,3], scope="conv7_1")
#        net = slim.conv2d(net,1024,[3,3], scope="conv7_2")
#        net = tf.image.resize_nearest_neighbor(net, (26, 26))
        
#        shortcut = slim.conv2d(shortcut,512,[3,3], scope="conv7_3")
#        shortcut = slim.conv2d(shortcut,512,[3,3], scope="conv7_4")
        
        #net = tf.concat([shortcut, net], axis=-1) 
        net = slim.conv2d(net, 1024, 3, 1, scope='conv8')
        
        #reorg
#        net = slim.conv2d(net,1024,[3,3], scope="conv7_1")
#        net = slim.conv2d(net,1024,[3,3], scope="conv7_2")
#        
#        shortcut = slim.conv2d(shortcut,512,[3,3], scope="conv7_3")
#        shortcut = slim.conv2d(shortcut,512,[3,3], scope="conv7_4")
#        shortcut = reorg(shortcut, 2)
#        
#        net = tf.concat([shortcut, net], axis=-1) 
#        net = slim.conv2d(net, 1024, 3, 1, scope='conv8')
        
        net = slim.conv2d(net, anchor_num*(5+num_classes), [1,1], normalizer_fn =None, activation_fn=None,scope="conv_dec")
    return net

def reorg(x,stride):
    return tf.space_to_depth(x,block_size=stride)   #返回一个与input具有相同的类型的Tensor
    # return tf.extract_image_patches(x,ksizes=[1,stride,stride,1],strides=[1,stride,stride,1],rates=[1,1,1,1],padding='VALID')


def train():
    
    input_size = (416,416)
    B = len(ANCHORS)
    C = len(class_map)
    learning_rate = 0.0004
    batches = 300000
    scales = (np.array([5,5,5,5]),1,.5,np.ones(C))
    output_sizes = input_size[0]//16, input_size[1]//16
    labels_path = './THZDataset/'
    traindata,testdata= THZData.read_all_data(labels_path)
    
    #增加一个在第0的维度--batch_size
    tf_image = tf.placeholder(tf.float32,[None,input_size[0],input_size[1],3])  #定义placeholder
    tf_label = tf.placeholder(tf.float32,[None,output_sizes[0],output_sizes[1],B*(5+C)])
    
    #with slim.arg_scope(arg_scope()):
    model_output = darknet19(tf_image)
    
    #output_decoded = decode(model_output,output_sizes,len(class_map),_ANCHORS)
    '''--------calculate loss--------'''
    loss = my_compute_loss(model_output,tf_label,ANCHORS,scales)
    '''--------Optimizer--------'''
    update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
    with tf.control_dependencies( update_ops ):
        optimizer = tf.train.AdamOptimizer( learning_rate=learning_rate ).minimize(loss)
    
    tf.summary.scalar( 'loss',  loss )
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    with tf.device('/cpu:0'):
        with tf.Session() as sess:  
            writer = tf.summary.FileWriter( "logs/", sess.graph )
            number = 0
            
            saver = tf.train.Saver( max_to_keep = 3 )
            save_path = "./models"
            last_checkpoint = tf.train.latest_checkpoint( save_path, 'checkpoint' )
            if last_checkpoint:
                saver.restore( sess, last_checkpoint )
                number = int( last_checkpoint[24 :] )+1
                print( 'Reuse model form: ', format( last_checkpoint ) )
            else:
                init.run()
            
            print ("start train")
            avg_loss = 0
            for i in range(number,batches+1):
                Images,labels = traindata.next_batch(8)
                _, _loss, rs = sess.run([optimizer,loss,merged],feed_dict={tf_image: Images,tf_label:labels})
                #bboxes, obj_probs, class_probs= sess.run(output_decoded,feed_dict={tf_image: Images}) # (1,26,26,425)
                #print(bboxes.shape)#bboxes, obj_probs, class_probs
                avg_loss += _loss
                writer.add_summary( rs, i )
                if i % 100 == 0 :
                    name =  'yolomodel.ckpt'
                    saver.save( sess, os.path.join( save_path, name ), global_step = i )
                    print( 'AVG_Cost after epoch %i: %f' %( i, avg_loss/100 ) )
                    avg_loss = 0
                    test_loss = 0
                    for j in range(10):
                        test_images, test_labels = testdata.next_batch(8)
                        test_loss += sess.run(loss,feed_dict={tf_image: test_images,tf_label:test_labels})
                    print( 'Test_Cost after epoch %i: %f' %( i, test_loss/10 ) )

def read_test():
    labels_path = './THZDataset/'
    traindata,testdata= THZData.read_all_data(labels_path)
    Images,labels = traindata.next_batch(1)
    
    print(Images.shape)
    print(labels.shape)
        
if __name__ == '__main__':
    #read_test()
    train()   
    
#labels_path = './THZDataset/'
#traindata,testdata= THZData.read_all_data(labels_path)
#Images,labels = traindata.next_batch(1)
#a = labels[0]      