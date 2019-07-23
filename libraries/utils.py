# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import h5py
from PIL import Image
import os
import glob
from scipy.ndimage import fourier_shift
from skimage.transform import rescale
from skimage.feature import register_translation
from skimage import io 


##########################################################
def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def load_from_directory_to_pickle(base_dir,out_dir,band='NIR'):
    
    '''
    base_dir: specifies the root probav directory (the one downloaded from probav chalenge website)
    out_dir: specifies where to place the pickles
    '''

    out_dir=out_dir.rstrip()
    
    train_dir = os.path.join(base_dir, 'train/'+band)
    dir_list=glob.glob(train_dir+'/imgset*')
    
    dir_list.sort()
    
    input_images_LR = np.array([[io.imread(fname,dtype=np.uint16) for fname in sorted(glob.glob(dir_name+'/LR*.png'))] 
                             for dir_name in dir_list ])
    
    input_images_LR.dump(out_dir+'/'+'LR_dataset_'+band+'.npy')
    
    input_images_HR = np.array([io.imread(glob.glob(dir_name+'/HR.png')[0],dtype=np.uint16) for dir_name in dir_list ])
    
    input_images_HR.dump(out_dir+'/'+'HR_dataset_'+band+'.npy')
    
    mask_HR = np.array([io.imread(glob.glob(dir_name+'/SM.png')[0],dtype=np.bool) for dir_name in dir_list ])
    
    mask_HR.dump(out_dir+'/'+'HR_mask_'+band+'.npy')
    
    mask_LR = np.array([[io.imread(fname,dtype=np.bool)for fname in sorted(glob.glob(dir_name+'/QM*.png'))] 
                             for dir_name in dir_list ])
    
    mask_LR.dump(out_dir+'/'+'LR_mask_'+band+'.npy')
        
        
    #load test 
    train_dir = os.path.join(base_dir, 'test/'+band)
    dir_list=glob.glob(train_dir+'/imgset*')
    dir_list.sort()
    test_images_LR = np.array([[io.imread(fname,dtype=np.uint16) for fname in sorted(glob.glob(dir_name+'/LR*.png'))] 
                             for dir_name in dir_list ])
    
    test_images_LR.dump(out_dir+'/'+'LR_test_'+band+'.npy')
    
    test_mask_LR = np.array([[io.imread(fname,dtype=np.bool) for fname in sorted(glob.glob(dir_name+'/QM*.png'))] 
                             for dir_name in dir_list ])
    
    test_mask_LR.dump(out_dir+'/'+'LR_mask_'+band+'_test.npy')
    
    
###Registration

def registration_imageset_against_best_image_without_union_mask(batch_training,batch_training_mask, upsample_factor):
    """ 
        this method registers all images of an imageset represented by the 2nd dimension of batch_training with respect to the 
        most cleared one, from the mask coverage point of view
        
        batch_training: A list of b 4-D numpy array  [n,h,w] where b is the total number of scenes, n is the number of images in 
        an imageset representing the same scene and h,w dimensions represent the image size.
        batch_training_mask: A list of b 4-D numpy array of shape [n,h,w]
    """
    
   
    batch_training_registered=[]#np.empty_like(batch_training_applied_mask)
    batch_training_mask_registered=[]#np.empty_like(batch_training_mask)
    #if we take at every imageset the most clear image than we have every time a new image ordering
    #within the imageset, we new to keep track the new order.
    new_index_orders=[]
    
    shifts=[]#np.empty([batch_training_applied_mask.shape[0],batch_training_applied_mask.shape[1],2])
    
    #For each image set
    for i in range(len(batch_training)):
        
        batch_training[i]=np.array(batch_training[i])
        
        imageset_training_registered=np.empty_like(batch_training[i])
        imageset_training_mask_registered=np.empty_like(batch_training_mask[i])
        imageset_shifts=np.empty([batch_training[i].shape[0],2])
        
        new_index_order=np.empty([batch_training[i].shape[0]],dtype='int16')
        #For each of the x images 
        
        #for image_set in mask_LR:
        index=np.argsort(np.sum(np.array(batch_training_mask[i]),axis=(1,2)))[::-1][0]
        z=0
        
        for j in range(batch_training[i].shape[0]):
            #we consider the first image in the set as the reference image

    
            reference_image=batch_training[i][index]
            
            if j==index:
                j_index=0
                z=1
            else:
                j_index=j+1-z
                
            #now we know the old index where it goes in the new ordering
            new_index_order[j_index]=j
            
            shifted_image=batch_training[i][j]
            
           
            #Compute the shift
            shift, error, diffphase = register_translation(reference_image.squeeze(), shifted_image.squeeze(),upsample_factor=upsample_factor)
            #print(shift)
            imageset_shifts[j_index]=np.asarray(shift)
            
            
            ###Image
            #shift is applied to the original image from the batch_training variable, in the fourier domain
            shifted_image_not_masked=batch_training[i][j]
            corrected_image = fourier_shift(np.fft.fftn(shifted_image_not_masked.squeeze()), imageset_shifts[j_index])
            corrected_image = np.fft.ifftn(corrected_image)
            imageset_training_registered[j_index]=corrected_image
            
            ###Mask
            #apply the same shift on the masks
            shifted_mask=batch_training_mask[i][j]
            corrected_mask = fourier_shift(np.fft.fftn(shifted_mask.squeeze()), imageset_shifts[j_index])
            corrected_mask = np.fft.ifftn(corrected_mask)
            imageset_training_mask_registered[j_index]=corrected_mask
    
        #Transform the masks from float64 back to bool
        imageset_training_mask_registered=np.round(imageset_training_mask_registered)
        imageset_training_mask_registered=imageset_training_mask_registered.astype('bool')

        batch_training_registered.append(imageset_training_registered)
        batch_training_mask_registered.append(imageset_training_mask_registered)
        shifts.append(imageset_shifts)
        new_index_orders.append(new_index_order)
            
    return batch_training_registered,batch_training_mask_registered,shifts,new_index_orders



def upsampling_mask(masks,scale=3):
    '''
    masks of shape like [b,9,128,128,1]
    '''
    masks_images=np.empty([masks.shape[0],
                        masks.shape[1],
                        masks.shape[2]*scale,
                        masks.shape[3]*scale],dtype='bool')
    
    for i in range(masks.shape[0]):
        
        upsampled_image=np.zeros( (masks.shape[2]*scale,
                                   masks.shape[3]*scale), 
                                 dtype=np.bool)
        for j in range(masks.shape[1]):
            upsampled_image=rescale(masks[i,j].squeeze(), 
                                    scale=3, 
                                    order=0,
                                    mode='constant',
                                    anti_aliasing=False,
                                    multichannel=False,
                                    preserve_range=True)
            upsampled_image=upsampled_image.astype('bool')
            masks_images[i,j]=upsampled_image
            
    return masks_images


#upsampling 
def upsampling_mask_all_imageset(masks,scale=3):
    '''
    masks of shape like [b,9,128,128,1]
    '''
    
    height=masks[0][0].shape[0]
    width=masks[0][0].shape[1]
    masks_images=np.empty([masks.shape[0]],dtype=object)
    
    for i in range(masks.shape[0]):
        
        list_maskset=[]
        
        upsampled_image=np.zeros( (height*scale,
                                   width*scale), 
                                 dtype='bool')
        for j in range(len(masks[i])):
            upsampled_image=rescale(masks[i][j].squeeze(), 
                                    scale=3, 
                                    order=0,
                                    mode='constant',
                                    anti_aliasing=False,
                                    multichannel=False,
                                    preserve_range=True)
            
            upsampled_image=np.round(upsampled_image).astype('bool')
            list_maskset.append(upsampled_image)
            
        masks_images[i]=list_maskset
            
    return masks_images
def upsampling_without_aggregation_all_imageset(batch_training_to_up,scale=3):
    '''
    batch_training of shape like [b,x,128,128]
    '''
    
    height=batch_training_to_up[0][0].shape[0]
    width=batch_training_to_up[0][0].shape[1]
    
    SR_images=np.empty([batch_training_to_up.shape[0]],dtype=object)
    
    for i in range(batch_training_to_up.shape[0]):
        
        list_imageset=[]
        
        upsampled_image=np.zeros( (height*scale,
                                   width*scale), 
                                 dtype=np.float32)
        
        for j in range(len(batch_training_to_up[i])):
            upsampled_image=rescale(batch_training_to_up[i][j].squeeze(), 
                                    scale=3, 
                                    order=3,
                                    mode='edge',
                                    anti_aliasing=False,
                                    multichannel=False,
                                    preserve_range=True)
            upsampled_image=upsampled_image.astype('float32')
            list_imageset.append(upsampled_image)
            
        SR_images[i]=list_imageset
            
    return SR_images    

#he_normal_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
###########################################################




#he_normal_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,seed=12345)
he_normal_init =tf.contrib.layers.xavier_initializer(uniform=False,seed=1234)

def BatchNorm(input, is_train, decay=0.999, name='BatchNorm'):
    '''
    https://github.com/zsdonghao/tensorlayer/blob/master/tensorlayer/layers.py
    https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
    http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow
    '''
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops
    
    axis = list(range(len(input.get_shape()) - 1))
    fdim = input.get_shape()[-1:]

    with tf.variable_scope(name):
        beta = tf.get_variable('beta', fdim, initializer=tf.constant_initializer(value=0.0))
        gamma = tf.get_variable('gamma', fdim, initializer=tf.constant_initializer(value=1.0))
        moving_mean = tf.get_variable('moving_mean', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
        moving_variance = tf.get_variable('moving_variance', fdim, initializer=tf.constant_initializer(value=0.0), trainable=False)
  
        def mean_var_with_update():
            batch_mean, batch_variance = tf.nn.moments(input, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, batch_variance, decay, zero_debias=True)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = control_flow_ops.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(input, mean, variance, beta, gamma, 1e-3) #, tf.stack([mean[0], variance[0], beta[0], gamma[0]])


def InstanceNorm(input, axis=[2,3] , decay=0.999, name='InstanceNorm',trainable=True):
    '''
    
    '''
    from tensorflow.python.training import moving_averages
    from tensorflow.python.ops import control_flow_ops
    
    #axis = list(range(len(input.get_shape()) - 1))
    fdim = input.get_shape()[-1:]
    #shape=np.array(fdim)
    #shape[axis]=1
    
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', fdim , dtype=tf.float32,initializer=tf.constant_initializer(value=0.0),trainable=trainable)
        gamma = tf.get_variable('gamma', fdim, dtype=tf.float32,initializer=tf.constant_initializer(value=1.0),trainable=trainable)
        
        instance_mean, instance_variance = tf.nn.moments(input, axis ,keep_dims=True)
    
    return tf.nn.batch_normalization(input, instance_mean, instance_variance, beta, gamma, 1e-3)#, tf.stack([mean[0], variance[0], beta[0], gamma[0]])


def Conv3D(input, kernel_shape, strides, padding, scope_name='Conv3d', W_initializer=he_normal_init, trainable=True,bias=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        W = tf.get_variable("W", kernel_shape, dtype=tf.float32,initializer=W_initializer,trainable=trainable)
        if bias is True:
            b = tf.get_variable("b", (kernel_shape[-1]),dtype=tf.float32,initializer=tf.constant_initializer(value=0.0),trainable=trainable)
        else:
            b = 0
        
    return tf.nn.conv3d(input, W, strides, padding) + b

def Conv2D(inputs, kernel_shape, strides, padding, scope_name='Conv2d',W_initializer=he_normal_init, bias=True,trainable=True):
    '''
    A method that does convolution + relu on inputs
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        kernels=tf.get_variable('W',shape=kernel_shape,dtype=tf.float32,initializer=W_initializer,trainable=trainable)
        
        if bias is True:
            biases=tf.get_variable('b',shape=[kernel_shape[-1]],dtype=tf.float32,initializer=tf.constant_initializer(),trainable=trainable)
        else:
            biases = 0
        conv=tf.nn.bias_add(tf.nn.conv2d(inputs,kernels,strides=strides,padding=padding),biases)   

    return conv

def Conv2D_transposed(inputs, kernel_shape, output_shape,strides, padding, scope_name='Conv2d',W_initializer=he_normal_init, bias=True):
    '''
    A method that does convolution + relu on inputs
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        kernels=tf.get_variable('W',shape=kernel_shape,dtype=tf.float32,initializer=W_initializer)
        
        if bias is True:
            biases=tf.get_variable('b',shape=[kernel_shape[-2]],dtype=tf.float32,initializer=tf.constant_initializer())
        else:
            biases = 0
        conv=tf.nn.bias_add(tf.nn.conv2d_transpose(inputs,kernels,output_shape,strides=strides,padding=padding),biases)   

    return conv

def depth_to_space_3D(x, block_size):
    ds_x = tf.shape(x)
    x = tf.reshape(x, [ds_x[0]*ds_x[1], ds_x[2], ds_x[3], ds_x[4]])
    
    y = tf.depth_to_space(x, block_size)
    
    ds_y = tf.shape(y)
    x = tf.reshape(y, [ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3]])
    return x
 