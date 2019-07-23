import tensorflow as tf
import numpy as np
import time
import glob
import scipy
import argparse
import sys
sys.path.insert(0, '../libraries')

import os
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.ndimage import fourier_shift
from skimage.transform import rescale
from skimage.feature import register_translation
from collections import defaultdict
from sklearn.utils import shuffle
import warnings
import skimage
from skimage import io
from tensorflow.python.client import timeline

from utils import safe_mkdir,BatchNorm, Conv3D, Conv2D,InstanceNorm

from dataloader import new_coordinate,load_testset_preprocesses,load_dataset_best9,load_dataset

import json
import shutil
import argparse


class SR_network(object):
    def __init__(self,config):
        
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        #self.base_dir=config['base_dir']
        self.tensorboard_dir=config['tensorboard_dir']
        
        self.skip_step = config['skip_step']
        
        self.channels=config['channels']
        # Size of input temporal depth
        self.T_in = config['T_in'] 
        # Upscaling factor
        self.R = config['R']
        #Use full training set or best 9 training set
        self.full=config['full']
        #
        self.patch_size_HR=config['patch_size_HR']
        self.patch_size_LR=config['patch_size_LR']
        self.border=config['border']
        self.spectral_band=config['spectral_band']
        
        self.dyn_filter_size=9
        
        self.RegNet_pretrain_dir=config['RegNet_pretrain_dir']
        self.SISRNet_pretrain_dir=config['SISRNet_pretrain_dir']
        self.dataset_path=config['dataset_path']
        self.n_chunks=config['n_chunks']
        
        self.TRAIN_REGIST_NET=True
        self.TRAIN_UPSAMPLING_NET=True
        self.TRAIN_FUSION_NET=True
        
        self.placeholder()
        self.sess=tf.Session()
      
        
        #Z-score 
        self.mu=config['mu']
        self.sigma=config['sigma']
        self.sigma_rescaled=config['sigma_rescaled']
        
        
        
    def placeholder(self):
        
        self.x=tf.placeholder('float32',shape=[None,None,None,None,None],name='x')
        #[b,9,128,128,1]
        #self.is_train = tf.placeholder(tf.bool, shape=[]) # Phase ,scalar
        self.y=tf.placeholder('float32',shape=[None,1,None,None,1],name='y')
        self.mask_y=tf.placeholder('float32',shape=[None,1,None,None,1],name='mask_y')
        self.y_filters=tf.placeholder('float32',shape=[None,None,self.dyn_filter_size**2],name='y_filters')
        self.fill_coeff=tf.placeholder(tf.float32,shape=[None,self.T_in,self.T_in,None,None,1],name='fill_coeff')
        self.norm_baseline=tf.placeholder('float32',shape=[None,1],name='norm_baseline')
        
    def get_data(self):
        with tf.name_scope('data'):
            
            #####################################################################
            #Load LR images, LR image masks, HR images and HR image masks (Both TRAINING and VALIDATION)
            
            #split=0.7
            
            try:
                print('Retrieving portion of training...')
                dataset_dict=next(self.gen)
            except StopIteration:
                return 0
            
            self.batch_training=dataset_dict['training']
            self.batch_training_mask=dataset_dict['training_mask']
            self.batch_training_y=dataset_dict['training_y']
            self.batch_mask_train_y=dataset_dict['training_mask_y']
            self.shifts=dataset_dict['shifts']
            
            self.batch_validation=dataset_dict['validation']
            self.batch_validation_mask=dataset_dict['validation_mask']
            self.batch_validation_y=dataset_dict['validation_y']
            self.batch_mask_valid_y=dataset_dict['validation_mask_y']
            
            self.shifts_valid=dataset_dict['shifts_valid']
            self.norm_validation=dataset_dict['norm_validation']
            
            #############before computing fill mask we have to register the mask also, at least with the 
            # the shifts from phase correlation and then we have to apply the fill_mask after the registration network
            #shift is applied to the original image from the batch_training variable, in the fourier domain
            #self.batch_training_mask_corrected=np.zeros_like(model.batch_training_mask,dtype='bool')
            
            self.batch_training_mask=np.round(self.batch_training_mask)
            self.batch_validation_mask=np.round(self.batch_validation_mask)
            self.batch_training_mask=self.batch_training_mask.astype('bool')
            self.batch_validation_mask=self.batch_validation_mask.astype('bool')
            
            for i in range(np.shape(self.batch_training_mask)[0]):
                shifted_mask_imageset=np.zeros_like(self.batch_training_mask[i],dtype='bool')
                for j in range(self.batch_training_mask[i].shape[0]):
                    shifted_mask=self.batch_training_mask[i][j]
                    corrected_mask = fourier_shift(np.fft.fftn(shifted_mask.squeeze()), -self.shifts[i][j])
                    corrected_mask = np.fft.ifftn(corrected_mask)
                    corrected_mask = corrected_mask.reshape([1,np.shape(self.batch_training_mask)[2],np.shape(self.batch_training_mask)[3],1])
                    shifted_mask_imageset[j]=np.round(corrected_mask)
                
                self.batch_training_mask[i]=shifted_mask_imageset
            
            
            for i in range(np.shape(self.batch_validation_mask)[0]):
                shifted_mask_imageset=np.zeros_like(self.batch_validation_mask[i],dtype='bool')
                for j in range(self.batch_validation_mask[i].shape[0]):
                    shifted_mask=self.batch_validation_mask[i][j]
                    corrected_mask = fourier_shift(np.fft.fftn(shifted_mask.squeeze()), -self.shifts_valid[i][j])
                    corrected_mask = np.fft.ifftn(corrected_mask)
                    corrected_mask = corrected_mask.reshape([1,np.shape(self.batch_validation_mask)[2],np.shape(self.batch_validation_mask)[3],1])
                    shifted_mask_imageset[j]=np.round(corrected_mask)
                
                self.batch_validation_mask[i]=shifted_mask_imageset
            
            ##############Compute coefficients for filling images where masked
            sh=self.batch_training_mask.shape
            self.fill_coeff_train=np.ones([sh[0],sh[1],sh[1],sh[2],sh[3],sh[4]],dtype='bool')
            for i in range(0,9):
                self.fill_coeff_train[:,:,i]=np.expand_dims(self.batch_training_mask[:,i],axis=1)

            for i in range(0,9):
                for j in range(i+1,9):
                    rows_indexes=[k for k in range(0,9) if k!=(j)]
                    #print(rows_indexes)
                    self.fill_coeff_train[:,rows_indexes,j]=self.fill_coeff_train[:,rows_indexes,j]*np.expand_dims(1-self.batch_training_mask[:,i],axis=1)
            
            for i in range(1,9):
                self.fill_coeff_train[:,i,0:i]=self.fill_coeff_train[:,i,0:i]*np.expand_dims(1-self.batch_training_mask[:,i],axis=1)
            
            #We need to fill in the regions where all the masks are zero. In this case we decide to uncover the hidden regions of
            #the considered image by turning the mask to 1 in those regions.
            f=np.sum(self.fill_coeff_train,axis=2)
            #[b,9,W,H,1]
            self.fill_coeff_train[:,range(9),range(9),:,:,:]=self.fill_coeff_train[:,range(9),range(9),:,:,:]+np.logical_not(f)[:,range(9),:,:,:]
            
            
            sh=self.batch_validation_mask.shape
            self.fill_coeff_valid=np.ones([sh[0],sh[1],sh[1],sh[2],sh[3],sh[4]],dtype='bool')
            for i in range(0,9):
                self.fill_coeff_valid[:,:,i]=np.expand_dims(self.batch_validation_mask[:,i],axis=1)

            for i in range(0,9):
                for j in range(i+1,9):
                    rows_indexes=[k for k in range(0,9) if k!=(j)]
                    #print(rows_indexes)
                    self.fill_coeff_valid[:,rows_indexes,j]=self.fill_coeff_valid[:,rows_indexes,j]*np.expand_dims(1-self.batch_validation_mask[:,i],axis=1)
            
            for i in range(1,9):
                self.fill_coeff_valid[:,i,0:i]=self.fill_coeff_valid[:,i,0:i]*np.expand_dims(1-self.batch_validation_mask[:,i],axis=1)
            
            #We need to fill in the regions where all the masks are zero. In this case we decide to uncover the hidden regions of
            #the considered image by turning the mask to 1 in those regions.
            f=np.sum(self.fill_coeff_valid,axis=2)
            #[b,9,W,H,1]
            self.fill_coeff_valid[:,range(9),range(9),:,:,:]=self.fill_coeff_valid[:,range(9),range(9),:,:,:]+np.logical_not(f)[:,range(9),:,:,:]
            
            ####################
            
            #standardize the dataset
            
            #self.mu=np.mean(self.batch_training)
            #print(self.mu)
            #self.sigma=np.std(self.batch_training)
            #print(self.sigma)
            #self.mu_rescaled=self.mu
            
            
            self.batch_training=(self.batch_training-self.mu)/self.sigma
            self.batch_training_y=(self.batch_training_y-self.mu)/self.sigma_rescaled
            self.batch_validation=(self.batch_validation-self.mu)/self.sigma
            self.batch_validation_y=(self.batch_validation_y-self.mu)/self.sigma_rescaled

            self.batch_training_norm=self.batch_training
            self.batch_validation_norm=self.batch_validation
            self.batch_training_y_norm=self.batch_training_y
            self.batch_validation_y_norm=self.batch_validation_y
            
            #####TRAINING SET
            #create y filters from the shifts computed during dataset creation
            self.y_filters_dyn=np.zeros([self.shifts.shape[0],np.shape(self.batch_training_norm)[1],
                        self.dyn_filter_size,
                        self.dyn_filter_size])

            #take the first 9 shifts in case we get from the loader the shifts for the whole imageset
            self.shifts=np.array([shift[0:self.T_in] for shift in self.shifts])
            #shift the index in order to get the index all positive in order to index a matrix
            self.shifts=self.shifts+int(self.dyn_filter_size/2)
            self.shifts=self.shifts.astype('int32')
            
            for i,shift in enumerate(self.shifts):
                self.y_filters_dyn[i,list(range(0,np.shape(self.batch_training_norm)[1])),self.shifts[i,:,0],self.shifts[i,:,1]]=1
                
            
            #[b,9,81]
            self.y_filters_dyn=np.reshape(self.y_filters_dyn,
                                          [-1,
                                           np.shape(self.batch_training_norm)[1],
                                           self.dyn_filter_size**2])
            
            self.y_filters_dyn=self.y_filters_dyn[:,1:,:]
            
            
            
            #VALIDATION SET
            #create y filters to match, so our multiclass output
            self.y_filters_valid_dyn=np.zeros([self.shifts_valid.shape[0],np.shape(self.batch_validation_norm)[1],
                        self.dyn_filter_size,
                        self.dyn_filter_size])

            #take the first 9 shifts in case we get from the loader the shifts for the whole imageset
            self.shifts_valid=np.array([shift[0:self.T_in] for shift in self.shifts_valid])
            #shift the index in order to get the index all positive in order to index a matrix
            self.shifts_valid=self.shifts_valid+int(self.dyn_filter_size/2)
            self.shifts_valid=self.shifts_valid.astype('int32')
            
            for i,shift in enumerate(self.shifts_valid):
                self.y_filters_valid_dyn[i,list(range(0,np.shape(self.batch_validation_norm)[1])),self.shifts_valid[i,:,0],self.shifts_valid[i,:,1]]=1
                
            #[b,9,81]
            self.y_filters_valid_dyn=np.reshape(self.y_filters_valid_dyn,
                                          [-1,
                                           np.shape(self.batch_validation_norm)[1],
                                           self.dyn_filter_size**2])
            #Remove the first
            self.y_filters_valid_dyn=self.y_filters_valid_dyn[:,1:,:]
      

            
    def loss(self):
        '''
        define loss function
        '''
        
        
        with tf.name_scope('loss') as scope:
            
            self.y_hat=self.logits
            
            s1=tf.shape(self.y)
            s2=tf.shape(self.y_hat)
            labels=tf.reshape(self.y,shape=[s1[0],s1[2],s1[3],s1[4]])
            predictions=tf.reshape(self.y_hat,shape=[s2[0],s2[2],s2[3],s2[4]])
            
            
            size_image=tf.shape(predictions)[1]
            #crop the center of the reconstructed HR image
            cropped_predictions=predictions[:,self.border:size_image-self.border,self.border:size_image-self.border]
            
            
            #All mse
            X=[]
            for i in range((2*self.border)+1):
                for j in range((2*self.border)+1):
                    cropped_labels=labels[:,i:i+(size_image-(2*self.border)),j:j+(size_image-(2*self.border))]
                    cropped_mask_y=self.mask_y[:,:,i:i+(size_image-(2*self.border)),j:j+(size_image-(2*self.border))]
                    
                    cropped_predictions_masked=cropped_predictions*tf.squeeze(cropped_mask_y,axis=1)
                    cropped_labels_masked=cropped_labels*tf.squeeze(cropped_mask_y,axis=1)
                    
                    #bias brightness
                    b=(1.0/tf.reduce_sum(cropped_mask_y,axis=[2,3,4]))*tf.reduce_sum(cropped_labels_masked-cropped_predictions_masked,axis=[1,2])
                    b=tf.reshape(b,[s1[0],1,1,1])
                    corrected_cropped_predictions=cropped_predictions_masked+b
                    corrected_cropped_predictions=corrected_cropped_predictions*tf.squeeze(cropped_mask_y,axis=1)
                    corrected_mse=(1.0/tf.reduce_sum(cropped_mask_y,axis=[2,3,4]))*tf.reduce_sum(tf.square(cropped_labels_masked-corrected_cropped_predictions),axis=[1,2])
                    #corrected_mse=tf.reduce_mean(tf.square(cropped_labels-corrected_cropped_predictions),axis=[1,2])
                    
                    X.append(corrected_mse) 
            
            X=tf.stack(X)
            #Take the minimum mse
            minim=tf.reduce_min(X,axis=0)
            mse=tf.reduce_mean(minim)
            
            self.loss = mse 
          
        
            
    def optimize(self):
        '''
        define optimization algorithm
        '''
        #with tf.name_scope('optimizer') as scope:
        #self.opt=tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
        optimizer=tf.train.AdamOptimizer(self.lr)
        
        self.grads_and_vars=optimizer.compute_gradients(self.loss)
        gradients, gradient_tensors = zip(*self.grads_and_vars)
        self.opt=optimizer.apply_gradients(self.grads_and_vars,global_step=self.gstep)
    
    def inference_FR(self):

        
        
        #padding
        stp = [[0,0], [1,1], [1,1], [1,1], [0,0]]
        sp = [[0,0], [0,0], [1,1], [1,1], [0,0]]

        
        ####################upsampling network
        F =64
        
        x1 = Conv3D(tf.pad(self.x, sp, mode='REFLECT'), [1,3,3,self.channels,F], [1,1,1,1,1], 'VALID', scope_name='conv_up_0',trainable=self.TRAIN_UPSAMPLING_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_up_0',trainable=self.TRAIN_UPSAMPLING_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1,3,3,F,F], [1,1,1,1,1], 'VALID', scope_name='conv_up_1',trainable=self.TRAIN_UPSAMPLING_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_up_1',trainable=self.TRAIN_UPSAMPLING_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1,3,3,F,F], [1,1,1,1,1], 'VALID', scope_name='conv_up_2',trainable=self.TRAIN_UPSAMPLING_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_up_2',trainable=self.TRAIN_UPSAMPLING_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1,3,3,F,F], [1,1,1,1,1], 'VALID', scope_name='conv_up_3',trainable=self.TRAIN_UPSAMPLING_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_up_3',trainable=self.TRAIN_UPSAMPLING_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1,3,3,F,F], [1,1,1,1,1], 'VALID', scope_name='conv_up_4',trainable=self.TRAIN_UPSAMPLING_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_up_4',trainable=self.TRAIN_UPSAMPLING_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1,3,3,F,F], [1,1,1,1,1], 'VALID', scope_name='conv_up_5',trainable=self.TRAIN_UPSAMPLING_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_up_5',trainable=self.TRAIN_UPSAMPLING_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1,3,3,F,F], [1,1,1,1,1], 'VALID', scope_name='conv_up_6',trainable=self.TRAIN_UPSAMPLING_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_up_6',trainable=self.TRAIN_UPSAMPLING_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [1,3,3,F,F], [1,1,1,1,1], 'VALID', scope_name='conv_up_7',trainable=self.TRAIN_UPSAMPLING_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_up_7',trainable=self.TRAIN_UPSAMPLING_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        #################### upsampling network end #######################
        
        ####################Registration Filters ###########################
        #we need to skip the first image  because it would be always convolved with the filter with 1 in the center
        #because we actually want that image to stay the same. So we remove it from the computation and it will be concat
        #at the end.
        self.x1_from_upsampling=tf.identity(x1)
        self.x1_to_filter=tf.identity(x1[:,1:,:,:,:])
        
        
        ###Duplicate the reference image multilple times and associate one reference image to each and every
        ### of the 8 other images
        self.references=tf.tile(x1[:,0:1,:,:,:],[1,self.T_in,1,1,1])

        x1=tf.stack([x1,self.references],axis=2)
        self.f1=x1
        sh=tf.shape(x1)
        x1=tf.reshape(x1,[sh[0],sh[1]*sh[2],sh[3],sh[4],sh[5]])
          
        
        #skip the first pair because we already know the filter
        x1=x1[:,2:,:,:,:]
        
        F1=64
        F2=64
        
        t = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [2,3,3,F2,128], [1,2,1,1,1], 'VALID', scope_name='Rconv1b',trainable=self.TRAIN_REGIST_NET)
        t = tf.nn.leaky_relu(t)
        
        t = Conv3D(tf.pad(t, sp, mode='REFLECT'), [1,3,3,128,F1],[1,1,1,1,1], 'VALID', scope_name='Rconv2b',trainable=self.TRAIN_REGIST_NET)
        t = tf.nn.leaky_relu(t)
        
        t = Conv3D(tf.pad(t, sp, mode='REFLECT'), [1,3,3,F1,F1], [1,1,1,1,1], 'VALID', scope_name='Rconv3b',trainable=self.TRAIN_REGIST_NET)
        t = tf.nn.leaky_relu(t)
        
        t = Conv3D(tf.pad(t, sp, mode='REFLECT'), [1,3,3,F1,F1], [1,1,1,1,1], 'VALID', scope_name='Rconv4b',trainable=self.TRAIN_REGIST_NET)
        t = tf.nn.leaky_relu(t)
        
        t = Conv3D(tf.pad(t, sp, mode='REFLECT'), [1,3,3,F1,self.dyn_filter_size**2], [1,1,1,1,1], 'VALID', scope_name='Rconv7_b',trainable=self.TRAIN_REGIST_NET) 
        #[b,9,96,96,81] 
        
        t=tf.reduce_mean(t,axis=[2,3])
        #to compute the accuracy
        self.logits_filters=tf.identity(t)
        
        self.filters = tf.nn.softmax(t, axis=2)
        sh=tf.shape(self.filters)
        self.filters=tf.reshape(self.filters,[sh[0],sh[1],self.dyn_filter_size,self.dyn_filter_size])
        
        #[b,9,dyn_filter_size,dyn_filter_size]
        
        ################ GLOBAL DYNAMIC CONVOLUTION ######################################################################
        #x1_to_filter are the high freq image representations
        self.x_to_fuse=self.registration_dyn_filters(self.x1_to_filter,self.filters)
        self.x_registered=self.registration_dyn_filters(self.x[:,1:,:,:,:],self.filters)
        ######################### END GLOBAL DYNAMIC CONVOLUTION #########################################################
        
        self.x_to_fuse=tf.concat([self.x1_from_upsampling[:,0:1,:,:,:],self.x_to_fuse],axis=1)
        self.x_registered=tf.concat([self.x[:,0:1,:,:,:],self.x_registered],axis=1)
        
    
        ################ FILL MASKED REGION OF FEATURE REPRESENTATIONS COMING FROM SISRNET ###############################
        self.fill_coeff_feature=tf.tile(self.fill_coeff, [1,1,1,1,1,F] )
        #[b,9,9,3*H,3*W,F]
        self.x_to_fuse=tf.reduce_sum(self.fill_coeff_feature[:,:,:,:,:,:]*tf.expand_dims(self.x_to_fuse[:,:,:,:,:],axis=1),axis=2)
        ################ FILL MASKED REGION BICUBIC UPSAMPLED IMAGES###############################
        #[b,9,9,3*H,3*W,F]
        self.x_registered=tf.reduce_sum(self.fill_coeff[:,:,:,:,:,:]*tf.expand_dims(self.x_registered[:,:,:,:,:],axis=1),axis=2)
        
        
        #################  FUSION NET    ##################################
        x1 = Conv3D(tf.pad(self.x_to_fuse, sp, mode='REFLECT'), [3,3,3,F,F1], [1,1,1,1,1], 'VALID', scope_name='conv_fuse_0',trainable=self.TRAIN_FUSION_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_fuse_0',trainable=self.TRAIN_FUSION_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [3,3,3,F1,F1], [1,1,1,1,1], 'VALID', scope_name='conv_fuse_1',trainable=self.TRAIN_FUSION_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_fuse_1',trainable=self.TRAIN_FUSION_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [3,3,3,F1,F1], [1,1,1,1,1], 'VALID', scope_name='conv_fuse_2',trainable=self.TRAIN_FUSION_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_fuse_2',trainable=self.TRAIN_FUSION_NET)
        x1 = tf.nn.leaky_relu(x1)
        
        x1 = Conv3D(tf.pad(x1, sp, mode='REFLECT'), [3,3,3,F1,F1], [1,1,1,1,1], 'VALID', scope_name='conv_fuse_3',trainable=self.TRAIN_FUSION_NET)
        x1=InstanceNorm(x1,axis=[2,3],name='inst_norm_fuse_3',trainable=self.TRAIN_FUSION_NET)
        x1 = tf.nn.leaky_relu(x1)

        self.x1 = Conv3D(x1, [1,1,1,F1,1], [1,1,1,1,1], 'VALID', scope_name='conv_fuse_4',trainable=self.TRAIN_FUSION_NET)
        
        
        #Take the 9 registered images and average them
        self.x_r=tf.reduce_mean(self.x_registered,axis=1)
        self.x_r=tf.expand_dims(self.x_r,axis=1)
        
        #Re-Normalized
        self.x_r=(self.x_r*self.sigma)/self.sigma_rescaled
        
        #Sum to residual
        self.x1+=self.x_r
        
        self.SR_temp=self.x1
        self.logits=self.x1 
        
    def registration_dyn_filters(self,x1_to_filter,filters):
        
        x1_to_filter=tf.identity(x1_to_filter)
        filters=tf.identity(filters)
        
        features_channels=tf.shape(x1_to_filter)[-1]
        depth=tf.shape(x1_to_filter)[1]
        batch=tf.shape(x1_to_filter)[0]
        batch_depth_dim=batch*depth
        
        sh=tf.shape(filters)
        #[b,9,7,7]
        filters=tf.reshape(filters, [sh[0]*sh[1],sh[2],sh[3]])
        #[b*9,7,7]
        #self.filters=tf.expand_dims(self.filters, axis=0)
        filters=tf.expand_dims(filters, axis=3)
        #[b*9,7,7,1]
        filters=tf.transpose(filters, perm=[1,2,0,3])
        #[7,7,b*9,1]
        filters=tf.tile(filters,[1,1,1,features_channels])
        #[7,7,b*9,F]
        sh=tf.shape(filters)
        #[7,7,b*9,F]
        filters=tf.reshape(filters, [sh[0],sh[1],sh[2]*sh[3]])
        filters=tf.expand_dims(filters, axis=-1)
        #[7,7,b*9*F,1]
        
        sh=tf.shape(x1_to_filter)
        #[b,9,96,96,64]
        x1_to_filter=tf.reshape(x1_to_filter,[sh[0]*sh[1],sh[2],sh[3],sh[4]])
        #[b*9,96,96,64]
        #self.x1_to_filter=tf.expand_dims(self.x1_to_filter,axis=4)
        
        x1_to_filter=tf.transpose(x1_to_filter,perm=[1,2,0,3])
        #[96,96,b*9,64]
        sh=tf.shape(x1_to_filter)
        x1_to_filter=tf.reshape(x1_to_filter,[sh[0],sh[1],sh[2]*sh[3]])
        x1_to_filter=tf.expand_dims(x1_to_filter,axis=0)
        #[1,96,96,b*9*64]
        
        pad_size=int((self.dyn_filter_size-1)/2)
        padding_reg=[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]]
        x_to_fuse=tf.nn.depthwise_conv2d(tf.pad(x1_to_filter, padding_reg, mode='REFLECT'),filters,[1,1,1,1],"VALID")
        #[1,96,96,b*9*64]
        sh=tf.shape(x_to_fuse)
        x_to_fuse=tf.reshape(x_to_fuse,[sh[0],sh[1],sh[2],batch_depth_dim,features_channels])
        sh=tf.shape(x_to_fuse)
        x_to_fuse=tf.reshape(x_to_fuse,[sh[0],sh[1],sh[2],batch,depth,sh[4]])
        x_to_fuse=tf.squeeze(x_to_fuse,axis=0)
        #[96,96,b,9,F]
        x_to_fuse=tf.transpose(x_to_fuse,perm=[2,3,0,1,4])
        #[b,9,96,96,64]
        
        return x_to_fuse
        
    
    def PSNR(self,norm=True):
        
        self.y_hat=self.logits
        
        y_hat=((self.y_hat)*self.sigma_rescaled)+self.mu
        y=((self.y)*self.sigma_rescaled)+self.mu
        
        
            
        s1=tf.shape(y)
        s2=tf.shape(y_hat)
        labels=tf.reshape(y,shape=[s1[0],s1[2],s1[3],s1[4]])
        predictions=tf.reshape(y_hat,shape=[s2[0],s2[2],s2[3],s2[4]])
            
        
        
        #crop the center of the reconstructed HR image
        size_image=tf.shape(predictions)[1]
        cropped_predictions=predictions[:,self.border:size_image-self.border,self.border:size_image-self.border]
        
        #All mse
        X=[]
        for i in range((2*self.border)+1):
            for j in range((2*self.border)+1):
                cropped_labels=labels[:,i:i+(size_image-(2*self.border)),j:j+(size_image-(2*self.border))]
                cropped_mask_y=self.mask_y[:,:,i:i+(size_image-(2*self.border)),j:j+(size_image-(2*self.border))]
                
                cropped_predictions_masked=cropped_predictions*tf.squeeze(cropped_mask_y,axis=1)
                cropped_labels_masked=cropped_labels*tf.squeeze(cropped_mask_y,axis=1)
        
                #bias brightness
                b=(1.0/tf.reduce_sum(cropped_mask_y,axis=[2,3,4]))*tf.reduce_sum(cropped_labels_masked-cropped_predictions_masked,axis=[1,2])
                b=tf.reshape(b,[s1[0],1,1,1])
                corrected_cropped_predictions=cropped_predictions_masked+b
                corrected_cropped_predictions=corrected_cropped_predictions*tf.squeeze(cropped_mask_y,axis=1)
                corrected_mse=(1.0/tf.reduce_sum(cropped_mask_y,axis=[2,3,4]))*tf.reduce_sum(tf.square(cropped_labels_masked-corrected_cropped_predictions),axis=[1,2])
                    
                cPSNR=10*tf.log((65535**2)/corrected_mse)/tf.log(10.0)
                X.append(cPSNR) 
                

        X=tf.stack(X)
        
        if norm:
            max_cPSNR=tf.reduce_max(X,axis=0)
            
            score=self.norm_baseline/max_cPSNR
            score=tf.reduce_mean(score)
            return score
        else:
            max_cPSNR=tf.reduce_max(X,axis=0)
            psnr=tf.reduce_mean(max_cPSNR)
            return psnr
     
    
    def summary(self):
        '''
        Create summaries to write on TensorBoard
        '''
        with tf.name_scope('performance') as scope:
            tf.summary.scalar('loss', self.loss, collections=['loss'])
            #PSNR
            tf.summary.scalar('score',self.score, collections=['metric'])#psnr normalized
            psnr=tf.summary.scalar('psnr',self.psnr, collections=['metric'])
            
        
        self.summary_loss=tf.summary.merge_all(key='loss')
        self.summary_metric=tf.summary.merge_all(key='metric')
        self.summary_psnr_metric=tf.summary.merge([psnr])
        
        self.summary_op = tf.summary.merge_all()
        
        
        #Images on tensorboard
        with tf.name_scope('images') as scope:
            #tf.summary.image('images_residual', tf.reshape(self.Rx, [-1, self.patch_size_HR, self.patch_size_HR, 1]), 3,collections=['images'])
            tf.summary.image('images_temp_SR', tf.reshape(self.SR_temp, [-1, tf.shape(self.SR_temp)[2], tf.shape(self.SR_temp)[3], 1]), 3,collections=['images'])
          
        with tf.name_scope('filters') as scope:
            filters=tf.reshape(self.filters,[-1,self.dyn_filter_size*8,self.dyn_filter_size])
            filters=tf.expand_dims(filters,axis=-1)
            tf.summary.image('filters',filters, 3,collections=['images'])



        # Merge all summaries related to images collection
        self.tf_images_summaries = tf.summary.merge_all(key='images')  
        
        
        
    def train_one_epoch(self,saver,train_writer,test_writer,epoch,step):
        start_time = time.time()
        n_batches=0
        total_loss=0
        
        #at every epoch start I need a NEW GENERATOR to load the training set npy
        #Which dataset to load
        if self.full:
            self.gen=load_dataset(self.dataset_path,self.n_chunks,self.spectral_band,num_images=9,how='best')
        else:
            self.gen=load_dataset_best9(self.dataset_path,self.n_chunks,band=self.spectral_band)
            
        while True:
            
            val=self.get_data()
            
            if val==0:
                print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
                print('Took: {0} seconds'.format(time.time() - start_time))
                return step  
        
        
            #shuffle
            self.batch_training_norm,\
            self.batch_training_mask,\
            self.batch_training_y_norm,\
            self.batch_mask_train_y,\
            self.shifts,\
            self.y_filters_dyn,\
            self.fill_coeff_train= shuffle(self.batch_training_norm,
                                        self.batch_training_mask,
                                        self.batch_training_y_norm,
                                        self.batch_mask_train_y,
                                        self.shifts,
                                        self.y_filters_dyn,
                                        self.fill_coeff_train
                                    )  
            
            
            
            
            
            for i in range(1,int(self.batch_training_norm.shape[0]/self.batch_size)+1):
                
                
                #Run session to compute summaries on test every 5 steps
                if (step+1)%200 == 0:
                    
                    
                    _,l,summaries,summary_psnr_metric=self.sess.run([self.opt,self.loss,self.summary_loss,self.summary_psnr_metric],feed_dict={
                                                                                               self.y_filters:self.y_filters_dyn[(i-1)*self.batch_size:i*self.batch_size],
                                                                                               self.y:self.batch_training_y_norm[(i-1)*self.batch_size:i*self.batch_size],
                                                                                               self.x:self.batch_training_norm[(i-1)*self.batch_size:i*self.batch_size],
                                                                                               self.fill_coeff:self.fill_coeff_train[(i-1)*self.batch_size:i*self.batch_size],
                                                                                               self.mask_y:self.batch_mask_train_y[(i-1)*self.batch_size:i*self.batch_size]
                                                                                            }
                                                                  )
                
                    
                    train_writer.add_summary(summaries, global_step=step)
                    train_writer.add_summary(summary_psnr_metric, global_step=step)
                    train_writer.flush()
                
                else:
                    _,l=self.sess.run([self.opt,self.loss],feed_dict={self.y:self.batch_training_y_norm[(i-1)*self.batch_size:i*self.batch_size],
                                                                  self.y_filters:self.y_filters_dyn[(i-1)*self.batch_size:i*self.batch_size],                            
                                                                  self.x:self.batch_training_norm[(i-1)*self.batch_size:i*self.batch_size],
                                                                  self.fill_coeff:self.fill_coeff_train[(i-1)*self.batch_size:i*self.batch_size],
                                                                  self.mask_y:self.batch_mask_train_y[(i-1)*self.batch_size:i*self.batch_size]
                                                                                              }
                                  
                                 )
                
                
                if (step+1)%self.skip_step==0:
                    
                    saver.save(self.sess, 'checkpoints/'+self.tensorboard_dir+'/'+'model.ckpt', step)
                    
                    print('Training Loss for a mini batch at step {0}: {1}'.format(step, l))
                    ##Evaluate model against test
                    self.eval_once(test_writer,epoch,step)
                    
                    #View images on tensorboard
                    
                    
                    #images_summaries=self.sess.run(self.tf_images_summaries,feed_dict={
                    #                                        self.x:self.batch_validation_norm[[0,5]],
                    #                                        self.fill_coeff:self.fill_coeff_valid[[0,5]],      
                    #                                        self.is_train:False})
                    #train_writer.add_summary(images_summaries, global_step=step)
                    
                
                
                total_loss+=l
                n_batches+=1
                step+=1
        
        
        
        
    def train(self,n_epochs):
        
        safe_mkdir('checkpoints')
        safe_mkdir('checkpoints/'+self.tensorboard_dir)
        #To plot two different curves on the same graph we need two different writers that write the
        #same group of summaries.
        train_writer = tf.summary.FileWriter('./graphs/'+self.tensorboard_dir + '/train', tf.get_default_graph())
        test_writer = tf.summary.FileWriter('./graphs/'+self.tensorboard_dir + '/test',tf.get_default_graph())
        self.sess.run(tf.global_variables_initializer())
        
        
        
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'+self.tensorboard_dir+'/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            #Restore pre-trained registration layers
            dir_checkpoints_registr_net=self.RegNet_pretrain_dir
            
            saver_regist_net = tf.train.Saver({
                                    'Rconv1b/W':tf.get_default_graph().get_tensor_by_name('Rconv1b/W:0'),
                                    'Rconv2b/W':tf.get_default_graph().get_tensor_by_name('Rconv2b/W:0'),
                                    'Rconv3b/W':tf.get_default_graph().get_tensor_by_name('Rconv3b/W:0'),
                                    'Rconv4b/W':tf.get_default_graph().get_tensor_by_name('Rconv4b/W:0'),
                                    'Rconv7_b/W':tf.get_default_graph().get_tensor_by_name('Rconv7_b/W:0'),
                                    'Rconv1b/b':tf.get_default_graph().get_tensor_by_name('Rconv1b/b:0'),
                                    'Rconv2b/b':tf.get_default_graph().get_tensor_by_name('Rconv2b/b:0'),
                                    'Rconv3b/b':tf.get_default_graph().get_tensor_by_name('Rconv3b/b:0'),
                                    'Rconv4b/b':tf.get_default_graph().get_tensor_by_name('Rconv4b/b:0'),
                                    'Rconv7_b/b':tf.get_default_graph().get_tensor_by_name('Rconv7_b/b:0'),
                                    
                                     
                                   
                                   })
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(dir_checkpoints_registr_net+'/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver_regist_net.restore(self.sess, ckpt.model_checkpoint_path)
            
            #Restore pre-trained  upsampling network 
            dir_checkpoints_upsampling_net=self.SISRNet_pretrain_dir
            
            
            conv_W={'conv_up_{0}/W'.format(i):tf.get_default_graph().get_tensor_by_name('conv_up_{0}/W:0'.format(i)) for i in range(8)}
            conv_b={'conv_up_{0}/b'.format(i):tf.get_default_graph().get_tensor_by_name('conv_up_{0}/b:0'.format(i)) for i in range(8)}
            in_beta={'inst_norm_up_{0}/beta'.format(i):tf.get_default_graph().get_tensor_by_name('inst_norm_up_{0}/beta:0'.format(i)) for i in range(8)}
            in_gamma={'inst_norm_up_{0}/gamma'.format(i):tf.get_default_graph().get_tensor_by_name('inst_norm_up_{0}/gamma:0'.format(i)) for i in range(8)}
            
            
            all_weights={**conv_W,**conv_b,**in_beta,**in_gamma}
            
            saver_upsampling_net = tf.train.Saver(all_weights)
            
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(dir_checkpoints_upsampling_net+'/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver_upsampling_net.restore(self.sess, ckpt.model_checkpoint_path)
        
        
            
        step = self.gstep.eval(session=self.sess)
        
        for epoch in range(n_epochs):
            step = self.train_one_epoch(saver, train_writer,test_writer, epoch, step)
            
        
        return step
    
    
    def eval(self):
        '''
        Compute quality evaluation metric PSNR and PSNR normalized
        '''
        with tf.name_scope('psnr'):
            self.score=self.PSNR(norm=True)
            self.psnr=self.PSNR(norm=False)
            
        with tf.name_scope('accuracy'):
            #metrics
            correct=tf.equal(tf.argmax(tf.nn.softmax(self.logits_filters,axis=2),axis=2),tf.argmax(self.y_filters,axis=2))
            self.accuracy=tf.reduce_mean(tf.cast(correct,'float'))
    
            
    def eval_once(self, writer, epoch, step):
        start_time = time.time()
        
        score_list=[]
        psnr_list=[]
        loss_list=[]
        accuracy_list=[]
        
        
        val_batch_size=1
        for i in range(1,int(self.batch_validation_norm.shape[0]/val_batch_size)+1):
            
            score, psnr, loss, accuracy = self.sess.run([self.score, self.psnr, self.loss,self.accuracy],
                                           feed_dict={
                                                      self.y_filters:self.y_filters_valid_dyn[(i-1)*val_batch_size:i*val_batch_size],
                                                      self.y:self.batch_validation_y_norm[(i-1)*val_batch_size:i*val_batch_size],
                                                      self.x:self.batch_validation_norm[(i-1)*val_batch_size:i*val_batch_size],
                                                      self.fill_coeff:self.fill_coeff_valid[(i-1)*val_batch_size:i*val_batch_size],
                                                      self.mask_y:self.batch_mask_valid_y[(i-1)*val_batch_size:i*val_batch_size],
                                                      self.norm_baseline:self.norm_validation[(i-1)*val_batch_size:i*val_batch_size]
                                                     })
            psnr_list.append(psnr)
            score_list.append(score)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
        
        
        average_score=np.mean(score_list)
        average_psnr=np.mean(psnr_list)
        average_loss=np.mean(loss_list)
        average_accuracy=np.mean(accuracy_list)
        
        # Create a new Summary object with your measure
        summary_average_score = tf.Summary()
        summary_average_score.value.add(tag="performance/score", simple_value=average_score)
        
        summary_average_psnr = tf.Summary()
        summary_average_psnr.value.add(tag="performance/psnr", simple_value=average_psnr)
        
        summary_average_loss = tf.Summary()
        summary_average_loss.value.add(tag="performance/loss", simple_value=average_loss)
        
        writer.add_summary(summary_average_score, global_step=step)
        writer.add_summary(summary_average_psnr, global_step=step)
        writer.add_summary(summary_average_loss, global_step=step)
        
        writer.flush()
            
        print('Score on test batch at epoch {0}: {1} '.format(epoch, average_score))
        print('Accuracy on test batch at epoch {0}: {1} '.format(epoch, average_accuracy))
        
        print('Took: {0} seconds'.format(time.time() - start_time))

        
    def predict_test(self,test_dir,n_slide):
        '''
        test_dir: directory from where to load the testset 
        n_slide: upper bound to the number of predictions with the images in each imageset
        '''
        
        ##Retrieve all testset in order to apply the sliding window
        input_images_LR_test=np.load(os.path.join(test_dir,'dataset_{0}_LR_test.npy'.format(self.spectral_band)),allow_pickle=True)
        mask_LR_test=np.load(os.path.join(test_dir,'dataset_{0}_mask_LR_test.npy'.format(self.spectral_band)),allow_pickle=True)
        shift_LR_test=np.load(os.path.join(test_dir,'shifts_test_{0}.npy'.format(self.spectral_band)),allow_pickle=True)
        
        #Restore the session from checkpoint
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/'+self.tensorboard_dir+'/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        #Order by mask
        indexes=[]
        
        for image_set in mask_LR_test:
            indexes.append(np.argsort(np.sum(np.array(image_set[0:]),axis=(1,2)))[::-1])
            
        input_images_LR_test=[image_set[indexes_set] for image_set,indexes_set in zip(input_images_LR_test,indexes)]
        mask_LR_test=np.array([image_set[indexes_set] for image_set,indexes_set in zip(mask_LR_test,indexes)])
        #####
        
        
        
        #val_batch_size=1
        SR_images=np.zeros([len(input_images_LR_test),1,384,384,1])
        for m in range(0,len(input_images_LR_test)):
        
            imageset=np.array(input_images_LR_test[m])
            imageset_mask=np.array(mask_LR_test[m])
            imageset_shift=np.array(shift_LR_test[m])
            
            #filter some images based on the mask
            #print(imageset.shape)
            percentage=0.9
            while True:
                indexes_0=np.argwhere((np.sum(imageset_mask[0:],axis=(1,2))/(384*384))>percentage).squeeze(axis=1)
                indexes_0=indexes_0 if indexes_0.ndim>0 else np.array([]) 
                
                
                indexes=np.array(list(indexes_0))
                if indexes.size>=9:
                    imageset=imageset[indexes]
                    #print(imageset.shape)
                    imageset_mask=imageset_mask[indexes]
                    imageset_shift=imageset_shift[indexes]
                    
                    break
                else:
                    
                    percentage-=0.05
                    continue
        
        
            
            #imageset_HR=input_images_HR_valid[i]
            #imageset_mask_HR=mask_HR_valid[i]
            #Maybe HERE WE CAN REMOVE VERY BAD LR IMAGES
            len_imageset=np.shape(imageset)[0]
            
            temporal_dim=9
            upper_bound=n_slide
            if len_imageset-temporal_dim+1>upper_bound:
                size=upper_bound+1
            else:
                size=len_imageset-temporal_dim+1
                
            SR_imageset=np.zeros([size,1,384,384,1])
            for n in range(0,size):
                
                imageset_9=imageset[n:n+temporal_dim]
                #print(imageset_9.shape)
                #imageset_9=np.concatenate([np.expand_dims(reference_image,axis=0),imageset_9])
                imageset_9=np.expand_dims(imageset_9,axis=0)
                imageset_9=np.expand_dims(imageset_9,axis=-1)
                
                imageset_9_mask=imageset_mask[n:n+temporal_dim]
                #imageset_9_mask=np.concatenate([np.expand_dims(reference_mask,axis=0),imageset_9_mask])
                imageset_9_mask=np.expand_dims(imageset_9_mask,axis=0)
                imageset_9_mask=np.expand_dims(imageset_9_mask,axis=-1)
                
                ########################Register the mask #############
                imageset_9_mask=np.round(imageset_9_mask)
                imageset_9_mask=imageset_9_mask.astype('bool')
                
                for j in range(imageset_9_mask.shape[1]):
                    shifted_mask=imageset_9_mask[:,j]
                    corrected_mask = fourier_shift(np.fft.fftn(shifted_mask.squeeze()), imageset_shift[j])
                    corrected_mask = np.fft.ifftn(corrected_mask)
                    corrected_mask = corrected_mask.reshape([1,np.shape(corrected_mask)[0],np.shape(corrected_mask)[1],1])
                    imageset_9_mask[:,j]=np.round(corrected_mask)
                    
                ##############Compute coefficients for filling images where masked
                sh=imageset_9_mask.shape
                fill_coeff_test=np.ones([sh[0],sh[1],sh[1],sh[2],sh[3],sh[4]],dtype='bool')
                for i in range(0,9):
                    fill_coeff_test[:,:,i]=np.expand_dims(imageset_9_mask[:,i],axis=1)
        
                for i in range(0,9):
                    for j in range(i+1,9):
                        rows_indexes=[k for k in range(0,9) if k!=(j)]
                        #print(rows_indexes)
                        fill_coeff_test[:,rows_indexes,j]=fill_coeff_test[:,rows_indexes,j]*np.expand_dims(1-imageset_9_mask[:,i],axis=1)
                
                for i in range(1,9):
                    fill_coeff_test[:,i,0:i]=fill_coeff_test[:,i,0:i]*np.expand_dims(1-imageset_9_mask[:,i],axis=1)
                
                #We need to fill in the regions where all the masks are zero. In this case we decide to uncover the hidden regions of
                #the considered image by turning the mask to 1 in those regions.
                f=np.sum(fill_coeff_test,axis=2)
                #[b,9,W,H,1]
                fill_coeff_test[:,range(9),range(9),:,:,:]=fill_coeff_test[:,range(9),range(9),:,:,:]+np.logical_not(f)[:,range(9),:,:,:]
                    
                ####################
                
                
                imageset_9=(imageset_9-self.mu)/self.sigma
                
                SR_image=self.sess.run(self.logits,feed_dict={
                                                              self.x:imageset_9,
                                                              self.fill_coeff:fill_coeff_test
                                                            })
                
                
                
                
                SR_imageset[n]=SR_image
                
                #self.SR=SR_imageset
                #sys.exit()
                
            #Register all images in SR_imageset with respect to the first
            #Compute the shift
            #SR_imageset_registered=np.zeros_like(SR_imageset)
            SR_imageset_registered=np.empty([0,1,384,384,1])
            for z in range(SR_imageset.shape[0]):
                #we consider the first image in the set as the reference image
                reference_image=SR_imageset[0]
                shifted_image=SR_imageset[z]
                
                shift, error, diffphase = register_translation(reference_image.squeeze(), shifted_image.squeeze(),upsample_factor=1)
                if (np.abs(shift)>4).any():
                    print('Skip image...too large shifts')
                    print(shift)
                    continue
                
                ###Image
                #shift is applied to the original image from the batch_training variable, in the fourier domain
                corrected_image = fourier_shift(np.fft.fftn(shifted_image.squeeze()), shift)
                corrected_image = np.fft.ifftn(corrected_image)
                corrected_image = corrected_image.reshape([1,1,384,384,1])
                #SR_imageset_registered[z]=corrected_image
                SR_imageset_registered=np.append(SR_imageset_registered,corrected_image,axis=0)   
                  
            SR_image=np.mean(SR_imageset_registered,axis=0,keepdims=True)
            SR_images[m]=SR_image
            
            print('Image number {0}'.format(m))
        #de-normalize
        SR_images=(SR_images*self.sigma_rescaled)+self.mu
         
        return SR_images
    
    
    
    def build(self):
        '''
        Build the computation graph
        '''
        
        #self.get_data()
        self.inference_FR()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()


