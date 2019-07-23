import random
import numpy as np
from collections import defaultdict
import progressbar

dataset_patch_dir='/home/bordone/Superresolution/1_patch_dataset_pickles/'
dataset_dir='/home/bordone/Superresolution/0_dataset_pickles/'

def new_coordinate(shape_original=[128,128],patch_size=[32,32]):
    image_size = shape_original
    portion_size = patch_size

    x1 = random.randint(0, image_size[0]-portion_size[0]-1)
    y1 = random.randint(0, image_size[1]-portion_size[1]-1)

    x2, y2 = x1+portion_size[0], y1+portion_size[1]
    
    return (x1,y1),(x2,y2)

########################################################################################################
###################Load 'top' LR images from complete LR dataset###########################################
def load_training_best(input_images_LR,mask_LR,top=9):   
    '''
    Load the training set and the clearance images for the top n most clear LR images
    '''
    #Find indexes of best LR images per image set
    indexes=[np.argsort(np.sum(np.array(image_set),axis=(1,2)))[::-1][0:top] for image_set in mask_LR]
    
    
    batch_training=np.array([np.array(l)[indexes[i]] for i,l in enumerate(input_images_LR)])
    sh=batch_training.shape
    batch_training=batch_training.reshape([-1,sh[1],sh[2],sh[3],1])
    
    
    batch_training_mask=np.array([np.array(l)[indexes[i]] for i,l in enumerate(mask_LR)])
    sh=batch_training_mask.shape
    batch_training_mask=batch_training_mask.reshape([-1,sh[1],sh[2],sh[3],1])

    return batch_training,batch_training_mask

def load_training_random(input_images_LR,mask_LR,num_images=9):   
    '''
    Load the training set and the clearance images for the n random LR images
    '''
    #Find indexes of best LR images per image set
    indexes=[random.sample(list(range(0,len(image_set))),num_images) for image_set in mask_LR]
    
    
    batch_training=np.array([np.array(l)[indexes[i]] for i,l in enumerate(input_images_LR)])
    sh=batch_training.shape
    batch_training=batch_training.reshape([-1,sh[1],sh[2],sh[3],1])
    
    
    batch_training_mask=np.array([np.array(l)[indexes[i]] for i,l in enumerate(mask_LR)])
    sh=batch_training_mask.shape
    batch_training_mask=batch_training_mask.reshape([-1,sh[1],sh[2],sh[3],1])

    return batch_training,batch_training_mask

def load_training_first(input_images_LR,mask_LR,top=9):   
    '''
    Load the training set and the clearance images for the first n LR images
    '''
    batch_training=np.array([np.array(l)[0:top] for i,l in enumerate(input_images_LR)])
    sh=batch_training.shape
    batch_training=batch_training.reshape([-1,sh[1],sh[2],sh[3],1])
    
    
    batch_training_mask=np.array([np.array(l)[0:top] for i,l in enumerate(mask_LR)])
    sh=batch_training_mask.shape
    batch_training_mask=batch_training_mask.reshape([-1,sh[1],sh[2],sh[3],1])

    return batch_training,batch_training_mask
    
########################################################################################################

########################################################################################################
###############    Methods to create Patches in dataset construction phase #############################

def create_patch_dataset_return_shifts(input_images_upsample,input_images_HR,mask_upsample,mask_HR,shifts,patch_size=96,num_patches_per_set=100,scale=3,smart_patching=False):
    '''
        Extract patch from low resolution images and return the shifts. These shifts come from 
        the registration that you do on the upsampled version of these LR images and you want to replicate the shifts for the times
        you extract a patch from one imageset. This because if even the number of patches per imageset is decided by the user
        it is not sure we actually reach that maximum. Maybe we reach first max_trial limit
        
        If scale is 3 it means that we are patching LR images and so we need to scale up the coordinates to patch the 
        corresponding HR image.
        If scale is 1 we are patching pre-upsampled images.
        
        smart_patching=True means that we accept a patch coordinate if at least 9 of the patches are 70% cleared. If false all of them have to be 70 % cleared 
    '''
    #[b,x,384,384] squeeze the input you pass to this function
    
    input_images_upsample_patch=[]
    input_images_HR_patch=[]
    mask_upsample_patch=[]
    mask_HR_patch=[]
    coordinates=[]
    shifts_patch=[]
    
    tot_num_patches=num_patches_per_set
    #max number of trial to reach the extraction of 50 patches from one image set.
    #since we want the patch to have not less than 60% of clear pixels and the corresponding HR image patch
    #not less than 75% of clear pixels.
    max_trial=100000
    scale=scale
    n_samples=len(input_images_upsample)
    
    shape_original=[input_images_upsample[0][0].shape[0],input_images_upsample[0][0].shape[1]]
    bar = progressbar.ProgressBar(maxval=n_samples, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    
    i=0
    for j in range(n_samples):
        
        image_set=input_images_upsample[j]
        mask_set=mask_upsample[j]
        #######To array
        upsample_images_set=np.array(image_set)
        upsample_mask_set=np.array(mask_set)
        ###############
        
        image_HR=input_images_HR[j]
        mask_image_HR=mask_HR[j]
        
        current_num_patches=0
        num_trial=0
        
        
        coordinates_for_one_set=[]
        
        
        while True:
            if current_num_patches>=tot_num_patches or num_trial>=max_trial:
                #i+=1
                #print(i)
                break
            

            
            x,y=new_coordinate(shape_original=shape_original,patch_size=[patch_size,patch_size])
            num_trial+=1
            patches_upsample=upsample_images_set[:,x[0]:y[0],x[1]:y[1]]
            patches_HR=image_HR[x[0]*scale:y[0]*scale,x[1]*scale:y[1]*scale]
            
            #check that each upsample patch in the set is at least 70% clear 

            patch_masks_upsample=upsample_mask_set[:,x[0]:y[0],x[1]:y[1]]
            
            checked_upsample=[(np.sum(patch_mask)/(patch_size**2))>0.70 for patch_mask in patch_masks_upsample]
            
            #check that each HR patch in the set is at least 90% clear 
            patch_masks_HR=mask_image_HR[x[0]*scale:y[0]*scale,x[1]*scale:y[1]*scale]
            
            checked_HR=((np.sum(patch_masks_HR)/((patch_size*scale)**2))>0.85 )
            
            if smart_patching:
                checked_LR=(sum(checked_upsample)>=9)
            else:
                checked_LR=all(checked_upsample)
            
            if  checked_LR and checked_HR:
                input_images_upsample_patch.append(patches_upsample)
                input_images_HR_patch.append(patches_HR)
                mask_upsample_patch.append(patch_masks_upsample)
                mask_HR_patch.append(patch_masks_HR)
                coordinates_for_one_set.append((x,y))
                shifts_patch.append(np.copy(shifts[j]))
                current_num_patches+=1
        
        coordinates.append(coordinates_for_one_set)
        
        
        bar.update(j+1)
    
    bar.finish() 
    dataset_patch=defaultdict()
    dataset_patch['training_patch']=np.array(input_images_upsample_patch)
    dataset_patch['training_mask_patch']=np.array(mask_upsample_patch)
    dataset_patch['training_y_patch']=np.array(input_images_HR_patch)
    dataset_patch['training_mask_y_patch']=np.array(mask_HR_patch)
    dataset_patch['shifts']=shifts_patch
    dataset_patch['coordinates']=coordinates
    
    return  dataset_patch


#high memory usage
def load_dataset(path,n_chuncks,band='NIR',num_images=9,how='best'):
    '''
    This is a generator. You can have multiple chuncks composing your dataset and it can be retrieved by the following generator chucnk by chunck
    '''
    #load validation once
    input_images_LR_valid=np.load(path+'dataset_{0}_LR_valid.npy'.format(band),allow_pickle=True)
    input_images_HR_valid=np.load(path+'dataset_{0}_HR_valid.npy'.format(band),allow_pickle=True)
    mask_LR_valid=np.load(path+'dataset_{0}_mask_LR_valid.npy'.format(band),allow_pickle=True)
    mask_HR_valid=np.load(path+'dataset_{0}_mask_HR_valid.npy'.format(band),allow_pickle=True)
    
    shifts_valid=np.load(path+'shifts_valid_{0}.npy'.format(band),allow_pickle=True)
    
    #the validation set is treated in the original size
    #Validation set
    
    if how=='first':
        batch_validation,batch_validation_mask=load_training_first(input_images_LR_valid,mask_LR_valid,top=9)
    elif how=='best':
       
        #the first image needs to remain in the first position because the prevomputed shifts refer to it
        indexes=[np.argsort(np.sum(np.array(image_set[1:]),axis=(1,2)))[::-1][0:8]+1 for image_set in mask_LR_valid]
        indexes=[np.append(0,indexes_imageset) for indexes_imageset in indexes]
                
        batch_validation=np.array([image_set[indexes_set] for image_set,indexes_set in zip(input_images_LR_valid,indexes)])
        batch_validation=np.expand_dims(batch_validation,axis=-1)
        
        batch_validation_mask=np.array([image_set[indexes_set] for image_set,indexes_set in zip(mask_LR_valid,indexes)])
        batch_validation_mask=np.expand_dims(batch_validation_mask,axis=-1)
        
        shifts_valid=np.array([shifts_set[indexes_set] for shifts_set,indexes_set in zip(shifts_valid,indexes)])
        #These shifts are the shifts to be done to align the images and not the relative shifts, so let's reverse
        shifts_valid=-shifts_valid
        
    
    
    #Reshape
    sh=input_images_HR_valid.shape
    batch_validation_y=input_images_HR_valid.reshape([-1,1,sh[1],sh[2],1])
    
    #reshape
    sh=mask_HR_valid.shape
    batch_mask_y_valid=mask_HR_valid.reshape([-1,1,sh[1],sh[2],1])
    
    
    #normalization
    norm_validation=np.load(path+'norm_'+band+'.npy')
   

    dataset=defaultdict()

    dataset['validation']=batch_validation
    dataset['validation_mask']=batch_validation_mask
    dataset['validation_y']=batch_validation_y
    dataset['validation_mask_y']=batch_mask_y_valid
    
    dataset['shifts_valid']=shifts_valid
    dataset['norm_validation']=norm_validation
    
    
    pickle_indexes=np.array([i for i in range(0,n_chuncks)])
    np.random.shuffle(pickle_indexes)
    
    for i in pickle_indexes:
    
        input_images_LR_patch=np.load(path+'{0}_dataset_{1}_patch_LR.npy'.format(i,band),allow_pickle=True)
        input_images_HR_patch=np.load(path+'{0}_dataset_{1}_patch_HR.npy'.format(i,band),allow_pickle=True)
        mask_LR_patch=np.load(path+'{0}_dataset_{1}_patch_mask_LR.npy'.format(i,band),allow_pickle=True)
        mask_HR_patch=np.load(path+'{0}_dataset_{1}_patch_mask_HR.npy'.format(i,band),allow_pickle=True)
        shifts=np.load(path+'{0}_shifts_patch_{1}.npy'.format(i,band))
        
        
        
        
        #Take 9 random images in the image set
        if how=='first':
            batch_training,batch_training_mask=load_training_first(input_images_LR_patch,mask_LR_patch,top=num_images)
        elif how=='best':
            #Find indexes of best LR images per image set
            indexes=[np.argsort(np.sum(np.array(image_set[1:]),axis=(1,2)))[::-1][0:8]+1 for image_set in mask_LR_patch]
            indexes=[np.append(0,indexes_imageset) for indexes_imageset in indexes]
            
            #images
            batch_training=np.array([image_set[indexes_set] for image_set,indexes_set in zip(input_images_LR_patch,indexes)])
            batch_training=np.expand_dims(batch_training,axis=-1)
            #masks
            batch_training_mask=np.array([image_set[indexes_set] for image_set,indexes_set in zip(mask_LR_patch,indexes)])
            batch_training_mask=np.expand_dims(batch_training_mask,axis=-1)
            #ahifts
            shifts=np.array([shifts_set[indexes_set] for shifts_set,indexes_set in zip(shifts,indexes)])
            #These shifts are the shifts to be done to align the images and not the relative shifts, so let's reverse
            shifts=-shifts
        
        
        #Reshape
        sh=input_images_HR_patch.shape
        batch_training_y=input_images_HR_patch.reshape([-1,1,sh[1],sh[2],1])
        sh=mask_HR_patch.shape
        batch_mask_y_train=mask_HR_patch.reshape([-1,1,sh[1],sh[2],1])
        
        dataset['training']=batch_training
        dataset['training_mask']=batch_training_mask
        dataset['training_y']=batch_training_y
        dataset['training_mask_y']=batch_mask_y_train
        dataset['shifts']=shifts
        
        #batch_training,batch_training_mask,batch_training_y,batch_mask_y
    
        yield dataset
    
#low memory usage: here we retrieve a dataset where the best 9 images have been prefetched and so we have pickles
#with a numpy array and not a list of numpy arrays.
def load_dataset_best9(path,n_chuncks,band='NIR'):
    
    
    '''
    
    '''
    
    
    batch_validation=np.load(path+'dataset_{0}_LR_valid_best9.npy'.format(band),allow_pickle=True)
    batch_validation_y=np.load(path+'dataset_{0}_HR_valid_best9.npy'.format(band),allow_pickle=True)
    batch_validation_mask=np.load(path+'dataset_{0}_mask_LR_valid_best9.npy'.format(band),allow_pickle=True)
    batch_mask_y_valid=np.load(path+'dataset_{0}_mask_HR_valid_best9.npy'.format(band),allow_pickle=True)
    
    shifts_valid=np.load(path+'shifts_valid_{0}_best9.npy'.format(band),allow_pickle=True)
    
    
    
    #normalization
    norm_validation=np.load(path+'norm_'+band+'.npy',allow_pickle=True)
   

    dataset=defaultdict()

    dataset['validation']=batch_validation
    dataset['validation_mask']=batch_validation_mask
    dataset['validation_y']=batch_validation_y
    dataset['validation_mask_y']=batch_mask_y_valid
    
    dataset['shifts_valid']=shifts_valid
    dataset['norm_validation']=norm_validation
    
    
    pickle_indexes=np.array([i for i in range(0,n_chuncks)])
    np.random.shuffle(pickle_indexes)
    
    for i in pickle_indexes:
    
        batch_training=np.load(path+'{0}_dataset_{1}_patch_LR_best9.npy'.format(i,band),allow_pickle=True)
        batch_training_y=np.load(path+'{0}_dataset_{1}_patch_HR_best9.npy'.format(i,band),allow_pickle=True)
        batch_training_mask=np.load(path+'{0}_dataset_{1}_patch_mask_LR_best9.npy'.format(i,band),allow_pickle=True)
        batch_mask_y_train=np.load(path+'{0}_dataset_{1}_patch_mask_HR_best9.npy'.format(i,band),allow_pickle=True)
        shifts=np.load(path+'{0}_shifts_patch_{1}_best9.npy'.format(i,band),allow_pickle=True)
        
        dataset['training']=batch_training
        dataset['training_mask']=batch_training_mask
        dataset['training_y']=batch_training_y
        dataset['training_mask_y']=batch_mask_y_train
        dataset['shifts']=shifts
        
        #batch_training,batch_training_mask,batch_training_y,batch_mask_y
    
        yield dataset


def load_testset_preprocesses(path,band='NIR',how='best',num_images=9):
    
    input_images_LR_test=np.load(path+'dataset_{0}_LR_test.npy'.format(band),allow_pickle=True)
    mask_LR_test=np.load(path+'dataset_{0}_mask_LR_test.npy'.format(band),allow_pickle=True)
    shifts_test=np.load(path+'shifts_test_{0}.npy'.format(band),allow_pickle=True)
    
    if how=='first':
        batch_test,batch_test_mask=load_training_first(input_images_LR_test,mask_LR_test,top=num_images)
    elif how=='best':
        #Find indexes of best LR images per image set
        indexes=[np.argsort(np.sum(np.array(image_set[1:]),axis=(1,2)))[::-1][0:8]+1 for image_set in mask_LR_test]
        indexes=[np.append(0,indexes_imageset) for indexes_imageset in indexes]
        
        #images
        batch_test=np.array([image_set[indexes_set] for image_set,indexes_set in zip(input_images_LR_test,indexes)])
        batch_test=np.expand_dims(batch_test,axis=-1)
        #masks
        batch_test_mask=np.array([image_set[indexes_set] for image_set,indexes_set in zip(mask_LR_test,indexes)])
        batch_test_mask=np.expand_dims(batch_test_mask,axis=-1)
        #shifts
        shifts_test=np.array([shifts_set[indexes_set] for shifts_set,indexes_set in zip(shifts_test,indexes)])
        #These shifts are the shifts to be done to align the images and not the relative shifts, so let's reverse
        shifts_test=-shifts_test
    
        
    dataset=defaultdict()
    dataset['test']=batch_test
    dataset['test_mask']=batch_test_mask
    dataset['shifts_test']=shifts_test
    
    return dataset

