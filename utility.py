import os
import numpy as np
import random
import _pickle as pickle
import tensorflow as tf
from tensorflow import keras

# def train_generate(directory, batch_size=1, num_samples=8):
#     """Replaces Keras' native ImageDataGenerator."""
#     i = 0
#     file_list = os.listdir(directory)
#     random.shuffle(file_list)
#     while True:
#         if i == len(file_list):
#             i = 0
#             random.shuffle(file_list)
                
#         sample = file_list[i]
#         i += 1
# #         img_path = [os.path.join(directory, sample, "pre", "TOF.nii.gz")]
# #         mask_path = [os.path.join(directory, sample, "aneurysms.nii.gz")]
        
# #         image = sitk.ReadImage(img_path)
# #         image = sitk.GetArrayFromImage(image)
# #         image = whitening(image)
# #         image=image[0]
# #         image = np.expand_dims(image, axis=-1)
        
# #         mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.int32)
# #         mask = (mask==True).astype(np.int32)
# #         mask = mask[0]
            
# # #             image, mask = _augment(image, mask)
            
# #         image, mask = extract_class_balanced_example_array(
# #                 image=image,
# #                 label=mask,
# #                 example_size=(32,128,128),
# #                 n_examples=num_samples,
# #                 classes=2,
# #             )
# #         image = image.transpose(0,4,1,2,3)
#         with open(os.path.join(directory,sample,'image.pkl'),'rb') as fp:
#             image = pickle.load(fp)
        
#         with open(os.path.join(directory,sample,'mask.pkl'),'rb') as fp:
#             mask = pickle.load(fp)
                  
#         j= -(batch_size)
#         mask = mask.astype(np.float32)
#         ind_list = list(range(image.shape[0]))
#         random.shuffle(ind_list)
#         image = image[ind_list,...]
#         mask = mask[ind_list,...]
#         while j+2*batch_size<=image.shape[0]:
#             j+=batch_size
#             yield image[j:j+batch_size],mask[j:j+batch_size]
            
# # def _augment(img, lbl):
# #     """An image augmentation function"""
# #     img = add_gaussian_noise(img, sigma=0.1)
# # #     [img, lbl] = flip([img, lbl], axis=1)

# #     return img, lbl   

# def val_generate(directory, batch_size=1, num_samples=8):
#     """Replaces Keras' native ImageDataGenerator."""
#     i = 0
#     file_list = os.listdir(directory)
#     random.shuffle(file_list)
#     while True:
#         if i == len(file_list):
#             i = 0
#             random.shuffle(file_list)
                
#         sample = file_list[i]
#         i += 1

#         with open(os.path.join(directory,sample,'image.pkl'),'rb') as fp:
#             image = pickle.load(fp)
        
#         with open(os.path.join(directory,sample,'mask.pkl'),'rb') as fp:
#             mask = pickle.load(fp)
                  
#         j= -(batch_size)
#         mask = mask.astype(np.float32)
#         while j+2*batch_size<=image.shape[0]:
#             j+=batch_size
#             yield image[j:j+batch_size],mask[j:j+batch_size]

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, "image", id_name)
        mask_path = os.path.join(self.path, "mask", id_name)
        
        ## Reading Image
        with open(image_path,'rb') as fp:
            image = pickle.load(fp)
        
        with open(mask_path,'rb') as fp:
            mask = pickle.load(fp)
        mask = np.expand_dims(mask,axis=0)
        mask = mask.astype(np.float32)
        
        return image, mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        np.random.shuffle(self.ids)
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
            
def load_test_sample(sample_folder_path):
    
    with open(os.path.join(sample_folder_path,'image.pkl'),'rb') as fp:
        image = pickle.load(fp) 
    with open(os.path.join(sample_folder_path,'mask.pkl'),'rb') as fp:
        mask = pickle.load(fp)
        
    mask = mask.astype(np.float32)
    
    return image,mask
    
    
def sliding_window(img, mask, patch_size):
    img = img[0]
    img_patches = list()
    mask_patches = list()
    sample_locs = list()
    i=0
    while i<img.shape[0]:
        
        j=0
        while j<img.shape[1]:
            
            k=0
            while k<img.shape[2]:
                
                sample_loc = list()
                
                if i+patch_size[0]<=img.shape[0]:
                    img_patch = img[i:i+patch_size[0],:,:]
                    mask_patch = mask[i:i+patch_size[0],:,:]
                    sample_loc.append(i)
                else:
                    img_patch = img[-patch_size[0]:,:,:]
                    mask_patch = mask[-patch_size[0]:,:,:]
                    sample_loc.append(img.shape[0]-patch_size[0])
                    
                    
                if j+patch_size[1]<=img.shape[1]:
                    img_patch = img_patch[:,j:j+patch_size[1],:]
                    mask_patch = mask_patch[:,j:j+patch_size[1],:]
                    sample_loc.append(j)
                else:
                    img_patch = img_patch[:,-patch_size[1]:,:]
                    mask_patch = mask_patch[:,-patch_size[1]:,:]
                    sample_loc.append(img.shape[1]-patch_size[1])
                    
                if k+patch_size[2]<=img.shape[2]:
                    img_patch = img_patch[:,:,k:k+patch_size[2]]
                    mask_patch = mask_patch[:,:,k:k+patch_size[2]]
                    sample_loc.append(k)
                else:
                    img_patch = img_patch[:,:,-patch_size[2]:]
                    mask_patch = mask_patch[:,:,-patch_size[2]:]
                    sample_loc.append(img.shape[2]-patch_size[2])
                
                img_patch = np.expand_dims(img_patch,0)
                
                img_patches.append(img_patch)
                mask_patches.append(mask_patch)
                sample_locs.append(sample_loc)
                
                k+=patch_size[2]
                
            j+=patch_size[1]
            
        i+=patch_size[0]
        
    return (np.array(img_patches),np.array(mask_patches),np.array(sample_locs))
    
def reconstruct(mask_patches, sample_locs, org_dim, patch_size):
    
    final_mask = np.zeros(org_dim, dtype=np.float32)
    
    for i in range(mask_patches.shape[0]):
        patch = mask_patches[i]
        loc = sample_locs[i]
        print(patch.shape)
        
        temp = final_mask[loc[0]:loc[0]+patch_size[0], loc[1]:loc[1]+patch_size[1], loc[2]:loc[2]+patch_size[2]].copy()
        print(temp.shape)
        final_mask[loc[0]:loc[0]+patch_size[0], loc[1]:loc[1]+patch_size[1], loc[2]:loc[2]+patch_size[2]] = np.maximum(temp,patch)
        
    return final_mask

def intensity_clip(image):
    step_size = (image.max()-image.min())/200
    up = image.max()-step_size
    lo = image.min()+step_size
    image[image<=lo] = 0.
    image[image>up] = up
    return image

def zscore(image):
    mean = np.mean(image)
    std = np.std(image)
    image = (image-mean)/std
    return image

def normalize(image):
    image = (image-image.min())/(image.max()-image.min())
    return image
    
    