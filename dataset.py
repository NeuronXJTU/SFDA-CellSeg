from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os
import cv2
# import albumentations as albu
# import albumentations.augmentations.functional as F

def pad_image(image, new_size):
    '''
    mean padding and keep original picture in center
    :param image: numpy.array, shape = [H,W,C]
    :param new_size: list (like [H, W, C]), shape = [3]
    :return: numpy.array, shape = new_size
    '''
    h_need_pad = int(new_size[1] - image.shape[1])
    h_top = int(h_need_pad/2)
    h_bottom = h_need_pad - h_top
    w_need_pad = int(new_size[2] - image.shape[2])
    w_left = int(w_need_pad/2)
    w_right = w_need_pad - w_left
    #pd_image = np.zeros(shape=new_size, dtype=np.uint8)
    pd_image = np.pad(image, ((0,0), (h_top, h_bottom), (w_left, w_right)), mode='constant', constant_values=0)
    # for i in range(image.shape[0]):
    #     #ch_mean = np.mean(image[i, :, :], axis=(1, 2))
    #     pd_image[i, :, :] = np.pad(image[i, :, :],
    #                                 ((h_top, h_bottom), (w_left, w_right)),
    #                                 mode='constant',
    #                                 constant_values=0)

    return pd_image

def crop_image(image, new_size, cropmode='center', left_top_point=(0,0)):

    h = image.shape[1]
    w = image.shape[2]
    if cropmode == 'center':
        h_top = int((image.shape[1] - new_size[1]) / 2)
        h_bottom = h_top + new_size[1]
        w_left = int((image.shape[2] - new_size[2]) / 2)
        w_right = w_left + new_size[2]
    if cropmode == 'handcraft':
        assert (left_top_point[0] + new_size[1]) <= image.shape[1]
        h_top = left_top_point[0]
        h_bottom = left_top_point[0] + new_size[1]
        assert (left_top_point[1] + new_size[2]) <= image.shape[2]
        w_left = left_top_point[1]
        w_right = left_top_point[1] + new_size[2]
    
    image = image[:, h_top:h_bottom, w_left:w_right] 
    image = pad_image(image, [3, h, w])
    return image

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((512,512))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        # assert img.size == mask.size, \
        #    f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)        
        
        # img_cut_anchor = crop_image(img, [3, 200, 200], cropmode='handcraft', left_top_point=(30,30))
        # img_cut_A = crop_image(img, [3, 200, 200])
        # img_cut_B = crop_image(img_cut_A, [3, 180, 180])
        img_rotate = np.rot90(img, -1, [1,2]).copy()#顺时针旋转90度,不能直接转
        #mask_rotate = np.rot90(img, -1)

        return [torch.from_numpy(img).type(torch.FloatTensor),torch.from_numpy(mask).type(torch.FloatTensor),idx]
        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor),
        #     'mask': torch.from_numpy(mask).type(torch.FloatTensor),
        #     'image_rotate': torch.from_numpy(img_rotate).type(torch.FloatTensor),#rotate needed
        #     #'mask_rotate': torch.from_numpy(mask_rotate).type(torch.FloatTensor),
        #     # 'image_cut_anchor': torch.from_numpy(img_cut_anchor).type(torch.FloatTensor),
        #     # 'image_cut_P': torch.from_numpy(img_cut_A).type(torch.FloatTensor),
        #     # 'image_cut_N': torch.from_numpy(img_cut_B).type(torch.FloatTensor),
        #     'name': idx
        # }

class BasicBorderDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, border_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.border_dir = border_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((512,512))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        border_file = glob(self.border_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(border_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {border_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        border = Image.open(border_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        assert img.size == border.size, \
            f'Image and border {idx} should be the same size, but are {img.size} and {border.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        border = self.preprocess(border, self.scale)
        
        # img_cut_anchor = crop_image(img, [3, 200, 200], cropmode='handcraft', left_top_point=(30,30))
        # img_cut_A = crop_image(img, [3, 200, 200])
        # img_cut_B = crop_image(img_cut_A, [3, 180, 180])
        img_rotate = np.rot90(img, -1, [1,2]).copy()#顺时针旋转90度,不能直接转
        #mask_rotate = np.rot90(img, -1)

        return [torch.from_numpy(img).type(torch.FloatTensor),torch.from_numpy(mask).type(torch.FloatTensor), torch.from_numpy(border).type(torch.FloatTensor)]

class BasicTestDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((512,512))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)        
        
        #mask_rotate = np.rot90(img, -1)

        return [torch.from_numpy(img).type(torch.FloatTensor),torch.from_numpy(mask).type(torch.FloatTensor),idx]
        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor),
        #     'mask': torch.from_numpy(mask).type(torch.FloatTensor),
        #     'name': idx
        # }

class TargetDataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((512,512))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])

        img = self.preprocess(img, self.scale)
        img_rotate = np.rot90(img, -1, [1,2]).copy()

        # img_cut_anchor = crop_image(img, [3, 200, 200], cropmode='handcraft', left_top_point=(30,30))
        # img_cut_A = crop_image(img, [3, 200, 200])
        # img_cut_B = crop_image(img_cut_A, [3, 180, 180])

        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor),
        #     'image_rotate': torch.from_numpy(img_rotate).type(torch.FloatTensor)
        #     # 'image_cut_anchor': torch.from_numpy(img_cut_anchor).type(torch.FloatTensor),
        #     # 'image_cut_P': torch.from_numpy(img_cut_A).type(torch.FloatTensor),
        #     # 'image_cut_N': torch.from_numpy(img_cut_B).type(torch.FloatTensor)
        # }
        return [torch.from_numpy(img).type(torch.FloatTensor), idx]

class TargetTestDataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((512,512))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])

        img = self.preprocess(img, self.scale)

        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor)
        # }
        return [torch.from_numpy(img).type(torch.FloatTensor)]

class Target_rotate_Dataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((512,512))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])

        img = self.preprocess(img, self.scale)
        img_rotate = np.rot90(img, -1, [1,2]).copy() #逆时针旋转

        # img_cut_anchor = crop_image(img, [3, 200, 200], cropmode='handcraft', left_top_point=(30,30))
        # img_cut_A = crop_image(img, [3, 200, 200])
        # img_cut_B = crop_image(img_cut_A, [3, 180, 180])

        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor),
        #     'image_rotate': torch.from_numpy(img_rotate).type(torch.FloatTensor)
        #     # 'image_cut_anchor': torch.from_numpy(img_cut_anchor).type(torch.FloatTensor),
        #     # 'image_cut_P': torch.from_numpy(img_cut_A).type(torch.FloatTensor),
        #     # 'image_cut_N': torch.from_numpy(img_cut_B).type(torch.FloatTensor)
        # }
        return [torch.from_numpy(img).type(torch.FloatTensor),torch.from_numpy(img_rotate).type(torch.FloatTensor), idx]

class Target_Filp_Dataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((512,512))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])

        img = self.preprocess(img, self.scale)
        img_filp = np.fliplr(img).copy() #左右翻转

        # img_cut_anchor = crop_image(img, [3, 200, 200], cropmode='handcraft', left_top_point=(30,30))
        # img_cut_A = crop_image(img, [3, 200, 200])
        # img_cut_B = crop_image(img_cut_A, [3, 180, 180])

        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor),
        #     'image_rotate': torch.from_numpy(img_rotate).type(torch.FloatTensor)
        #     # 'image_cut_anchor': torch.from_numpy(img_cut_anchor).type(torch.FloatTensor),
        #     # 'image_cut_P': torch.from_numpy(img_cut_A).type(torch.FloatTensor),
        #     # 'image_cut_N': torch.from_numpy(img_cut_B).type(torch.FloatTensor)
        # }
        return [torch.from_numpy(img).type(torch.FloatTensor),torch.from_numpy(img_filp).type(torch.FloatTensor), idx]

class Target_Ranking_Dataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((512,512))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])

        img = self.preprocess(img, self.scale)
        img_filp = np.fliplr(img).copy() #左右翻转

        img_cut_anchor = crop_image(img, [3, 300, 300], cropmode='handcraft', left_top_point=(30,30)).copy() 
        img_cut_A = crop_image(img, [3, 300, 300]).copy() 
        img_cut_B = crop_image(img_cut_A, [3, 200, 200]).copy() 

        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor),
        #     'image_rotate': torch.from_numpy(img_rotate).type(torch.FloatTensor)
        #     # 'image_cut_anchor': torch.from_numpy(img_cut_anchor).type(torch.FloatTensor),
        #     # 'image_cut_P': torch.from_numpy(img_cut_A).type(torch.FloatTensor),
        #     # 'image_cut_N': torch.from_numpy(img_cut_B).type(torch.FloatTensor)
        # }
        return [torch.from_numpy(img).type(torch.FloatTensor),torch.from_numpy(img_cut_anchor).type(torch.FloatTensor),torch.from_numpy(img_cut_A).type(torch.FloatTensor), torch.from_numpy(img_cut_B).type(torch.FloatTensor), idx]

class Target_resnet_Dataset(Dataset):
    def __init__(self, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((224,224))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])

        img = self.preprocess(img, self.scale)
        img_rotate = np.rot90(img, -1, [1,2]).copy() #逆时针旋转

        # img_cut_anchor = crop_image(img, [3, 200, 200], cropmode='handcraft', left_top_point=(30,30))
        # img_cut_A = crop_image(img, [3, 200, 200])
        # img_cut_B = crop_image(img_cut_A, [3, 180, 180])

        # return {
        #     'image': torch.from_numpy(img).type(torch.FloatTensor),
        #     'image_rotate': torch.from_numpy(img_rotate).type(torch.FloatTensor)
        #     # 'image_cut_anchor': torch.from_numpy(img_cut_anchor).type(torch.FloatTensor),
        #     # 'image_cut_P': torch.from_numpy(img_cut_A).type(torch.FloatTensor),
        #     # 'image_cut_N': torch.from_numpy(img_cut_B).type(torch.FloatTensor)
        # }
        return [torch.from_numpy(img).type(torch.FloatTensor), idx]

class Basic_resnet_Dataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((224,224))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)        
        
        # img_cut_anchor = crop_image(img, [3, 200, 200], cropmode='handcraft', left_top_point=(30,30))
        # img_cut_A = crop_image(img, [3, 200, 200])
        # img_cut_B = crop_image(img_cut_A, [3, 180, 180])
        img_rotate = np.rot90(img, -1, [1,2]).copy()#顺时针旋转90度,不能直接转
        #mask_rotate = np.rot90(img, -1)

        return [torch.from_numpy(img).type(torch.FloatTensor),torch.from_numpy(mask).type(torch.FloatTensor),idx]

class resnet_TS_Data(Dataset):
    def __init__(self, source_imgs_dir, source_masks_dir, target_imgs_dir, scale=1):
        self.source_imgs_dir = source_imgs_dir
        self.source_masks_dir = source_masks_dir
        self.target_imgs_dir = target_imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(source_imgs_dir)
                    if not file.startswith('.')]
        self.idt = [splitext(file)[0] for file in listdir(target_imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} source examples')
        logging.info(f'Creating dataset with {len(self.idt)} target examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((224,224))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i % len(self.ids)]
        idt = self.idt[i % len(self.idt)]
        source_mask_file = glob(self.source_masks_dir + idx + '.*')
        source_img_file = glob(self.source_imgs_dir + idx + '.*')
        target_img_file = glob(self.target_imgs_dir + idt + '.*')

        assert len(source_mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {source_mask_file}'
        assert len(source_img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {source_img_file}'
        source_mask = Image.open(source_mask_file[0])
        source_img = Image.open(source_img_file[0])
        target_img = Image.open(target_img_file[0])

        assert source_img.size == source_mask.size, \
            f'Image and mask {idx} should be the same size, but are {source_img.size} and {source_mask.size}'

        source_img = self.preprocess(source_img, self.scale)
        source_mask = self.preprocess(source_mask, self.scale)
        target_img = self.preprocess(target_img, self.scale)        
        
        # img_cut_anchor = crop_image(img, [3, 200, 200], cropmode='handcraft', left_top_point=(30,30))
        # img_cut_A = crop_image(img, [3, 200, 200])
        # img_cut_B = crop_image(img_cut_A, [3, 180, 180])
        source_img_rotate = np.rot90(source_img, -1, [1,2]).copy()#顺时针旋转90度,不能直接转
        #mask_rotate = np.rot90(img, -1)

        return [torch.from_numpy(source_img).type(torch.FloatTensor),torch.from_numpy(source_mask).type(torch.FloatTensor),torch.from_numpy(target_img).type(torch.FloatTensor), idx, idt]

class Basic_TS_Data(Dataset):
    def __init__(self, source_imgs_dir, source_masks_dir, target_imgs_dir, scale=1):
        self.source_imgs_dir = source_imgs_dir
        self.source_masks_dir = source_masks_dir
        self.target_imgs_dir = target_imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(source_imgs_dir)
                    if not file.startswith('.')]
        self.idt = [splitext(file)[0] for file in listdir(target_imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} source examples')
        logging.info(f'Creating dataset with {len(self.idt)} target examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((512,512))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i % len(self.ids)]
        idt = self.idt[i % len(self.idt)]
        source_mask_file = glob(self.source_masks_dir + idx + '.*')
        source_img_file = glob(self.source_imgs_dir + idx + '.*')
        target_img_file = glob(self.target_imgs_dir + idt + '.*')

        assert len(source_mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {source_mask_file}'
        assert len(source_img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {source_img_file}'
        source_mask = Image.open(source_mask_file[0])
        source_img = Image.open(source_img_file[0])
        target_img = Image.open(target_img_file[0])

        assert source_img.size == source_mask.size, \
            f'Image and mask {idx} should be the same size, but are {source_img.size} and {source_mask.size}'

        source_img = self.preprocess(source_img, self.scale)
        source_mask = self.preprocess(source_mask, self.scale)
        target_img = self.preprocess(target_img, self.scale)        
        
        # img_cut_anchor = crop_image(img, [3, 200, 200], cropmode='handcraft', left_top_point=(30,30))
        # img_cut_A = crop_image(img, [3, 200, 200])
        # img_cut_B = crop_image(img_cut_A, [3, 180, 180])
        source_img_rotate = np.rot90(source_img, -1, [1,2]).copy()#顺时针旋转90度,不能直接转
        #mask_rotate = np.rot90(img, -1)

        return [torch.from_numpy(source_img).type(torch.FloatTensor),torch.from_numpy(source_mask).type(torch.FloatTensor),torch.from_numpy(target_img).type(torch.FloatTensor), idx, idt]