import os
import torch
from skimage import io
from skimage.color import rgb2lab
from skimage.util import img_as_float
import scipy
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
from config import RAND_SEED
from random import Random
myrandom = Random(RAND_SEED)

# Dataset
class BSDS500(Dataset):
    """BSDS500 Dataset"""

    def __init__(self, root='/workspace/data_and_code/BSDS500', split='train', transform = None, height = 321, width = 481, num_classes = 50):
        self.root = root
        self.split = split
        self.images_dir = os.path.join(self.root, 'data', 'images', split)
        self.targets_dir = os.path.join(self.root, 'data', 'groundTruth', split)
        self.transform = transform
        self.height = height
        self.width = width
        self.num_classes = num_classes
        
        # the close() method is automatically called via the "with" statement
        with open(os.path.join(root, split + '.txt')) as f:
            self.imgname_list = [line.rstrip() for line in f.readlines()]

    def __len__(self):
        return len(self.imgname_list)

    def __getitem__(self, index):
        
        imgname = self.imgname_list[index]

        img_fullpath = os.path.join(self.images_dir, imgname + '.jpg')
        img = img_as_float(io.imread(img_fullpath))                     # raw rgb image

        gt_fullpath = os.path.join(self.targets_dir, imgname + '.mat')
        target_all = loadmat(gt_fullpath)                               # includes all five annotations for the img specified by imgname
        
        t = np.random.randint(0, len(target_all['groundTruth'][0]))     # randomly chooses one of the five annotations as the 'target'
        target = target_all['groundTruth'][0][t][0][0][0]

        if img.shape[0] == self.height and img.shape[1] == self.width:
            pass
        else:
            img = np.transpose(img, (1, 0, 2))
            target = np.transpose(target, (1, 0))

        sample = (img, target)

        if self.transform:
            sample = self.transform(sample)
    
        return sample

# Transforms
class scale_sample(object):

    def __call__(self, sample):
        img, target = sample
        
        rand_factor = np.random.normal(1, 0.75)

        s_factor = np.min((3.0, rand_factor))
        s_factor = np.max((0.75, s_factor))

        img = scipy.ndimage.zoom(img, (s_factor, s_factor, 1), order = 1)
        target = scipy.ndimage.zoom(target, (s_factor, s_factor), order = 0)
        
        return (img, target)

class img2lab(object):

    def __call__(self, sample):
        img, target = sample
        img = rgb2lab(img)

        return (img, target)

class left_right_flip_sample(object):

    def __call__(self, sample):
        img, target = sample

        if np.random.uniform(0, 1) > 0.5:
        # the general syntax for a slice is array[start : stop : step]
        # any or all of the values start, stop, step may be left out
        # in which case, they resume their default values;
        # start = 0     (start index included)
        # stop = len    (stop index not included)
        # step = 1
        # 
        # the flip is therefore performed by "::-1"
        #
            img = img[:, ::-1, ...] - np.zeros_like(img)        # img has size [H, W, C]; np.zeros_like(img) is subtracted cuz when we eventually change to torch.from_numpy doesn't support numpy arrays that have been reversed using negative stride
            target = target[:, ::-1]                            # target has size [H, W] 

        return (img, target)

class random_crop_sample(object):

    def __init__(self, patch_size):
        self.patch_height = patch_size[0]        # patch_size = [patch_height, patch_width]
        self.patch_width = patch_size[1]

    def __call__(self, sample):
        img, target = sample
        img_height, img_width = target.shape

        # randomize the start point for patching / cropping
        start_row = myrandom.randint(0, img_height - self.patch_height)
        start_col = myrandom.randint(0, img_width - self.patch_width)

        # crop image to size [out_height, out_width]
        img_cropped = img[start_row : start_row + self.patch_height,
                        start_col : start_col + self.patch_width, :]

        # crop the ground truth
        target_cropped = target[start_row : start_row + self.patch_height,
                            start_col : start_col + self.patch_width]

        return (img_cropped, target_cropped)

class convert_label(object):
    """Ignores labels bigger or equal to num_classes = 50."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        img, target = sample

        probtarget = np.zeros((self.num_classes, target.shape[0], target.shape[1])).astype(np.float32)

        ct = 0
        for t in np.unique(target).tolist():
            if ct >= self.num_classes:
                break
            else:
                probtarget[ct, :, :] = (target == t)
            ct += 1

        return (img, np.squeeze(np.argmax(probtarget, axis = 0)))


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img, target = sample
        assert isinstance(img, np.ndarray)
        assert isinstance(target, np.ndarray)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img = img.transpose((2, 0, 1))

        # add an extra axis to targets s.t. it has the same number of dimensions as img
        target = target[None, :, :]
        
        return (torch.from_numpy(img), torch.from_numpy(target.astype(np.int64)))

# Composed Transform
def transform_patch_data_train(patch_size, num_classes):
    return transforms.Compose([scale_sample(),
                               img2lab(),
                               left_right_flip_sample(),
                               random_crop_sample(patch_size),
                               convert_label(num_classes),
                               ToTensor()])

# compose transform for validation
def transform_convert_label(num_classes):
    return transforms.Compose([convert_label(num_classes),
                               ToTensor()])

def totensor():
    return transforms.Compose([ToTensor()])
