'''
Misc Utility functions
'''
from collections import OrderedDict
import os
import numpy as np
import torch
import random
import torchvision
import torch.nn.functional as F
import os 

def dict2string(loss_dict):
    loss_string = ''
    for key, value in loss_dict.items():
        loss_string += key+' {:.4f}, '.format(value)
    return loss_string[:-2]
def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)    

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])


def torch2cvimg(tensor,min=0,max=1):
    '''
    input:
        tensor -> torch.tensor BxCxHxW C can be 1,3
    return
        im -> ndarray uint8 HxWxC 
    '''
    im_list = []
    for i in range(tensor.shape[0]):
        im = tensor.detach().cpu().data.numpy()[i]
        im = im.transpose(1,2,0)
        im = np.clip(im,min,max)
        im = ((im-min)/(max-min)*255).astype(np.uint8)
        im_list.append(im)
    return im_list
def cvimg2torch(img,min=0,max=1):
    '''
    input:
        im -> ndarray uint8 HxWxC 
    return
        tensor -> torch.tensor BxCxHxW 
    '''
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img


def setup_seed(seed):
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed) #cpu
    # torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    # torch.backends.cudnn.benchmark = False   #训练集变化不大时使训练加速


