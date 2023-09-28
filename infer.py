import torch
import torch.nn.functional as F
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from utils import convert_state_dict
from data.preprocess.crop_merge_image import stride_integral
os.sys.path.append('./models/UNeXt')
from models.UNeXt.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L


def test_model1_model2(model1,model2,path_list,in_folder,sav_folder):

    for im_path in tqdm(path_list):
        in_name = im_path.split('_')[-1].split('.')[0]

        im_org = cv2.imread(im_path)
        im_org,padding_h,padding_w = stride_integral(im_org)
        h,w = im_org.shape[:2]
        im = cv2.resize(im_org,(512,512))
        im = im_org
        with torch.no_grad():
            im = torch.from_numpy(im.transpose(2,0,1)/255).unsqueeze(0)
            im = im.float().cuda()
            im_org = torch.from_numpy(im_org.transpose(2,0,1)/255).unsqueeze(0)
            im_org = im_org.float().cuda()

            shadow = model1(im)
            shadow = F.interpolate(shadow,(h,w))
            
            model1_im = torch.clamp(im_org/shadow,0,1)
            pred,_,_,_ = model2(torch.cat((im_org,model1_im),1))
           
            shadow = shadow[0].permute(1,2,0).data.cpu().numpy()
            shadow = (shadow*255).astype(np.uint8)
            shadow = shadow[padding_h:,padding_w:]

            model1_im = model1_im[0].permute(1,2,0).data.cpu().numpy()
            model1_im = (model1_im*255).astype(np.uint8)
            model1_im = model1_im[padding_h:,padding_w:]

            pred = pred[0].permute(1,2,0).data.cpu().numpy()
            pred = (pred*255).astype(np.uint8)
            pred = pred[padding_h:,padding_w:]


        cv2.imwrite(im_path.replace(in_folder,sav_folder),pred) 


if __name__ == '__main__':

    model1_path = 'checkpoints/gcnet/checkpoint.pkl'
    model2_path = 'checkpoints/drnet/checkpoint.pkl'
    model1 = UNext_full_resolution_padding(num_classes=3, input_channels=3,img_size=512).cuda()
    state = convert_state_dict(torch.load(model1_path,map_location='cuda:0')['model_state'])    
    model1.load_state_dict(state)
    model2 = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6,img_size=512).cuda()
    state = convert_state_dict(torch.load(model2_path,map_location='cuda:0')['model_state'])    
    model2.load_state_dict(state)

    model1.eval()
    model2.eval()

    img_folder = './distorted/'
    sav_folder = './enhanced/'
    if not os.path.exists(sav_folder):
        os.mkdir(sav_folder)

    im_paths = glob.glob(os.path.join(img_folder,'*'))
    test_model1_model2(model1,model2,im_paths,img_folder,sav_folder)