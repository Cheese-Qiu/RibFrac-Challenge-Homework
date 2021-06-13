from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from scipy import ndimage
import config
from utils import logger, weights_init, metrics,common
from dataset.dataset_lits_test import Test_Datasets,to_one_hot_3d
import SimpleITK as sitk
import os
import numpy as np
from models import UNet
from utils.metrics import DiceAverage
import torch.nn as nn

def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    target = to_one_hot_3d(img_dataset.label, args.n_labels)
    
    with torch.no_grad():
        for data in tqdm(dataloader,total=len(dataloader)):
            data = data.to(device)
            output = model(data)
            output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred,dim=1)
    
    pred = np.asarray(pred.numpy(),dtype='uint8')
    pred = sitk.GetImageFromArray(np.squeeze(pred,axis=0))

    return pred

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./Homework_experiments', "up3")
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model = UNet(in_channel=1, out_channel=args.n_labels).to(device)
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])

    test_log = logger.Test_Logger(save_path,"test_log")
    # data info
    test_data_path = './test_data'
    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    datasets = Test_Datasets(test_data_path,args=args)
    for img_dataset,file_idx in datasets:
        pred_img = predict_one_img(model, img_dataset, args)
        seg_array1 = sitk.GetArrayFromImage(pred_img)
        print(seg_array1.shape)
        if (seg_array1==0).all():
            print("all 0")
        sitk.WriteImage(pred_img, os.path.join(result_save_path, file_idx.replace('image','label')))
