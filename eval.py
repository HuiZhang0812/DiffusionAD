import matplotlib.pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve,average_precision_score
from sklearn.metrics import roc_auc_score
import time
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import torch.nn as nn
from models.Recon_subnetwork import UNetModel, update_ema_params
from models.Seg_subnetwork import SegmentationSubNetwork
import torch.nn as nn
from data.dataset_beta_thresh import MVTecTrainDataset,MVTecTestDataset,VisATrainDataset,VisATestDataset,DAGMTrainDataset,DAGMTestDataset,MPDDTestDataset,MPDDTrainDataset
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from math import exp
import torch.nn.functional as F
torch.cuda.empty_cache()
from tqdm import tqdm
import json
import os
from collections import defaultdict
import pandas as pd
import torchvision.utils
import os
from torch.utils.data import DataLoader
from skimage.measure import label, regionprops
import sys

def pixel_pro(mask,pred):
    mask=np.asarray(mask, dtype=np.bool_)
    print(mask.shape)
    pred = np.asarray(pred)
    print(pred.shape)

    max_step = 1000
    expect_fpr = 0.3  # default 30%
    max_th = pred.max()
    min_th = pred.min()
    delta = (max_th - min_th) / max_step
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(pred, dtype=np.bool_)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[pred <= thred] = 0
        binary_score_maps[pred >  thred] = 1
        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
        for i in range(len(binary_score_maps)):    # for i th image
            # pro (per region level)
            label_map = label(mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image    # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], mask[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], mask[i]).astype(np.float32).sum()
            if mask[i].any() > 0:    # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        gt_masks_neg = ~mask
        fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / gt_masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)
    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)
    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)
    # best per image iou
    best_miou = ious_mean.max()
    #print(f"Best IOU: {best_miou:.4f}")
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = min_max_norm(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]    
    seg_pro_auc = auc(fprs_selected, pros_mean_selected)
    return seg_pro_auc

def gridify_output(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
            0, 2,
            1
            ).contiguous().permute(
            2, 1, 0
            )


def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd


def load_checkpoint(param, device,sub_class,checkpoint_type,args):

    ck_path = f'{args["output_path"]}/model/diff-params-ARGS={param}/{sub_class}/params-{checkpoint_type}.pt'
    print("checkpoint",ck_path)
    loaded_model = torch.load(ck_path, map_location=device)
          
    return loaded_model


def load_parameters(device,sub_class,checkpoint_type):
    
    param = "args1.json"
    with open(f'./args/{param}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = param[4:-5]
    args = defaultdict_from_json(args)

    output = load_checkpoint(param[4:-5], device,sub_class,checkpoint_type,args)
 
    return args, output


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def image_transform(image):
     return np.clip(image* 255, 0, 255).astype(np.uint8)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True):  
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap
def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam) 


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def heatmap(real: torch.Tensor, recon: torch.Tensor, mask, filename, save=True):
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = mse > 0
    mse_threshold = (mse_threshold.float() * 2) - 1
    if save:
        output = torch.cat((real, recon.reshape(1, *recon.shape), mse, mse_threshold, mask))
        plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
        plt.axis('off')
        plt.savefig(filename)
        plt.clf()


def testing(testing_dataset_loader, args,unet_model,seg_model,data_len,sub_class,class_type,checkpoint_type,device):
    
    
    normal_t=args["eval_normal_t"]
    noiser_t=args["eval_noisier_t"]
    
    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/visualization_{normal_t}_{noiser_t}_{args["condition_w"]}condition_{checkpoint_type}ck', exist_ok=True)
    
    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    
    
    
    print("data_len",data_len)
    total_image_pred = np.array([])
    total_image_gt =np.array([])
    total_pixel_gt=np.array([])
    total_pixel_pred = np.array([])
    gt_matrix_pixel=[]
    pred_matrix_pixel=[]
    tbar = tqdm(testing_dataset_loader)
    for i, sample in enumerate(tbar):
        image = sample["image"].to(device)
        target=sample['has_anomaly'].to(device)
        gt_mask = sample["mask"].to(device)
        image_path = sample["file_name"]
        
        normal_t_tensor = torch.tensor([normal_t], device=image.device).repeat(image.shape[0])
        noiser_t_tensor = torch.tensor([noiser_t], device=image.device).repeat(image.shape[0])
        loss,pred_x_0_condition,pred_x_0_normal,pred_x_0_noisier,x_normal_t,x_noiser_t,pred_x_t_noisier = ddpm_sample.norm_guided_one_step_denoising_eval(unet_model, image, normal_t_tensor,noiser_t_tensor,args)
        
        pred_mask = seg_model(torch.cat((image, pred_x_0_condition), dim=1)) 


        out_mask = pred_mask

        #pixel_aupro

        gt_matrix_pixel.extend(gt_mask[0].detach().cpu().numpy().astype(int))
        pred_matrix_pixel.extend(out_mask[0].detach().cpu().numpy())

        topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
        topk_out_mask = torch.topk(topk_out_mask, 50, dim=1, largest=True)[0]
        image_score = torch.mean(topk_out_mask)
        
        total_image_pred=np.append(total_image_pred,image_score.detach().cpu().numpy())
        total_image_gt=np.append(total_image_gt,target[0].detach().cpu().numpy())


        flatten_pred_mask=out_mask[0].flatten().detach().cpu().numpy()
        flatten_gt_mask =gt_mask[0].flatten().detach().cpu().numpy().astype(int)
            
        
        total_pixel_gt=np.append(total_pixel_gt,flatten_gt_mask)
        total_pixel_pred=np.append(total_pixel_pred,flatten_pred_mask)

        
        
        
        x_noiser_t = image_transform(x_noiser_t.detach().cpu().numpy()[0])
        x_normal_t = image_transform(x_normal_t.detach().cpu().numpy()[0])
        pred_x_t_noisier = image_transform(pred_x_t_noisier.detach().cpu().numpy()[0])
       
        raw_image = image_transform(image.detach().cpu().numpy()[0])

        ano_map = gaussian_filter(
                out_mask[0, 0, :, :].detach().cpu().numpy(), sigma=4)
        ano_map=min_max_norm(ano_map)
        ano_map=cvt2heatmap(ano_map*255.0)

        image_cv2 = np.uint8(np.transpose(raw_image,(1,2,0)))
        ano_map = show_cam_on_image(image_cv2[...,::-1], ano_map)
        ano_map=ano_map[...,::-1]
        recon_condition = image_transform(pred_x_0_condition.detach().cpu().numpy()[0])
        recon_normal_t = image_transform(pred_x_0_normal.detach().cpu().numpy()[0])
        recon_noisier_t = image_transform(pred_x_0_noisier.detach().cpu().numpy()[0])

        savename = image_path[0].split("/")
        savename = "_".join(savename[-4:])
        savename = os.path.join(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/visualization_{normal_t}_{noiser_t}_{args["condition_w"]}condition_{checkpoint_type}ck', savename)

        f, axes = plt.subplots(2, 5 )
        f.suptitle(f'image score:{str(image_score.detach().cpu().numpy())}')
        axes[0][0].imshow(raw_image.transpose(1, 2, 0).astype(np.uint8))
        axes[0][0].set_title('Input')
        axes[0][0].axis('off')

        axes[0][1].imshow((gt_mask[0][0]*255.0).detach().cpu().numpy().astype(np.uint8),cmap ='gray')
        axes[0][1].set_title('GT')
        axes[0][1].axis('off')

        axes[0][2].imshow(x_normal_t.transpose(1, 2, 0).astype(np.uint8))
        axes[0][2].set_title('x_normal_t')
        axes[0][2].axis('off')

        axes[0][3].imshow(x_noiser_t.transpose(1, 2, 0).astype(np.uint8))
        axes[0][3].set_title('x_noiser_t')
        axes[0][3].axis('off')

        axes[0][4].imshow(pred_x_t_noisier.transpose(1, 2, 0).astype(np.uint8))
        axes[0][4].set_title('pred_x_t_noisier')
        axes[0][4].axis('off')

        axes[1][0].imshow(ano_map)
        axes[1][0].set_title('heatmap')
        axes[1][0].axis('off')
        
        axes[1][1].imshow((out_mask[0][0]*255.0).detach().cpu().numpy().astype(np.uint8),cmap ='gray')
        axes[1][1].set_title('out_mask')
        axes[1][1].axis('off')

        axes[1][2].imshow(recon_normal_t.transpose(1, 2, 0).astype(np.uint8))
        axes[1][2].set_title('recon_normal')
        axes[1][2].axis('off')

        axes[1][3].imshow(recon_noisier_t.transpose(1, 2, 0).astype(np.uint8))
        axes[1][3].set_title('recon_noisier')
        axes[1][3].axis('off')

        axes[1][4].imshow(recon_condition.transpose(1, 2, 0).astype(np.uint8))
        axes[1][4].set_title('recon_con')
        axes[1][4].axis('off')

        f.set_size_inches(3 * (5 ), 3*(2))
        f.tight_layout()
        f.savefig(savename)
        plt.close()


    
    auroc_image = round(roc_auc_score(total_image_gt,total_image_pred),3)*100
    print("Image AUC-ROC: " ,auroc_image)
    
    
    auroc_pixel =  round(roc_auc_score(total_pixel_gt, total_pixel_pred),3)*100
    print("Pixel AUC-ROC:" ,auroc_pixel)

    ap_pixel =  round(average_precision_score(total_pixel_gt, total_pixel_pred),3)*100
    print("Pixel-AP:",ap_pixel) 
    


    aupro_pixel = round(pixel_pro(gt_matrix_pixel,pred_matrix_pixel),3)*100
    print("Pixel-AUPRO:" ,aupro_pixel)
    
    temp = {"classname":[sub_class],"Image-AUROC": [auroc_image],"Pixel-AUROC":[auroc_pixel],"Pixel-AUPRO":[aupro_pixel],"Pixel_AP":[ap_pixel]}
    df_class = pd.DataFrame(temp)
    df_class.to_csv(f"{args['output_path']}/metrics/ARGS={args['arg_num']}/{normal_t}_{noiser_t}t_{class_type}_{args['condition_w']}condition_{checkpoint_type}ck.csv", mode='a',header=False,index=False)
   


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mvtec_classes = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw',
     'toothbrush', 'transistor', 'zipper']
    mpdd_classes = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'] 
    visa_classes = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
             'pcb3', 'pcb4', 'pipe_fryum']

    dagm_class = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']
    
    current_classes=mvtec_classes
    checkpoint_type='best'

    for sub_class in current_classes:
        args, output = load_parameters(device,sub_class,checkpoint_type)
        print(f"args{args['arg_num']}")
        print("class",sub_class)
        
        in_channels = args["channels"]

        unet_model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                    "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
                in_channels=in_channels
                ).to(device)



        seg_model=SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)

        unet_model.load_state_dict(output["unet_model_state_dict"])
        unet_model.to(device)
        unet_model.eval()

        seg_model.load_state_dict(output["seg_model_state_dict"])
        seg_model.to(device)
        seg_model.eval()

        print("EPOCH:",output['n_epoch'])

        if sub_class in visa_classes:
            subclass_path = os.path.join(args['visa_root_path'],sub_class)
            print("visa_eval_root_path",args["visa_root_path"])
            print("subclass_path",subclass_path)
            testing_dataset = VisATestDataset(
                subclass_path,sub_class,img_size=args["img_size"],
                )
            class_type='VisA'
        elif sub_class in mpdd_classes:
            subclass_path = os.path.join(args["mpdd_root_path"],sub_class)
            testing_dataset = MVTecTestDataset(
                subclass_path,sub_class,img_size=args["img_size"],
                )
            class_type='MPDD'
        elif sub_class in mvtec_classes:
            subclass_path = os.path.join(args["mvtec_root_path"],sub_class)
            testing_dataset = MVTecTestDataset(
                subclass_path,sub_class,img_size=args["img_size"],
                )
            class_type='MVTec'

        elif sub_class in dagm_class:
            subclass_path = os.path.join(args["dagm_root_path"],sub_class)
            testing_dataset =DAGMTestDataset(
                subclass_path,sub_class,img_size=args["img_size"],
                )
            class_type='DAGM'
        
                
        data_len = len(testing_dataset) 
        test_loader = DataLoader(testing_dataset, batch_size=1,shuffle=False, num_workers=4)

        
        # make arg specific directories
        
        for i in [f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}']:
            try:
                os.makedirs(i)
            except OSError:
                pass


        testing(test_loader, args,unet_model,seg_model,data_len,sub_class,class_type,checkpoint_type,device)


if __name__ == '__main__':
    
    main()
