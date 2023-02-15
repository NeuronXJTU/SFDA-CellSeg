import argparse
import torch
import random
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable#maybe not used
from tensorboardX import SummaryWriter

import logging
import os
import sys
import math

from unet.unet_model import UNet
from unet.Unet_model_tent import UNet_tent
from model.resnet_encoder import resnet101
from Loss import Adversarial_loss, DiceLoss, entropy_loss, DisCrossEntropyLoss, CRloss, em_loss
from dataset import BasicDataset, TargetDataset, Target_rotate_Dataset
from SSDA_train import SSDA_main
from evaluate import mask_to_image
from model.resnet import resnet34
from gate_crf_loss import ModelLossSemsegGatedCRF

from tqdm import tqdm
from collections import OrderedDict
from collections.abc import Iterable
from medpy.metric import binary
import shutil


DIR_IMG_TARGET = 'data/img_target/'
DIR_IMG_SOURCE = 'data/img_source/'
DIR_MASK_SOURCE = 'data/mask_source/'
DIR_CHECKPOINT = 'checkpoints/'
DIR_IMG_TARGET_LABELED = 'data/img_target_labeled/'
DIR_MASK_TARGET_LABELED = 'data/mask_target_labeled/'
#dir_mask_target = 'data/mask_target'#这个要与无标签的target做个区分进入不同的S2


#后面可不可以整成输入size可变的

#freeze

# def set_freeze_by_names(model, layer_names, freeze=True):
#     if not isinstance(layer_names, Iterable):
#         layer_names = [layer_names]
#     for name, child in model.named_children():
#         if name not in layer_names:
#             continue
#         for param in child.parameters():
#             param.requires_grad = not freeze
            
# def freeze_by_names(model, layer_names):
#     set_freeze_by_names(model, layer_names, True)

# def unfreeze_by_names(model, layer_names):
#     set_freeze_by_names(model, layer_names, False)

# def set_freeze_by_idxs(model, idxs, freeze=True):
#     if not isinstance(idxs, Iterable):
#         idxs = [idxs]
#     num_child = len(list(model.children()))
#     idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
#     for idx, child in enumerate(model.children()):
#         if idx not in idxs:
#             continue
#         for param in child.parameters():
#             param.requires_grad = not freeze
            
# def freeze_by_idxs(model, idxs):
#     set_freeze_by_idxs(model, idxs, True)

# def unfreeze_by_idxs(model, idxs):
#     set_freeze_by_idxs(model, idxs, False)


def get_args():
    parser = argparse.ArgumentParser(description='Train the CellSegUDA/CellSegSSDA on source images and target images')
    parser.add_argument('--model', type=str, default='Unet',
                        help="available options : Unet")
    parser.add_argument('--type', type=str, default='UDA',
                        help='UDA or SSDA')
    parser.add_argument('--mode', type=str, default='Source',
                        help='Source or Target')
    parser.add_argument('--mode_bn', type=str, default='inc')
    parser.add_argument('--epochs', type=int, default= 100,
                        help='Number of epoches')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Number of images sent to the network in one step')
    parser.add_argument('--input-size-source', type=str, default='400,400', 
                        help='Comma-separated string with height and width of source images')
    parser.add_argument('--images-dir-source', type=str, default=DIR_IMG_SOURCE,
                        help='Path to the dictionary containing the source images')
    parser.add_argument('--masks-dir-source',type=str,default=DIR_MASK_SOURCE,
                        help='Path to the dictionary containing the source masks')
    parser.add_argument('--input-size-target', type=str, default='512,512',
                        help='Comma-separated string with height and width of target images')
    parser.add_argument('--images-dir-target',type=str, default=DIR_IMG_TARGET,
                        help='Path to the dictionary containing the target images')
    parser.add_argument('--images-dir-target_labeled',type=str, default=DIR_IMG_TARGET_LABELED)
    parser.add_argument('--masks-dir-target_labeled',type=str, default=DIR_MASK_TARGET_LABELED)
    parser.add_argument('--images-dir-eval', default='./data/img_evaluate/')#eval
    parser.add_argument('--masks-dir-eval', metavar='INPUT_masks', default='./data/masks_evaluate/')#eval
    parser.add_argument('--img-source-vali', metavar='vali_img', default='./data/img_source_vali/')#vali
    parser.add_argument('--mask-source-vali', metavar='vali_mask', default='./data/mask_source_vali/')#vali
    parser.add_argument('--img-target-vali', metavar='vali_target_img', default='./data/img_target_vali/')#vali
    parser.add_argument('--mask-target-vali', metavar='vali_target_mask', default='./data/mask_target_vali/')#vali
    parser.add_argument('--save-mask-dir', metavar='save_mask_dir', default='./data/result_target_mask/')
    parser.add_argument('--save-temp-mask-dir', metavar='save_temp_mask_dir', default='./data/result_temp_target_mask/')
    parser.add_argument('--logs-dir', type=str, default='./logs/',
                        help='Path to the dictionary containing the logs')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--txt-dir', type=str, default= './txts/',
                        help='Path to the dictionary containing the txts')
    parser.add_argument('--learning-rate-S', type=float, default=0.0001,
                        help='learning rate of Segmentation network')
    parser.add_argument('--learning-rate-D', type=float, default=0.001,
                        help='learning rate of Discriminator')
    parser.add_argument('--learning-rate-R', type=float, default=0.001,
                        help='learning rate of Decoder')
    parser.add_argument("--S-num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--R-num-classes", type=int, default=3,
                        help="Number of classes Reconstruction need to predict (including background).")
    parser.add_argument('--power',type=float, default=0.9,
                        help='Decay parameter to compute the learning rate')
    parser.add_argument('--resume', type=str, default='./logs/UDA100.pth')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Regularisation parameter for L2-loss')
    parser.add_argument('--lambda-adv', type=float, default=0.001,
                        help='The weight of adv loss')
    parser.add_argument('--lambda-recons', type=float, default=0.01,
                        help='The weight of recons loss')
    #parser.add_argument('--start-epoch', type=int, default=0)
    # parser.add_argument("--gpu", type=int, default=0,
    #                     help="choose gpu device.")
    parser.add_argument('--seed', type=int, default=256)#随机种子数
    parser.add_argument('--device', type=str, default='cuda')#运行设备
    parser.add_argument('--set', type=str, default='train',
                        help='choose adaptation set')
    return parser.parse_args()

args = get_args()

def load_model(model, model_path):
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def main():
    #create the model and start training
    # h, w = map(int, args.input_size_source.split(','))
    # input_size_source = (h, w)
    # h, w = map(int, args.input_size_target.split(','))
    # input_size_target = (h, w)

    TensorboardWriter = SummaryWriter(comment='BN_Freeze')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.enabled = True
    #create network
    model_S = UNet_tent(n_channels=3, n_classes=args.S_num_classes-1)#unet's n_classes should be 1 when actually 2 classes
        
    model_S = torch.nn.DataParallel(model_S).to(args.device)
    # if args.mode == 'Target':
    #     model_S = UNet2(n_channels=3, n_classes=args.S_num_classes-1)#unet's n_classes should be 1 when actually 2 classes
        
    #     model_S = torch.nn.DataParallel(model_S).to(args.device)

    start_epoch = 0
    lr_S = args.learning_rate_S

    if os.path.isfile(args.resume):
        print('loading checkpoint:{}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        lr_S = checkpoint['lr_S']
        model_S.load_state_dict(checkpoint['state_dict_S'])
        print('load successfully!')

    model_S.train()

    cudnn.benchmark = True

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)


    train_S_dataset = BasicDataset(args.images_dir_source, args.masks_dir_source, 1)#image_scale = 1 
    train_S_loader = data.DataLoader(train_S_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    
    #train_S_loader_iter = enumerate(train_S_loader)

    test_vali_source_loader = data.DataLoader(BasicDataset(args.img_source_vali, args.mask_source_vali), batch_size=1, shuffle=False, pin_memory=False)
    test_vali_target_loader = data.DataLoader(BasicDataset(args.img_target_vali, args.mask_target_vali), batch_size=1, shuffle=False, pin_memory=False)
    target_dataset = Target_rotate_Dataset(args.images_dir_target, 1)
    target_loader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)

    #targetloader_iter = enumerate(target_loader)
    # for idx in model_S.named_modules():
    #     print(idx)

    if args.mode == 'Target':
        for idx in model_S.named_modules():
            if 'bn' not in idx[0]:
                for param in idx[1].parameters():
                    param.requires_grad = False
            if 'bn' in idx[0]:
                for param in idx[1].parameters():
                    param.requires_grad = True
            # if 'outc' in idx[0]:
            #     for param in idx[1].parameters():
            #         param.requires_grad = True
            if 'inc' in idx[0]:
                for param in idx[1].parameters():
                    param.requires_grad = True
        optimizer_S = optim.Adam(filter(lambda p: p.requires_grad, model_S.parameters()), lr=0.00001)
        # optimizer_S.add_param_group({'params': model_S.fc1.parameters()})
    if args.mode == 'Source':
        optimizer_S = optim.Adam(model_S.parameters(), lr=lr_S)#adam default
        optimizer_S.zero_grad()


    seg_loss = DiceLoss()
    Target_loss = nn.CrossEntropyLoss()
    entropymin_loss = entropy_loss()
    mse_loss = nn.MSELoss()
    gatecrf_loss = ModelLossSemsegGatedCRF()

    loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_gatedcrf_radius = 5
    weight_crf = 0.1
    best_dice = 0.1

    for epoch in range(start_epoch, args.epochs):

        model_S.train()

        optimizer_S.zero_grad()
        
        #train source
        if(args.mode == 'Source'):
            for i,(images,mask, border) in enumerate(train_S_loader):
                images_S = images
                True_masks_S = mask
                True_border_S = border
                True_masks_S = True_masks_S.to(args.device)
                images_S = images_S.to(args.device)

                pred_masks_S = model_S(images_S)
                pred_masks_S_faltten = pred_masks_S.flatten()#this flatten might be not right
                True_masks_S_faltten = True_masks_S.flatten()
                #Loss_seg
                loss_seg = seg_loss(pred_masks_S_faltten, True_masks_S_faltten)
                optimizer_S.zero_grad()

                loss_seg.backward()

                optimizer_S.step()

        #compute the pred masks of target
        if(args.mode == 'Target'):
            for i,(imagest, imagesTr, idt) in enumerate(target_loader):
                images_T = imagest
                images_T = images_T.to(args.device)
                images_T_rotate = imagesTr
                images_T_rotate = images_T_rotate.to(args.device)

                pred_masks_T = model_S(images_T)
                outputs_soft = torch.softmax(pred_masks_T, dim=1) 

                # outputs_soft = torch.sigmoid(pred_masks_T) 

                _, s_tgt, _ = torch.linalg.svd(outputs_soft)

                alpha = 0.5
                if weight_crf < 1:
                    weight_crf = 0.05 * epoch
                loss_Entorpy = entropymin_loss(pred_masks_T)
                BNM_method_loss = -torch.mean(s_tgt)

                # BNM_method_loss = -torch.norm(outputs_soft,'nuc')

                pred_masks_T_rotate = model_S(images_T_rotate)
                pred_masks_T_2_rotate = torch.rot90(pred_masks_T, -1, [2,3])
                loss_rotate_T = mse_loss(pred_masks_T_2_rotate, pred_masks_T_rotate)
                loss_rotate_T = math.exp(-1 * pow(- epoch + start_epoch, 2)) *loss_rotate_T
                loss_Entorpy = entropymin_loss(pred_masks_T)
                out_gatedcrf = gatecrf_loss(
                    outputs_soft,
                    loss_gatedcrf_kernels_desc,
                    loss_gatedcrf_radius,
                    images_T,
                    512,
                    512,
                    )["loss"]
                optimizer_S.zero_grad()

                loss_target = loss_Entorpy + weight_crf * out_gatedcrf + alpha * BNM_method_loss + loss_rotate_T

                loss_target.backward()

                optimizer_S.step()
        if(args.mode=='Target'):

            print('path = {}'.format(args.logs_dir))
            print(
                'epoch = {0:6d}, loss_Entorpy = {1:.4f}'.format(
                    epoch, loss_Entorpy
                )
            )
            TensorboardWriter.add_scalar('loss_target', loss_target, global_step=epoch)
            TensorboardWriter.add_scalar('loss_crf', out_gatedcrf, global_step=epoch)
            TensorboardWriter.add_scalar('loss_bnm', BNM_method_loss, global_step=epoch)
            TensorboardWriter.add_scalar('loss_rotate', loss_rotate_T, global_step=epoch)

        if(args.mode=='Source'):    

            print('path = {}'.format(args.logs_dir))
            print(
                'epoch = {0:6d}, loss_seg = {1:.4f}'.format(
                    epoch, loss_seg
                )
            )
            TensorboardWriter.add_scalar('loss_seg', loss_seg, global_step=epoch)
            
        if epoch >= args.epochs - 1:
            print('save model ...')
            torch.save({
                'epoch' : epoch,
                'state_dict_S': model_S.state_dict(),
                'lr_S': optimizer_S.param_groups[0]['lr']
            }, os.path.join(args.logs_dir, 'UDA' + str(args.epochs) + '.pth'))

        # if epoch % args.save_every == 0 and epoch != 0:
        #     print('save model when epoch = {}'.format(epoch))
        #     torch.save({
        #         'epoch' : epoch,
        #         'state_dict_S': model_S.state_dict(),
        #         'lr_S': optimizer_S.param_groups[0]['lr']
        #     }, os.path.join(args.logs_dir, 'UDA' + str(epoch) + '.pth'))

        #eval
        print('begin eval!')
        model_S.eval()
        
        temp = 0
        dice_temp = 0
        num = 0
        assd_temp = 0
        hd95_temp = 0
        if(args.mode=='Target'):
            with torch.no_grad():
                for i,(images,mask,name) in enumerate(test_vali_target_loader):
                    mask = mask
                    image = images
                    #img = image.unsqueeze(0)
                    img = image.to(device=args.device, dtype=torch.float32)
                    mask = mask.to(device=args.device, dtype=torch.float32)
                    
                    output = model_S(img)
                    probs = torch.sigmoid(output)

                    probs = probs.squeeze(0)

                    tf = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.ToTensor()
                        ]
                    )

                    probs = tf(probs.cpu())
                    full_mask = probs.squeeze().cpu().numpy()

                    out = full_mask > 0.5
                    result = mask_to_image(out)
                    out_file = args.save_temp_mask_dir + '/' + str(name)
                    result.save(out_file +'.png')

                    tf2 = transforms.Compose([transforms.ToTensor()])
                    result_t = tf2(result)
                    result_t = result_t.to(device=args.device, dtype=torch.float32)
                    a = torch.flatten(mask)
                    b = torch.flatten(result_t)

                    a[a > 0.5] = 1
                    a[a <= 0.5] = 0
                    b[b > 0.5] = 1
                    b[b <= 0.5] = 0
                    nul = a * b

                    dice = 2 * torch.sum(nul) / (torch.sum(a) + torch.sum(b))
                    dice_temp = dice_temp + dice

                    intersection = torch.sum(nul)
                    union = torch.sum(a) + torch.sum(b) - intersection + 1e-6
                    miou = intersection / union

                    mask = mask.cpu().numpy()
                    result_t = result_t.cpu().numpy()
                    mask = np.reshape(mask, result_t.shape)
                    
                    assd = binary.assd(result_t, mask, connectivity=1)
                    hd95 = binary.hd95(result_t, mask)
                    hd95_temp = hd95_temp + hd95
                    assd_temp = assd_temp + assd
                    temp = temp + miou
                    num = num + 1

            t_assd = assd_temp / num
            t_miou = temp / num
            t_dice = dice_temp / num
            t_hd95 = hd95_temp / num
            if t_dice > best_dice:
                print('save best model ...')
                torch.save({
                    'epoch' : epoch,
                    'state_dict_S': model_S.state_dict(),
                    'lr_S': optimizer_S.param_groups[0]['lr']
                }, os.path.join(args.logs_dir, 'UDA' + str(args.epochs) +'best.pth'))
                for root, _, fnames in os.walk(args.save_temp_mask_dir):
                    for fname in sorted(fnames):  # sorted函数把遍历的文件按文件名排序
                        fpath = os.path.join(root, fname)
                        shutil.copy(fpath, args.save_mask_dir)  # 完成文件拷贝
                best_dice = t_dice

            TensorboardWriter.add_scalar('target vali eval miou', t_miou, global_step=epoch)
            TensorboardWriter.add_scalar('target vali eval dice', t_dice, global_step=epoch)
            TensorboardWriter.add_scalar('target vali eval assd', t_assd, global_step=epoch)
            TensorboardWriter.add_scalar('target vali eval hd95', t_hd95, global_step=epoch)
            TensorboardWriter.add_scalar('target vali eval best dice', best_dice, global_step=epoch)
            print("target vali eval miou is {0}, target vali eval dice is {1}".format(t_miou,t_dice))
    
        if(args.mode=='Source'):
            with torch.no_grad():
                for i,(images,mask,name) in enumerate(test_vali_source_loader):
                    mask = mask
                    image = images
                    #img = image.unsqueeze(0)
                    img = image.to(device=args.device, dtype=torch.float32)
                    mask = mask.to(device=args.device, dtype=torch.float32)
                    
                    output = model_S(img)
                    probs = torch.sigmoid(output)

                    probs = probs.squeeze(0)

                    tf = transforms.Compose(
                        [
                            transforms.ToPILImage(),
                            transforms.ToTensor()
                        ]
                    )

                    probs = tf(probs.cpu())
                    full_mask = probs.squeeze().cpu().numpy()

                    out = full_mask > 0.5
                    result = mask_to_image(out)

                    tf2 = transforms.Compose([transforms.ToTensor()])
                    result = tf2(result)
                    TensorboardWriter.add_image(str(name)+'mask', result, global_step=epoch, walltime=None, dataformats='CHW')
                    result = result.to(device=args.device, dtype=torch.float32)
                    a = torch.flatten(mask)
                    b = torch.flatten(result)

                    a[a > 0.5] = 1
                    a[a <= 0.5] = 0
                    b[b > 0.5] = 1
                    b[b <= 0.5] = 0
                    nul = a * b

                    dice = 2 * torch.sum(nul) / (torch.sum(a) + torch.sum(b))
                    dice_temp = dice_temp + dice

                    intersection = torch.sum(nul)
                    union = torch.sum(a) + torch.sum(b) - intersection + 1e-6
                    miou = intersection / union
                    b = b.cpu().numpy()
                    a = a.cpu().numpy()
                    if epoch > 5:
                        assd = binary.assd(b, a, connectivity=1)
                        hd95 = binary.hd95(b, a)
                        hd95_temp = hd95_temp + hd95
                        assd_temp = assd_temp + assd
                    temp = temp + miou
                    num = num + 1

            t_assd = assd_temp / num
            t_miou = temp / num
            t_dice = dice_temp / num
            t_hd95 = hd95_temp / num
            TensorboardWriter.add_scalar('target vali eval miou', t_miou, global_step=epoch)
            TensorboardWriter.add_scalar('target vali eval dice', t_dice, global_step=epoch)
            TensorboardWriter.add_scalar('target vali eval assd', t_assd, global_step=epoch)
            TensorboardWriter.add_scalar('target vali eval hd95', t_hd95, global_step=epoch)
            print("source vali eval miou is {0}, source vali eval dice is {1}".format(t_miou,t_dice))

if __name__ == '__main__':
    if args.type == 'UDA':
        main()
    elif args.type == 'SSDA':
        SSDA_main(args)
