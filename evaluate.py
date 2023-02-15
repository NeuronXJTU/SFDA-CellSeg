import argparse
import logging
import os

from torch.utils import data
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from dataset import BasicDataset, BasicTestDataset
import cv2

from unet.unet_model import UNet, Unet_out
from unet.attUnet import AttU_Net
from data_vis import plot_img_and_mask
from dataset import BasicDataset
from model.resnet import resnet34



def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='./restore/model.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--type', type=str, default=' ')
    parser.add_argument('--model-type', type=str, default=' ')
    parser.add_argument('--input', metavar='INPUT', default='./data/img_evaluate/',
                        help='filenames of input images')
    parser.add_argument('--input-masks', metavar='INPUT_masks', default='./data/masks_evaluate/',
                        help='filenames of input masks')
    parser.add_argument('--output',  metavar='OUTPUT', default='./data/result/eval',
                        help='Filenames of ouput images')
    parser.add_argument('--mask-threshold', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

from collections import OrderedDict

if __name__ == "__main__":
    args = get_args()
    testloader = data.DataLoader(BasicTestDataset(args.input, args.input_masks), batch_size=1, shuffle=False, pin_memory=True)
    if args.model_type == 'AttUnet':
        model = AttU_Net(img_ch=3, output_ch=1)
    elif args.model_type == 'ResNet':
        model = resnet34()
    else:
        model = UNet(3,1)
    model.eval()

    logging.info("Loading model {}".format(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    model.to(device=device)
    checkpoint = torch.load(args.model)
    if args.type == 'MUL':
        # original saved file with DataParallel
        #model.load_state_dict(torch.load(args.model))

        state_dict = checkpoint['state_dict_S']
        # create new OrderedDict that does not contain `module.`

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    else:
        #model.load_state_dict(checkpoint['state_dict_S'])
        model.load_state_dict(torch.load(args.model, map_location=device))

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    logging.info("Model loaded !")

    temp = 0
    dice_temp = 0
    num = 0
    with torch.no_grad():
        for i,(images,mask,name) in enumerate(testloader):
            mask = mask
            image = images
            #img = image.unsqueeze(0)
            img = image.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)

            output = model(img)
            probs = torch.sigmoid(output)

            probs = probs.squeeze(0)

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor()
                ]
            )

            # if args.model_type=='ResNet':
            #     tf = transforms.Compose(
            #         [
            #             transforms.ToPILImage(),
            #             transforms.Resize((400,400)),
            #             transforms.ToTensor()
            #         ]
            #     )
                
            probs = tf(probs.cpu())
            full_mask = probs.squeeze().cpu().numpy()

            out = full_mask > 0.5
            result = mask_to_image(out)
            out_file = args.output + '/' + str(name)
            result.save(out_file +'.png')
            logging.info("Mask saved to {}".format(out_file + '.png'))

            tf2 = transforms.Compose([transforms.ToTensor()])
            result = tf2(result)
            result = result.to(device=device, dtype=torch.float32)
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

            # atemp = torch.zeros(a.size())
            # btemp = torch.zeros(b.size())
            # aoht = atemp.scatter_(index=a, dim=1, value=1)
            # boht = btemp.scatter_(index=b, dim=1, value=1)
            # mul = aoht * boht
            # ious = []
            # for j in range(a.shape[1]):
            #     intersection = torch.sum(mul[j])
            #     union = torch.sum(aoht[j]) + torch.sum(boht[j]) - intersection + 1e-6
            #     iou = intersection / union
            #     ious.append(iou)
            #     miou = np.mean(ious)#计算该图像的miou

            temp = temp + miou
            num = num + 1
    
    t_miou = temp / num
    t_dice = dice_temp / num
    print("eval miou is {}".format(t_miou))
    print("eval dice is {}".format(t_dice))



