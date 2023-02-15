import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable#maybe not used

import logging
import os
import sys

from unet.unet_model import Unet_out, Unet_outR
from Discriminator import GANDiscriminator
from Loss import Adversarial_loss, DiceLoss, Reconstruction_loss, DisCrossEntropyLoss
from dataset import BasicDataset, TargetDataset

from tqdm import tqdm


def SSDA_main(getargs):
    #create the model and start training
    # h, w = map(int, args.input_size_source.split(','))
    # input_size_source = (h, w)
    # h, w = map(int, args.input_size_target.split(','))
    # input_size_target = (h, w)
    args = getargs

    cudnn.enabled = True

    #create network
    model_S = Unet_out(n_classes=args.S_num_classes-1)#unet's n_classes should be 1 when actually 2 classes

    #could have a restore part

    model_S.train()
    model_S.cuda(args.gpu)



    #Init D and R
    model_D = GANDiscriminator(num_classes=args.S_num_classes-1)
    model_R = Unet_outR(n_classes=args.R_num_classes)#这里是不是3呢？好吧是4

    model_D.train()
    model_D.cuda(args.gpu)

    model_R.train()
    model_R.cuda(args.gpu)

    cudnn.benchmark = True

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)


    train_S_dataset = BasicDataset(args.images_dir_source, args.masks_dir_source, 1)#image_scale = 1 
    train_S_loader = data.DataLoader(train_S_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    #train_S_loader_iter = enumerate(train_S_loader)
    target_dataset = TargetDataset(args.images_dir_target, 1)
    target_loader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    target_L_dataset = BasicDataset(args.images_dir_target_labeled, args.masks_dir_target_labeled, 1)
    target_L_loader = data.DataLoader(target_L_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    #targetloader_iter = enumerate(target_loader)

    logging.info(
        '''
        Start Training:
        '''
    )#log need

    optimizer_S = optim.Adam(model_S.parameters(), lr=args.learning_rate_S)#adam default
    optimizer_S.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D)
    optimizer_D.zero_grad()

    optimizer_R = optim.Adam(model_R.parameters(), lr=args.learning_rate_R)
    optimizer_R.zero_grad()

    adv_loss = Adversarial_loss()
    seg_loss = DiceLoss()
    dis_loss = DisCrossEntropyLoss()
    recons_loss = Reconstruction_loss()

    # Labels for Discriminater loss
    source_label = 1
    target_label = 0

    for epoch in range(args.epochs):

        optimizer_D.zero_grad()
        optimizer_R.zero_grad()
        optimizer_S.zero_grad()

        #train S
        #remove grads
        for param in model_D.parameters():
            param.requires_grad = False
        for param in model_R.parameters():
            param.requires_grad = False
        
        #train source
        for batch in train_S_loader:
            images_S = batch['image']
            True_masks_S = batch['mask']
            True_masks_S = Variable(True_masks_S).cuda(args.gpu)
            images_S = Variable(images_S).cuda(args.gpu)

            pred_masks_S = model_S(images_S)
            pred_masks_S_faltten = pred_masks_S.flatten()#this flatten might be not right
            True_masks_S_faltten = True_masks_S.flatten()

            #Loss_seg
            loss_seg = seg_loss(pred_masks_S_faltten, True_masks_S_faltten)

            optimizer_S.zero_grad()
            loss_seg.backward()
            optimizer_S.step()


        #train target SSDA needed
            for batch in target_L_loader:
                images_T_L = batch['image']
                images_T_L = Variable(images_T_L).cuda(args.gpu)
                True_masks_T_L = batch['mask']
                True_masks_T_L = Variable(True_masks_T_L).cuda(args.gpu)
                pred_masks_T_L = model_S(images_T_L)
                pred_masks_T_L_faltten = pred_masks_T_L.flatten()#this flatten might be not right
                True_masks_T_L_faltten = True_masks_T_L.flatten()

                #Loss_seg_L
                loss_seg_L = seg_loss(pred_masks_T_L_faltten, True_masks_T_L_faltten)
                loss_seg_L.backward()

                pred_masks_T_L = model_S(images_T_L)

                D_out_L= model_D(pred_masks_T_L)
                loss_adv = adv_loss(D_out_L, args.gpu)

                loss_adv = loss_adv * args.lambda_adv
                loss_adv.backward()

                #train R
                for param in model_R.parameters():
                    param.requires_grad = True
                
                pred_masks_T_L = pred_masks_T_L.detach()
                pred_images_T_L = model_R(pred_masks_T_L)

                loss_recons = recons_loss(pred_images_T_L, images_T_L)
                loss_recons = loss_recons * args.lambda_recons

                optimizer_R.zero_grad()
                loss_recons.backward()
                # loss = loss_adv + loss_recons + loss_seg
                # loss.backward()

                #train D
                for param in model_D.parameters():
                    param.requires_grad = True

                pred_masks_S = pred_masks_S.detach()
                D_out_S = model_D(pred_masks_S)
                loss_dis_s = dis_loss(D_out_S, source_label, args.gpu)

                optimizer_D.zero_grad()
                loss_dis_s.backward()

                pred_masks_T_L = pred_masks_T_L.detach()#detach切断与之前的联系
                D_out_T_L = model_D(pred_masks_T_L)
                loss_dis_t = dis_loss(D_out_T_L, target_label, args.gpu)# Z caution
                loss_dis_t.backward()

                optimizer_D.step()
                optimizer_R.step()

        #compute the pred masks of target

            for batch in target_loader:
                images_T = batch['image']#target don't have masks
                images_T = Variable(images_T).cuda(args.gpu)

                pred_masks_T = model_S(images_T)

                D_out= model_D(pred_masks_T)
                loss_adv = adv_loss(D_out, args.gpu)

                loss_adv = loss_adv * args.lambda_adv
                loss_adv.backward()

                #train R
                for param in model_R.parameters():
                    param.requires_grad = True
                
                pred_masks_T = pred_masks_T.detach()
                pred_images_T = model_R(pred_masks_T)

                loss_recons = recons_loss(pred_images_T, images_T)
                loss_recons = loss_recons * args.lambda_recons

                optimizer_R.zero_grad()
                loss_recons.backward()
                # loss = loss_adv + loss_recons + loss_seg
                # loss.backward()

                #train D
                for param in model_D.parameters():
                    param.requires_grad = True

                pred_masks_T = pred_masks_T.detach()#detach切断与之前的联系
                D_out_T = model_D(pred_masks_T)
                loss_dis_t = dis_loss(D_out_T, target_label, args.gpu)# Z caution
                loss_dis_t.backward()

                optimizer_D.step()
                optimizer_R.step()

        print('path = {}'.format(args.logs_dir))
        print(
            'epoch = {0:6d}, loss_seg = {1:.4f}, loss_adv = {2:.4f}, loss_recons = {3:.4f}, loss_dis_T = {4:.4f}, loss_dis_T = {5:.4f}'.format(
                epoch, loss_seg, loss_adv, loss_recons, loss_dis_s, loss_dis_t
            )
        )

        f_loss = open(os.path.join(args.logs_dir, 'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n'.format(
            loss_seg, loss_adv, loss_recons, loss_dis_s, loss_dis_t
        ))
        f_loss.close()
        if epoch >= args.epochs - 1:
            print('save model ...')
            torch.save(model_S.state_dict(), os.path.join(args.logs_dir, 'SSDA' + str(args.epochs) + '.pth'))
            torch.save(model_D.state_dict(), os.path.join(args.logs_dir, 'SSDA' + str(args.epochs) + '_D.pth'))
            torch.save(model_R.state_dict(), os.path.join(args.logs_dir, 'SSDA' + str(args.epochs) + '_R.pth'))
            break

        if epoch % args.save_every == 0 and epoch != 0:
            print('save model when epoch = {}'.format(epoch))
            torch.save(model_S.state_dict(), os.path.join(args.logs_dir, 'SSDA' + str(epoch) + '.pth'))
            torch.save(model_D.state_dict(), os.path.join(args.logs_dir, 'SSDA' + str(epoch) + '_D.pth'))
            torch.save(model_R.state_dict(), os.path.join(args.logs_dir, 'SSDA' + str(epoch) + '_R.pth'))