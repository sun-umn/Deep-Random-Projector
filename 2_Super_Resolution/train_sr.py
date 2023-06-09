#----------------This is the main script for super-resolution -----------------#

from torch.autograd import Variable
from util import *
from tqdm import tqdm
import time
import torch.optim.lr_scheduler as lrs
import math
import pandas as pd
from psnr_ssim import *
import torch
import torch.optim
import numpy as np
import time
from BN_Net import *
from models import *
from utils.denoising_utils import *
from models.downsampler import Downsampler
from utils.sr_utils import *
from skimage.measure import compare_psnr
import skimage.io as ski




def DIP_train(imgs,learning_rate_model,OPTIMIZER,LOSS,width, input_w, input_h, input_c,max_epoch,print_step,gpu,cur_image,tv_weight,factor):
    dir_name = '{}/UntrainedNN_training/'.format(cur_image)
    dir_name_best = '{}/0_BEST_Results/'.format(cur_image)
    make_dir([dir_name, dir_name_best])

    #------------------------ Using GPU when it is available ------------------------#
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    print('#_______________________')
    print('#_______________________')
    print(torch.cuda.is_available())
    print('device info:')
    print(device)
    print('#_______________________')
    print('#_______________________')

    #------------------------ Noise & Net ------------------------#
    pad = 'reflection'
    KERNEL_TYPE = 'lanczos2'
    input_depth = input_c
    img_HR_var = np_to_torch(imgs['HR_np']).to(device)
    output_depth = img_HR_var.size()[1]


    net = get_net(input_depth, 'skip', pad,n_channels=output_depth,
                  skip_n33d=width,
                  skip_n33u=width,
                  skip_n11=4,
                  num_scales=1,
                  upsample_mode='bilinear')
    #### Init our net
    #net.apply(weights_init)
    print(net)
    for param_key in net.state_dict():
        # custom initialization in new_weight_dict,
        # You can initialize partially i.e only some of the variables and let others stay as it is
        print(param_key)

    net.to(device)
    print(net)

    CNN_weight_name = [
        '1.0.1.1.weight', '1.0.1.1.bias',
        '1.1.1.1.weight', '1.1.1.1.bias',
        '1.1.4.1.weight', '1.1.4.1.bias',
        '3.1.weight', '3.1.bias',
        '6.1.weight', '6.1.bias',
        '9.1.weight', '9.1.bias']

    # first, let's get how many trainable parameters we have in this network
    net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_input_params = input_c * input_w * input_h
    output_size = img_HR_var.size()[1]*img_HR_var.size()[2]*img_HR_var.size()[3]
    over_ratio_net = round(total_input_params / net_total_params, 3)
    over_ratio_out = round(total_input_params / output_size, 3)
    param_dict = {'Type': [],
                  'Num': []}
    param_dict['Type'].append('Input_Dim')
    param_dict['Num'].append(total_input_params)
    param_dict['Type'].append('Network_Param')
    param_dict['Num'].append(net_total_params)
    param_dict['Type'].append('Output_Size')
    param_dict['Num'].append(output_size)
    param_dict['Type'].append('In2Net')
    param_dict['Num'].append(over_ratio_net)
    param_dict['Type'].append('In2Out')
    param_dict['Num'].append(over_ratio_out)
    param_df = pd.DataFrame.from_dict(param_dict)
    param_count_file = os.path.join(dir_name, 'Param_Count.csv')
    param_df.to_csv(param_count_file, index=False)

    # # let's fix the CNN weights of networks but keep the BN trainable
    for name, param in net.named_parameters():
        ### here we do not want to fix the BN
        if name in CNN_weight_name:
            param.requires_grad = False
    net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('After freezing, the left trainable param is {}'.format(net_total_params))

    ## BN net
    bnnet = BNNet(input_c)
    bnnet.to(device)

    #----------------------------- net input------------------------------#
    # Let's get the network input
    
    ###------------- Choice 1: using Normal Distribution-------------------###
    noise_like = torch.empty(1, input_c, input_h, input_w).to(device)
    g_noise = torch.zeros_like(noise_like).normal_() * 1e-1
    g_noise.requires_grad = True
    p_c = [g_noise]

    ###------------- Choice 2: using Uniform Distribution-------------------###
    # noise_like = torch.empty(1, input_c, input_w, input_h).to(device)
    # g_noise = torch.zeros_like(noise_like)
    # g_noise.data.uniform_()
    # g_noise.data *=1./10
    # g_noise.requires_grad = True
    # p_c = [g_noise]


    #------------------------------- Set up downsampler------------------------------------#
    img_LR_var = np_to_torch(imgs['LR_np']).to(device)
    downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).to(device)


    #-------------------------- Define Loss Function & Opt -------------------------#
    l1_loss_func = L1_Func()
    l2_loss_func = L2_Func()


    #--------------------------  Define Optimizer --------------------------#
    if OPTIMIZER == 'SGD':
        optimizer_net = torch.optim.SGD(net.parameters(), learning_rate_model, momentum=0.9)
        optimizer_noise = torch.optim.SGD(p_c, lr=learning_rate_model)
        optimizer_bn = torch.optim.SGD(bnnet.parameters(), lr=learning_rate_model)

    elif OPTIMIZER == 'Adam':
        optimizer_net = torch.optim.Adam(net.parameters(), learning_rate_model)
        optimizer_noise = torch.optim.Adam(p_c, lr=learning_rate_model)
        optimizer_bn = torch.optim.Adam(bnnet.parameters(), lr=learning_rate_model)

    else:
        assert False, "Optimizer is not supported!"

    #--------------------------  Start to Train --------------------------#
    all_mse_loss = {'Epoch':[], 'MSE':[]}
    wall_time_dict = {'Epoch':[], 'Per_Time':[], 'All_Time':[]}
    all_used_time = 0


    total_loss = []
    total_loss_epoch = []
    total_epoch = []

    psnr_LR = []
    psnr_HR = []

    ssim_LR = []
    ssim_HR = []

    for epoch in range(max_epoch):
        if epoch != 0:
            # ------ start to measure the time in this iteration
            t_start = time.time()

        net.train()
        bnnet.train()
        epoch_loss = []

         # ------ start to measure the time in this iteration

        optimizer_net.zero_grad()
        optimizer_noise.zero_grad()
        optimizer_bn.zero_grad()

        ################## Train and Backpropagation ##################
        g_noise_input = bnnet(g_noise)
        out_HR = net(g_noise_input)
        out_LR = downsampler(out_HR)

        if LOSS == "L1":
            # print('**************')
            # print('Using L1 loss')
            # print('**************')
            loss = l1_loss_func(out_LR, img_LR_var)  # L1
            loss = torch.sqrt(loss)
            loss =  loss + tv_weight * tv1_loss(out_HR)

        elif LOSS == "MSE":
            # print('**************')
            # print('Using MSE')
            # print('**************')
            loss = l2_loss_func(out_LR, img_LR_var)  # MSE
            loss = torch.sqrt(loss) 
            loss = loss + tv_weight * tv1_loss(out_HR)

        else:
            assert False, "Loss function is wrong!"

        loss.backward()
        optimizer_net.step()
        optimizer_noise.step()
        optimizer_bn.step()
        epoch_loss.append(loss.item())

        # after one epoch
        # ------ after one epoch: end to measure the time in this iteration
        if epoch != 0:
            t_end = time.time()
            per_used_time = t_end - t_start
            all_used_time += per_used_time

            wall_time_dict['Epoch'].append(epoch)
            wall_time_dict['Per_Time'].append(per_used_time)
            wall_time_dict['All_Time'].append(all_used_time)

            # after we have the time, let's save it to csv
            if epoch % 100 == 0:  # save the results for every 100 epochs
                wall_time_df = pd.DataFrame.from_dict(wall_time_dict)
                wall_time_file = dir_name + '00_wall_time.csv'
                wall_time_df.to_csv(wall_time_file, index=False)

        total_loss.append(np.mean(epoch_loss))
        total_loss_epoch.append(epoch)

        all_mse_loss['Epoch'].append(epoch)
        all_mse_loss['MSE'].append(np.mean(epoch_loss))


        #----------- after one epoch training, we can start to test our model ---------------#
        if epoch%print_step == 0:
            figure_name = dir_name + '00_rec.png'

            cur_psnr_LR, cur_psnr_HR, cur_ssim_LR, cur_ssim_HR = DIP_test(imgs, out_LR, out_HR,figure_name)
            psnr_LR.append(cur_psnr_LR)
            psnr_HR.append(cur_psnr_HR)
            ssim_LR.append(cur_ssim_LR)
            ssim_HR.append(cur_ssim_HR)
            total_epoch.append(epoch)

            ##### plot loss
            if epoch % 100 == 0:  # save the results for every 100 epochs
                loss_file_name = dir_name + 'loss.png'
                display_loss(total_loss, print_step, loss_file=loss_file_name)

            # let's save our loss
            if epoch % 100 == 0:  # save the results for every 100 epochs
                loss_csv_file_name = dir_name + 'loss.csv'
                all_mse_df = pd.DataFrame.from_dict(all_mse_loss)
                all_mse_df.to_csv(loss_csv_file_name, index=False)


            #### plost psnr
            if epoch % 100 == 0:  # save the results for every 100 epochs
                loss_file_name = dir_name + 'psnr.png'
                display_psnr(psnr_LR, psnr_HR, print_step, loss_file=loss_file_name)

            #### after training, we can save the psnr values
            psnr_dict = {'epoch': total_epoch,
                         'psnr_LR': psnr_LR,
                         'psnr_HR': psnr_HR,
                         'ssim_LR': ssim_LR,
                         'ssim_HR': ssim_HR,
                         }

            if epoch % 100 == 0:  # save the results for every 100 epochs
                psnr_df = pd.DataFrame.from_dict(psnr_dict)
                psnr_file = dir_name + '00_psnr.csv'
                psnr_df.to_csv(psnr_file, index=False)


def DIP_test(imgs, out_LR, out_HR,figure_name):
    # Log
    psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
    ssim_LR = calcualte_SSIM(imgs['LR_np'], torch_to_np(out_LR), multichannel=True)

    out_HR_np = np.clip(torch_to_np(out_HR), 0, 1)
    q1 = out_HR_np[:3].sum(0)
    t1 = np.where(q1.sum(0) > 0)[0]
    t2 = np.where(q1.sum(1) > 0)[0]
    # center_out_HR_np = put_in_center(out_HR_np, imgs['HR_np'].shape[1:])
    psnr_HR = compare_psnr_y(imgs['HR_np'][:3, t2[0] + 4:t2[-1] - 4, t1[0] + 4:t1[-1] - 4],
                             out_HR_np[:3, t2[0] + 4:t2[-1] - 4, t1[0] + 4:t1[-1] - 4])
    ssim_HR = calcualte_SSIM(imgs['HR_np'][:3, t2[0] + 4:t2[-1] - 4, t1[0] + 4:t1[-1] - 4],
                             out_HR_np[:3, t2[0] + 4:t2[-1] - 4, t1[0] + 4:t1[-1] - 4], multichannel=True)


    out_HR_np = torch_to_np(out_HR)
    #plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)
    draw_HR_tensor = torch.from_numpy(imgs['HR_np'])
    draw_Bicubic_HR_tensor = torch.from_numpy(imgs['bicubic_np'])
    draw_out_HR_tensor = torch.from_numpy(np.clip(out_HR_np, 0, 1))
    all_images = [draw_HR_tensor,draw_Bicubic_HR_tensor,draw_out_HR_tensor]
    draw_figures(all_images, figure_name)
    return psnr_LR, psnr_HR, ssim_LR, ssim_HR


if __name__ == '__main__':
    pass
