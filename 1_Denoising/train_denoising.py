#----------------This is the main script for denoising-----------------#

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



def DIP_train(learning_rate_model,OPTIMIZER,LOSS,width, input_w, input_h, input_c,clean_image_path,corrupted_image_path,max_epoch,print_step,gpu,cur_image,tv_weight):
    dir_name = '{}/UntrainedNN_training/'.format(cur_image)
    dir_name_best = '{}/0_BEST_Results/'.format(cur_image)
    make_dir([dir_name, dir_name_best])

    ################## Using GPU when it is available ##################
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    print('#_______________________')
    print('#_______________________')
    print(torch.cuda.is_available())
    print('device info:')
    print(device)
    print('#_______________________')
    print('#_______________________')

    ############################ Settings from noise & network
    INPUT = 'noise'  # 'meshgrid'
    pad = 'reflection'
    reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50


    input_depth = input_c

    net = get_net(input_depth, 'skip', pad,
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


    CNN_weight_name = [
                       '1.0.1.1.weight','1.0.1.1.bias',
                        '1.1.1.1.weight','1.1.1.1.bias',
                        '1.1.4.1.weight','1.1.4.1.bias',
                        '3.1.weight','3.1.bias',
                        '6.1.weight','6.1.bias',
                        '9.1.weight','9.1.bias']

    net.to(device)
    print(net)

    # first, let's get how many trainable parameters we have in this network
    net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_input_params = input_c * input_w * input_h
    output_size = 3 * 512 * 512
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

    # let's fix the weights of networks
    # for param in net.parameters():
    #     param.requires_grad = False
    # net.eval()

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
    noise_like = torch.empty(1, input_c, input_w, input_h).to(device)
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

    #----------------------------- Get Dataset -----------------------------#
    train_loader = prepare_data(clean_image_path, corrupted_image_path)


    #----------------------------- Define Loss Function & Opt -----------------------------#
    l1_loss_func = L1_Func()
    l2_loss_func = L2_Func()


    #----------------------------- Define Optimizer -----------------------------#
    if OPTIMIZER == 'SGD':
        optimizer_net = torch.optim.SGD(net.parameters(), learning_rate_model,momentum=0.9)
        optimizer_noise = torch.optim.SGD(p_c, lr=learning_rate_model)
        optimizer_bn = torch.optim.SGD(bnnet.parameters(), lr=learning_rate_model)

    elif OPTIMIZER == 'Adam':
        optimizer_net = torch.optim.Adam(net.parameters(), learning_rate_model)
        optimizer_noise = torch.optim.Adam(p_c, lr=learning_rate_model)
        optimizer_bn = torch.optim.Adam(bnnet.parameters(), lr=learning_rate_model)

    else:
        assert False, "Optimizer is not supported!"


    #----------------------------- Start to Train -----------------------------#
    all_mse_loss = {'Epoch':[], 'MSE':[]}
    wall_time_dict = {'Epoch':[], 'Per_Time':[], 'All_Time':[]}
    all_used_time = 0


    total_loss = []
    total_loss_epoch = []
    total_epoch = []

    train_psnr_corr = []
    train_psnr_rec = []

    train_ssim_corr = []
    train_ssim_rec = []


    for epoch in range(max_epoch):
        if epoch != 0:
            # ------ start to measure the time in this iteration
            t_start = time.time()

        net.train()
        bnnet.train()
        epoch_loss = []
        for step, (X_clean, X_corrupted, idx) in enumerate(train_loader):
            # ------ start to measure the time in this iteration

            optimizer_net.zero_grad()
            optimizer_noise.zero_grad()
            optimizer_bn.zero_grad()

            #----------------------------- Get Training & Traget Dataset -----------------------------#
            X_clean, X_corrupted = X_clean.to(device), X_corrupted.to(device)

            #----------------------------- Train and Backpropagation -----------------------------#
            g_noise_input = bnnet(g_noise)
            out = net(g_noise_input)

            if LOSS == "L1":
                # print('**************')
                # print('Using L1 loss')
                # print('**************')
                loss = l1_loss_func(X_corrupted, out)  # L1
                loss = torch.sqrt(loss)
                loss =  loss + tv_weight * tv1_loss(out)

            elif LOSS == "MSE":
                # print('**************')
                # print('Using MSE')
                # print('**************')
                loss = l2_loss_func(X_corrupted, out)  # MSE
                loss = torch.sqrt(loss)
                loss = loss + tv_weight * tv1_loss(out)

            elif LOSS == 'Huber':
                # print('**************')
                # print('Using Huber')
                # print('**************')
                delta = 0.05
                loss = Huber_Loss(X_corrupted, out, delta)
                loss = torch.sqrt(loss)
                loss = loss + tv_weight * tv1_loss(out)
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


        #-----------after one epoch training, we can start to test our model ------------------#
        if epoch%print_step == 0:
            figure_name = dir_name + '00_rec.png'
            train_mean_psnr_corr, train_mean_psnr_rec, train_mean_ssim_corr, train_mean_ssim_rec = DIP_test(X_clean, X_corrupted, out,figure_name)
            train_psnr_corr.append(train_mean_psnr_corr)
            train_psnr_rec.append(train_mean_psnr_rec)
            train_ssim_corr.append(train_mean_ssim_corr)
            train_ssim_rec.append(train_mean_ssim_rec)
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
                display_psnr(train_psnr_rec, train_psnr_corr, print_step, loss_file=loss_file_name)

            #### after training, we can save the psnr values
            psnr_dict = {'epoch': total_epoch,
                         'train_psnr_corr': train_psnr_corr,
                         'train_psnr_rec': train_psnr_rec,
                         'train_ssim_corr': train_ssim_corr,
                         'train_ssim_rec': train_ssim_rec,
                         }

            if epoch % 100 == 0:  # save the results for every 100 epochs
                psnr_df = pd.DataFrame.from_dict(psnr_dict)
                psnr_file = dir_name + '00_psnr.csv'
                psnr_df.to_csv(psnr_file, index=False)


def DIP_test(test_clean_torch, test_corrupted_torch, rec_image,figure_name):
    true_image = test_clean_torch.cpu()
    corrupted_image = test_corrupted_torch.cpu()
    rec_image = rec_image.detach().cpu()
    all_images = torch.cat([true_image, corrupted_image, rec_image], dim=0)
    draw_figures(all_images, figure_name=figure_name)

    clean_torch, corr_torch = test_clean_torch, test_corrupted_torch
    X_clean = clean_torch[0:1, :]
    X_corrupted = corr_torch[0:1, :]

    #------------------- calculate PSNR & SSIM -------------------#
    psnr_corr = []
    psnr_rec = []
    ssim_corr = []
    ssim_rec = []

    cur_clean = X_clean[0]
    cur_corr = X_corrupted[0]
    cur_rec = rec_image[0]

    psnr_clean_img = cur_clean.cpu().numpy()
    psnr_clean_img = np.clip(psnr_clean_img, 0, 1)
    psnr_corrupted_img = cur_corr.cpu().numpy()
    psnr_corrupted_img = np.clip(psnr_corrupted_img, 0, 1)
    psnr_rec_ae_img = cur_rec.detach().cpu().numpy()
    psnr_rec_ae_img = np.clip(psnr_rec_ae_img, 0, 1)

    cur_psnr_corr = calcualte_PSNR(psnr_clean_img, psnr_corrupted_img)
    psnr_corr.append(cur_psnr_corr)

    cur_psnr_rec = calcualte_PSNR(psnr_clean_img, psnr_rec_ae_img)
    psnr_rec.append(cur_psnr_rec)

    cur_ssim_corr = calcualte_SSIM(psnr_clean_img, psnr_corrupted_img, multichannel=True)
    ssim_corr.append(cur_ssim_corr)

    cur_ssim_rec = calcualte_SSIM(psnr_clean_img, psnr_rec_ae_img, multichannel=True)
    ssim_rec.append(cur_ssim_rec)

    mean_psnr_corr = np.mean(psnr_corr)
    mean_psnr_rec = np.mean(psnr_rec)

    mean_ssim_corr = np.mean(ssim_corr)
    mean_ssim_rec = np.mean(ssim_rec)

    return mean_psnr_corr, mean_psnr_rec, mean_ssim_corr, mean_ssim_rec


if __name__ == '__main__':
    pass