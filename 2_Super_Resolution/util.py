import torch
import torch.utils.data as Data
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import math
import cv2
from skimage.measure import compare_psnr

###################################################################
################# Prepare Dataset into Batch Size #################
###################################################################
class IMGDataset(Dataset):
    def __init__(self, clean_data, corrupted_data, transform=None):
        self.transform = transform
        self.clean_data = clean_data
        self.corrupted_data = corrupted_data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample_clean = self.clean_data
        sample_corrputed = self.corrupted_data
        if self.transform:
            sample_clean = self.transform(sample_clean)
            sample_corrputed = self.transform(sample_corrputed)
        # print('Clean Range:')
        # print(torch.min(sample_clean))
        # print(torch.max(sample_clean))
        # print('Corr Range:')
        # print(torch.min(sample_corrputed))
        # print(torch.max(sample_corrputed))
        return (sample_clean, sample_corrputed, idx)


#################  #################
def prepare_data(clean_image_path, corrupted_image_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
    ])

    clean_data, corrupted_data = create_clean_corrupted_dataset(clean_image_path,corrupted_image_path)
    img_dataset = IMGDataset(clean_data, corrupted_data, transform=transform)
    dataloader = DataLoader(img_dataset, batch_size=1, shuffle=False, drop_last=False)
    return dataloader

def create_clean_corrupted_dataset(clean_file,corrupted_file):
    # this function will load the dataset we want
    clean_data = np.load(clean_file)['arr_0']
    corrupted_data = np.load(corrupted_file)['arr_0']
    return clean_data, corrupted_data



def show_img_clean_noise(clean_img, noise_img, fig_name):
    plt.subplot(1,2,1)
    plt.imshow(clean_img)
    plt.title('clean image')
    plt.subplot(1,2,2)
    plt.imshow(noise_img)
    plt.title('noisy image')
    plt.savefig(fig_name)
    plt.close()


def show_img_clean_noise_rec(clean_img, noise_img, rec_img, fig_name):
    plt.subplot(1,3,1)
    plt.imshow(clean_img)
    plt.title('clean image')
    plt.subplot(1,3,2)
    plt.imshow(noise_img)
    plt.title('noisy image')
    plt.subplot(1, 3, 3)
    plt.imshow(rec_img)
    plt.title('rec image')
    plt.savefig(fig_name)
    plt.close()


###################################################################
##################### Prepare Latent Code #########################
###################################################################
def prepare_code(train_loader, code_dim):
    # initialize representation space
    Z = np.empty((len(train_loader.dataset), code_dim))
    Z = np.random.randn(len(train_loader.dataset), code_dim)
    return Z


###################################################################
##################### Define Loss Function #########################
###################################################################
def L2_Func():
    return torch.nn.MSELoss()

def L1_Func():
    return torch.nn.L1Loss()

def L1_Smooth_Func():
    return torch.nn.SmoothL1Loss()

def Pseudo_Huber_Loss(true_data, pred_data, delta, device):
    t = torch.abs(true_data - pred_data)
    flag = torch.tensor(delta).to(device)
    ret = torch.where(flag==delta, delta **2 *((1+(t/delta)**2)**0.5-1), t)
    mean_loss = torch.mean(ret)
    return mean_loss

def Huber_Loss(true_data, pred_data, delta):
    t = torch.abs(true_data - pred_data)
    ret = torch.where(t <= delta, 0.5 * t ** 2, delta * t - 0.5 * delta ** 2)
    mean_loss = torch.mean(ret)
    return mean_loss

def get_L2(check_matrix):
    cur_l2_norm = torch.norm(check_matrix, p=2)
    return cur_l2_norm

def get_L1_L2(check_matrix):
    cur_l1_norm = torch.norm(check_matrix, p=1)
    cur_l2_norm = torch.norm(check_matrix, p=2)
    measurement = cur_l1_norm / cur_l2_norm
    return measurement
###################################################################
##################### Save Model and Code #########################
###################################################################
def save_model(model, model_file_name):
    torch.save(model.state_dict(), model_file_name)
    print('The trained model has been saved!')

def save_code(code, code_file_name):
    np.savez_compressed(code_file_name, code)
    print('The updated code has been saved!')

def make_dir(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            pass
            # os.system('rm {}*'.format(dir))


###################################################################
##################### custom weights initialization ###############
###################################################################
def weights_init(m):
    classname = m.__class__.__name__
    # initialize Linear layers
    if classname.find('Linear') != -1:
        #torch.nn.init.normal_(m.weight.data, 0.0, 1e-4)
        #torch.nn.init.kaiming_normal_(m.weight.data)
        print('Initialize Linear layers!')

    if classname.find('embedding') != -1:
        # torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.normal_(m.weight.data, 0.0, 1)
        print('Initialize Z layers!')

    # initialize Conv/Deconv layers
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        #torch.nn.init.constant_(m.bias.data, 0)
        print('Initialize Conv/Deconv layers!')
    # initialize Bathnorm layers
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0)
        print('Initialize Bathnorm layers!')

###################################################################
########################### plot figures ###########################
###################################################################
################### Handle Loss ###################
def display_loss(total_loss, print_step, loss_file='training_results/1_loss.png'):
    plt.plot(total_loss,label='Training Loss')
    plt.xlabel('Epoch X{}'.format(print_step))
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_file)
    plt.close()

def display_znorm(total_loss, print_step, loss_file='training_results/1_loss.png'):
    plt.plot(total_loss,label='Z F-norm')
    plt.xlabel('Epoch X{}'.format(print_step))
    plt.ylabel('F-norm')
    plt.legend()
    plt.savefig(loss_file)
    plt.close()

def display_psnr(train_psnr, test_psnr, print_step, loss_file='training_results/1_loss.png'):
    #plt.plot(train_psnr,label='PSNR-LR')
    plt.plot(test_psnr, label='PSNR-HR')
    plt.xlabel('Epoch X{}'.format(print_step))
    plt.ylabel('PSNR')
    plt.title('{}'.format(np.max(test_psnr)))
    plt.legend()
    plt.savefig(loss_file)
    plt.close()


def display_hist_err(hist_epoch, hist_err, best_epoch, figure_name):
    best_idx = np.where(hist_epoch==best_epoch)[0]
    print(best_idx)
    best_err = hist_err[best_idx]
    plt.figure(figsize=(30,15))
    plt.plot(hist_err, label='Rec Err by AE')
    plt.ylabel('Err')
    plt.xlabel('Epoch')
    index_ls = hist_epoch
    scale_ls = range(len(index_ls))
    _ = plt.xticks(scale_ls, index_ls,rotation=45)
    plt.scatter(best_idx, best_err,marker='x',c='r',s=150)
    plt.legend()
    plt.savefig(figure_name)
    plt.close()



def save_loss(total_loss,loss_file = 'training_results/corrupted_train_loss.npz'):
    print('Final loss = {}'.format(total_loss[-1]))
    total_loss = np.asarray(total_loss)
    np.savez(loss_file, total_loss)
    print('Loss has been saved!')



def draw_figures(all_images, figure_name):
    #all_images = torch.cat([true_images, fake_images], dim=0)
    #plt.figure(figsize=(32, 32))
    plt.figure()
    plt.axis("off")
    plt.title("HR Images (1st), Bicube-HR Images (2nd), DIP-HR Images (3rd)")
    plt.imshow(np.transpose(vutils.make_grid(all_images, nrow=3, padding=2, normalize=True),(1,2,0)))
    plt.savefig(figure_name)
    plt.close()



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def weights_init_noise(m):
    classname = m.__class__.__name__
    # initialize Linear layers
    if classname.find('Embedding') != -1:
        # torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.normal_(m.weight.data, 0.0, 1e-1)
        print('Initialize noise layers!')



def tv1_loss(x):
    #### here, our input must be {batch, channel, height, width}
    ndims = len(list(x.size()))
    if ndims != 4:
        assert False, "Input must be {batch, channel, height, width}"
    n_pixels = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    tot_var = torch.sum(dh) + torch.sum(dw)
    tot_var = tot_var / n_pixels
    return tot_var


#### the TV loss from DIP github
def tv_loss(x, beta=0.5):
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2)
    dw = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2)

    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))



def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def compare_psnr_y(x, y):
    return compare_psnr(rgb2ycbcr(x.transpose(1,2,0))[:,:,0], rgb2ycbcr(y.transpose(1,2,0))[:,:,0])


if __name__ == '__main__':
    pass