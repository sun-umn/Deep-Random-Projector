import os

from train_sr import *
import time
import datetime
# from include import *
from utils import *
from utils.sr_utils import *
from utils.common_utils import *




#-----------------------------------------------------------
#-----------------------------------------------------------
# This is the Main of the code
#-----------------------------------------------------------
#-----------------------------------------------------------


if __name__ == '__main__':
    # Set random seed for reproducibility
    manualSeed = 100
    manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    ##### Set up hyper-parameters
    #-----------------------------------------------------------
    gpu = 1

    print_step = 1
    max_epoch = 5000
    max_epoch = int(max_epoch / print_step) * print_step + 1

    #-----------------------------------------------------------
    tv_weight = 0.75 # 
    learning_rate_model = 0.5
    OPTIMIZER = 'Adam'
    LOSS = "MSE"#'Huber' or "L1"
 
    factor = 4

    # image_list = ['Peppers']

    # image_list = ['baby', 'bird', 'butterfly', 'head', 'woman',
    #               'baboon', 'barbara', 'bridge', 'coastguard', 'comic', 'face', 'flowers',
    #               'foreman', 'lenna', 'man', 'monarch', 'pepper', 'ppt3', 'zebra']

    # Set5 = ['baby', 'bird', 'butterfly', 'head', 'woman']

    # Set14 = ['baboon', 'barbara', 'bridge', 'coastguard', 'comic', 'face', 'flowers',
    #          'foreman', 'lenna', 'man', 'monarch', 'pepper', 'ppt3', 'zebra']

    Set5 = ['baby']

    Set14 = ['baboon', 'barbara', 'bridge', 'coastguard', 'comic', 'face', 'flowers',
             'foreman', 'lenna', 'man', 'monarch', 'pepper', 'ppt3', 'zebra']


    image_list = Set5
    ###################################### Processing images one by one #################################
    for cur_image in image_list:
        # set up the path info
        if cur_image in Set5:
            path = '../0_Dataset/Super_Resolution/Set5/'
        elif cur_image in Set14:
            path = '../0_Dataset/Super_Resolution/Set14/'
        else:
            assert False, 'Error: please put your images under Set5 or Set14!'

        #---load image
        PLOT = False
        imsize = -1
        enforse_div32 = 'CROP' # set to be NONE for Set5/14 'CROP'  # we usually need the dimensions to be divisible by a power of two (32 in this case)
        # Starts here
        path_to_image = os.path.join(path, cur_image+'.png')
        imgs = load_LR_HR_imgs_sr(path_to_image, imsize, factor, enforse_div32)

        imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

        if PLOT:
            plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4, 12)
            print('PSNR bicubic: %.4f   PSNR nearest: %.4f' % (
                compare_psnr(imgs['HR_np'], imgs['bicubic_np']),
                compare_psnr(imgs['HR_np'], imgs['nearest_np'])))

        ### set up input noise width, height, and channel
        HR_np_shape = imgs['HR_np'].shape
        input_h = HR_np_shape[1]
        input_w = HR_np_shape[2]
        input_c = 1
        width = 128

        DIP_train(imgs,
                  learning_rate_model,
                  OPTIMIZER,
                  LOSS,
                  width,
                  input_w,
                  input_h,
                  input_c,
                  max_epoch,
                  print_step,
                  gpu,
                  cur_image,
                  tv_weight,
                  factor
                 )
