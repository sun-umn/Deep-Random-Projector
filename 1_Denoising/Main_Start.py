from train_denoising import *
import time
import datetime

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

    # Set up hyper-parameters
    #-----------------------------------------------------------
    gpu = 0
    corr_level = 2

    corr_type = 'Gaussian_Noise'

    corr_type_level = corr_type + '_' + str(corr_level)
    print_step = 1
    max_epoch = 5000
    max_epoch = int(max_epoch / print_step) * print_step + 1

    tv_weight = 0.45
    learning_rate_model = 1e-1
    OPTIMIZER = 'Adam'
    LOSS = "MSE"# or 'Huber' or "L1"
  
    input_w = 512
    input_h = 512
    input_c = 3
    width = 64
    #-----------------------------------------------------------

    image_list = ['Peppers']

    #image_list = ['Baboon', 'F16', 'House',
                  #'kodim01', 'kodim02', 'kodim03',
                  #'kodim12', 'Lena', 'Peppers']
    ###################################### Processing images one by one #################################
    for cur_image in image_list:

        clean_image_path = '../0_Dataset/{}/Clean/{}_clean.npz'.format(corr_type, cur_image)
        corrupted_image_path = '../0_Dataset/{}/{}/{}_{}.npz'.format(corr_type,corr_type_level,cur_image,corr_type_level)

        DIP_train(learning_rate_model,
                  OPTIMIZER,
                  LOSS,
                  width,
                  input_w,
                  input_h,
                  input_c,
                  clean_image_path,
                  corrupted_image_path,
                  max_epoch,
                  print_step,
                  gpu,
                  cur_image,
                  tv_weight
                 )