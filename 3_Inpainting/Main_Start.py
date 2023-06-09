from train_ip import *
import time
import datetime
from include import *




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
    gpu = 2

    print_step = 1
    max_epoch = 5000
    max_epoch = int(max_epoch / print_step) * print_step + 1

    
    tv_weight = 1e-2
    learning_rate_model = 0.05

    OPTIMIZER = 'Adam'
    LOSS = "MSE"#'Huber' or "L1"

    IP_ratio = 0.1

    input_w = 512
    input_h = 512
    input_c = 2
    width = 128
    #-----------------------------------------------------------

    # image_list = ['Peppers']

    #image_list = ['couple','fingerprint', 'hill', 'house', 'Lena512','man', 'montage', 'peppers256']

    image_list = ['barbara']
    
    ###################################### Processing images one by one #################################
    for cur_image in image_list:
        path = '../0_Dataset/Inpainting/0_IP11/IP_Dataset_{}/'.format(IP_ratio)

        # load clean image
        img_path = path + cur_image + "_img_{}.png".format(IP_ratio)
        img_pil = Image.open(img_path)
        img_np = pil_to_np(img_pil)

        # ----------------------------
        # ----- Clean Var-----
        # ----------------------------
        img_clean_var = np_to_var(img_np).type(dtype)

        output_depth = img_np.shape[0]

        # load its mask
        mask_path = path + cur_image + '_mask_{}.png'.format(IP_ratio)
        img_mask_pil = Image.open(mask_path)
        mask_np = pil_to_np(img_mask_pil)
        mask_np = np.array([mask_np[0, :, :] / np.max(mask_np)] * output_depth)

        # ----------------------------
        # ----- Mask Var-----
        # ----------------------------
        mask_var = np_to_var(mask_np).type(dtype)

        ##################Generate inpainted image
        # ----------------------------
        # ----- Noisy Var-----
        # ----------------------------
        img_noisy_var = img_clean_var * mask_var
        img_noisy_np = var_to_np(img_noisy_var)


        DIP_train(img_clean_var,
                 img_noisy_var,
                 mask_var,
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
                  tv_weight
                 )
