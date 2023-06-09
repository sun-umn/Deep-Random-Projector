### Generate inpainting mask
### The mask is a binary matrix that randomly drops 50% pixels

import numpy as np
import skimage.io as ski
from skimage.transform import resize
import matplotlib.pyplot as plt




def get_bernoulli_mask(image_shape, zero_fraction=0.5):
    img_mask_np = (np.random.random_sample(size=image_shape) > zero_fraction).astype(int)
    return img_mask_np



if __name__ == '__main__':
    # a = ski.imread('IP_Dataset/barbara_mask.png')
    # b = ski.imread('IP_Dataset/barbara_img.png')
    # plt.imshow(a * b, cmap='gray')
    # plt.show()

    zero_fraction = 0.5 ### how many % of pixels we want to drop/mask
    image_list = ['barbara', 'boat', 'Cameraman256', 'couple',
                  'fingerprint', 'hill', 'house', 'Lena512',
                  'man', 'montage', 'peppers256']

    for cur_img in image_list:
        cur_img_path = 'Original_Images/{}.png'.format(cur_img)
        cur_img_data = ski.imread(cur_img_path)
        h, w = cur_img_data.shape

        if h!=512:
            cur_img_data = resize(cur_img_data,(512,512))

        # now, let's get is mask
        cur_mask_data = get_bernoulli_mask(cur_img_data.shape, zero_fraction=zero_fraction)

        # plt.figure()
        # plt.hist(cur_mask_data.flatten())
        # plt.show()

        # now, let's save the clean image and its mask
        cur_IP_img_file = 'IP_Dataset_{}/{}_img_{}.png'.format(zero_fraction,cur_img,zero_fraction)
        ski.imsave(cur_IP_img_file, cur_img_data)

        cur_IP_mask_file = 'IP_Dataset_{}/{}_mask_{}.png'.format(zero_fraction,cur_img,zero_fraction)
        ski.imsave(cur_IP_mask_file,cur_mask_data)

        # visualize them
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(cur_img_data, cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(cur_mask_data, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(cur_img_data*cur_mask_data, cmap='gray')
        plt.savefig('IP_Dataset_{}/{}_vis_{}.png'.format(zero_fraction, cur_img, zero_fraction))
        plt.close()


