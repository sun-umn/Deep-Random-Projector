# Deep-Random-Projector
## Deep Random Projector: Accelerated Deep Image Prior

This is the official implementation of our paper *Deep Random Projector: Accelerated Deep Image Prior* which has been accepted to the [CVPR2023](https://cvpr2023.thecvf.com/). You can find our paper via [this link](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Deep_Random_Projector_Accelerated_Deep_Image_Prior_CVPR_2023_paper.html).


## Set Up the Environment

1. Get and clone the github repository:

   `git clone https://github.com/sun-umn/Deep-Random-Projector/`

2. Switch to `Deep-Random-Projector` :

   `cd XXX/Deep-Random-Projector`  
   (*Note*: `XXX` here indicates the upper directory of `Deep-Random-Projector`. For example, if you clone `Deep-Random-Projector` under `/home/Download`, then you should replace `XXX` with `/home/Download`.)

3. Create a new conda environment with the YML file we provide:

    `conda env create -f environment.yml`
   
4.  Activate conda environment and you are now ready to explpre the codes/models!
    
    `conda activate pytorch_py3.6`
    
    
## Explore the Codes/Models

- **0_Dataset**: we provide a ready-to-use *example* dataset.

- **1_Denoising**: the code for image denoising.

- **2_Super_Resolution**: the code for image super-solution.

- **3_Inpainting**: the code for image inpainting.

We have provided the detailed inline comments in each Python file. You can modify any parameters you want to explore the models. Or, you can try our method by simply running the command below:

```
python Main_Start.py
```


## Citation/BibTex

More technical details and experimental results can be found in our paper:

Taihui Li, Hengkang Wang, Zhong Zhuang, and Ju Sun. "Deep Random Projector: Accelerated Deep Image Prior." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18176-18185. 2023.

```
@inproceedings{li2023deep,
  title={Deep Random Projector: Accelerated Deep Image Prior},
  author={Li, Taihui and Wang, Hengkang and Zhuang, Zhong and Sun, Ju},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18176--18185},
  year={2023}
}
```


## Acknowledgements
Our code is based on the [deep image prior (DIP)](https://github.com/DmitryUlyanov/deep-image-prior) and [deep decoder (DD)](https://github.com/reinhardh/supplement_deep_decoder). 


## Contact
- Taihui Li, lixx5027@umn.edu, [https://taihui.github.io/](https://taihui.github.io/)
- Hengkang Wang, wang9881@umn.edu, [https://www.linkedin.com/in/hengkang-henry-wang-a1b293104/](https://www.linkedin.com/in/hengkang-henry-wang-a1b293104/)
- Zhong Zhuang, zhuan143@umn.edu, [https://scholar.google.com/citations?user=rGGxUQEAAAAJ](https://scholar.google.com/citations?user=rGGxUQEAAAAJ)
- Ju Sun, jusun@umn.edu, [https://sunju.org/](https://sunju.org/)
