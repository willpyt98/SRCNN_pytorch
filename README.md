# SRCNN_pytorch
An implementation of SRCNN using pytorch
The orignal paper is [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/pdf/1501.00092.pdf) here

# Difference from the original paper
1. Adam is used instead of SGD
2. Removed weight initialization 
3. Trained on a different dataset. The dataset is available here: [BaiduYun](https://pan.baidu.com/s/1c0TvFyw)

# Result
The result is better than bicubic to enlarge a 2x pic.
Follow are the bicubic-img and srcnn-img
![avatar](https://raw.githubusercontent.com/willpyt98/SRCNN_pytorch/main/test/4_bicubic.jpg)
![avatar](https://raw.githubusercontent.com/willpyt98/SRCNN_pytorch/main/test/4_srcnn.jpg)

***psnr of Bicuibic image = 26.492; psnr of SRCNN image = 28.687***
