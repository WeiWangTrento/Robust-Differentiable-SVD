# Robust Differentiable SVD

If you find this code is helpful, please consider to cite the following paper.

```
@article{wang2021robust,
  title={Robust Differentiable SVD.},
  author={Wang, Wei and Dang, Zheng and Hu, Yinlin and Fua, Pascal and Salzmann, Mathieu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
```

## Requirements

The code might not be compatible with lower version of the specified packages.

```
Python = 3.7.2
PyTorch >= 1.1.0
Torchvision >= 0.2.2
Scipy >= 1.2.1
Numpy >= 1.16.3
tensorboardX
```

The pytorch must be GPU version, as we have not test the code on CPU machine with single GPU.
Now our code does not support multi-GPU setting.
You need to run the following command to train the model.
Here are the code for training ResNet50 on TinyImageNet

run ZCA whitening: 

CUDA_VISIBLE_DEVICES=0 python main_.py --norm=zcanormpiv2 
