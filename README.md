# Robust Differentiable SVD

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
