# CIFAR-10 classification with CNNs

This is an example of CIFAR-10 image classification task with some classical 
CNN models such as [VGG](./vgg.py), [(pre-activation) ResNet](./resnet.py), 
and [DenseNet](./densenet.py).

Here we highlight how OJAX-NN can easily tackle practical deep learning tasks 
with the support of `optax` packages for DL optimizers and `datasets` for 
working with various datsets.

to run this example, try one of the following commands:
- `python run.py vgg11 0.01`
- `python run.py resnet20 0.1`
- `python run.py densenet40k12 0.1`

Feel free to also try out other model and learning rate combinations!