# AlexNET-on-CIFAR10
Training AlexNET model on CIFAR10 Dataset


# UNDERSTANDING OF THE PROBLEM

The problem is a multi-class image classification problem, where we have to identify the object in the images from the CIFAR dataset from a set of 1000 possible categories. However, we are using the CIFAR10 dataset, which has only 10 classes.
To do this job, we will build a deep neural net model following the AlexNET architecture, with 5 convolution networks, pooling layers, 3 fully connected layers, and dropout layers.



# THOUGHT BEHIND THE CODE

There are four main parts to the code:

Loading the Dataset: The images are of size 32*32 pixels. We transform all images to a common size of 227*227 to be able to implement the AlexNET architecture. We load the dataset available in the PyTorch Datasets class and define the training and testing dataloader.

Defining the model: We define a class for a custom model AlexNet that inherits nn.Module, which allows for automatic parameter initialization and tracking and gradient computation. We define the layers in the initialization method and define the forward pass. We will modify the last Fully connected layer to have 10 output classes, instead of 1000 as in the original architecture since we are using the CIFAR10 dataset.

Training: We set the model in training mode using model.train(). We iterate over each batch in the training dataloader for n number of epochs, setting the gradients to zero at the
beginning of each batch processing, getting the model outputs, calculating loss, and performing a backward pass to calculate the gradients, and then finally updating the model parameters. We calculate and record the loss and accuracy after each epoch.

Testing: We set the model in evaluation mode using model.eval() [which is also a functionality provided by the nn.Module class]. We iterate over the testing dataloader and get the testing loss and accuracy for each epoch.


# FINDINGS

The best-performing model achieved an accuracy of 81.36%, with three Conv2d layers, ReLU activation function and max pooling, and three fully connected layers (as in the original AlexNet model architecture). I used BCE loss function to calculate training loss.

![image](https://github.com/AishwaryaHastak/AlexNET-on-CIFAR10-/assets/31357026/2c1d5b08-1c3e-4833-aec1-bca6e60b13b1)

I used the Adam optimizing algorithm to update the weights. I tried various combinations of the hyperparameters and found that the following work best, learning rate of 0.0001, weight decay of 0.0005, momentum of 0.95, and a batch size of 128.

The better performance by decreasing the number of convolution layers could be attributed to decreasing the model complexity. Since the CIFAR10 is a smaller dataset, it can work better with a smaller model. 

We saw that decreasing the number of fully connected layers moderately worsened the model performance. This could be because the model couldn’t properly make sense of the data which might be leading to underfitting of the data. Increasing the number of fully connected layers further worsened the model performance which might be because of increased complexity.


  
