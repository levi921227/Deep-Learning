# About DNN
## Overview
DNN is a type of artificial neural network that consists of multiple layers of 
interconnected nodes or neurons. These networks are typically composed of 
three main types of layers: the input layer, the hidden layers, and the output 
layer. The input layer receives the raw data, the hidden layers process this data 
through weighted connections and nonlinear activation functions, and the output 
layer produces the final result or prediction. 

Compared to MLP, DNN has higher number of hidden layers, which allows it to 
learn and model more complex patterns and relationships in data. Also, DNN 
uses backpropagation to update the weight so that it can improve the accuracy 
of prediction. However, the additional layers also introduce more cost and 
typically require more data and longer training times than MLP. 

## Gradient descent
Gradient descent is used to minimize the cost function in machine learning. The 
purpose of gradient descent is to find the optimal set of parameters that 
minimize the error between the model's predictions and the actual target values.  
First, gradient descent will randomly choose the model parameters then it’ll 
calculate the loss. Second, gradient descent will calculate the gradient to obtain 
the direction and rate in order to update the parameters. Gradient descent will 
repeat these action until the gradient become very small, indicating that the loss 
function has reached a local minimum. 


## data set
- nmist : The dataset that include many types of handwrite number picture, it's picture is 28*28 pixels
  
  ![DNN_raw](https://github.com/user-attachments/assets/a78fddea-b78b-4234-bf6c-b3a2d3fbf531)

- cifar10 : The dataset that include 10 types of picture(plane, cat, bird...), it's picture has 3 channels
  
  ![DNN_raw_cifar](https://github.com/user-attachments/assets/62798c89-707d-436b-9ba0-209349f0bea1)

  
