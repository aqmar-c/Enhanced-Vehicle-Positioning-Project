<a name="br1"></a> 

**Enhanced Vehicle Positioning Using Machine Learning and Neural Networks**

**Abstract**

This project addresses the challenge of vehicle positioning by employing Machine Learning

(ML) and Artificial Neural Network (ANN) approaches, focusing on the classification of

Line-of-Sight (LoS) and Non-Line-of-Sight (NLoS) signals and subsequent vehicle location

estimation using Channel State Information (CSI). Two distinct methods were explored for each

task: a K-Nearest Neighbors (KNN) algorithm for baseline predictions and a custom-designed

ANN for capturing the nonlinear complexities of the CSI data. The preprocessing of the data

involved meticulous cleaning, normalization, and feature engineering to prepare the CSI inputs

for effective learning. Through rigorous testing on an emulated unreliable network, our models

were evaluated based on accuracy, computational efficiency, and model robustness.

**Data Preprocessing**

The preprocessing of the CSI data was crucial in ensuring that the input data to the models was

clean, normalized, and conducive to learning the underlying patterns. The raw CSI data was

initially in a complex form, representing both amplitude and phase information. The following

preprocessing steps were taken:

1\. **Extraction:** The CSI and label data were extracted from the original training and

validation datasets, ensuring that the data was correctly aligned and relevant to the

tasks at hand.

2\. **Conversion:** The labels for the classification task were converted into a long tensor

format, which is a suitable data type for classification targets in PyTorch.

3\. **Modulus Calculation:** The CSI data was complex-valued, and the modulus was

computed to transform it into a real-valued representation, retaining the magnitude while

discarding the phase. This was based on the hypothesis that the magnitude of CSI

would be more indicative of the LoS/NLoS state and the vehicle's position.

4\. **Normalization:** The modulus of the CSI data was then normalized to have a mean of

zero and a standard deviation of one. This normalization was calculated from the training

set to ensure the model was not inadvertently exposed to information from the validation

set. The normalization transform was then applied to both the training and validation CSI

data, facilitating a faster and more stable convergence during the training of the models.



<a name="br2"></a> 

**LoS/NLoS Classification**

**1. KNN**

For classifying LoS and NLoS states from CSI, a K-Nearest Neighbors (KNN) classifier

was implemented, leveraging its simplicity and efficacy for classification tasks. The

methodology involved reshaping the normalized CSI data into a 2D numpy array format

suitable for scikit-learn, fitting the KNN model with the training data, and then predicting

the class labels on the validation set. The KNN classifier was initialized with three

neighbors, a commonly chosen starting point for such tasks.After prediction, the model's

accuracy was calculated by comparing predicted labels against the true labels in the

validation dataset.

Evaluation with # of neighbors = 3



![image](https://github.com/aaqmarr2001/Enhanced-Vehicle-Positioning-Project/assets/91699635/2147f212-f4a6-4ab0-95ab-7fd34cf657d1)
![image](https://github.com/aaqmarr2001/Enhanced-Vehicle-Positioning-Project/assets/91699635/b3195016-6e78-44e9-9523-c6b185e73eda)


2\. ANN Model

The ANN consists of a sequential model with the following layers:

1\. **Input Layer**: The input layer is a fully connected (dense) layer (fc1) implemented using

nn.Linear. It takes the flattened CSI data as input, where the size is inferred from

train\_data\_flat.shape[1], indicating the number of features from the

preprocessed data.

2\. **Hidden Layers**: There are three hidden layers in the architecture, each implemented as

fully connected layers with the following dimensions:

○

○

○

The first hidden layer (fc2) takes the output from fc1 and reduces the

dimension to 128 neurons.

The second hidden layer (fc3) further compresses the information to 64

neurons.

The third hidden layer (fc4) condenses the information to 32 neurons.

Each hidden layer uses a ReLU activation function to introduce non-linearity into the model,

which helps the network learn more complex patterns in the data.

1\. **Output Layer**: The final output layer (fc5) has a dimensionality corresponding to the

number of classes in the classification task (mentioned as 10 in the code comment but

should be adjusted to the actual number of LoS/NLoS classes). This layer does not

apply an activation function because the CrossEntropyLoss function used later includes

a Softmax activation.

**Training Process:**

●

●

**Data Preparation**: The CSI data is flattened and wrapped in TensorDataset objects,

which are then loaded into DataLoader objects to handle mini-batch training.

**Loss Function**: The CrossEntropyLoss function is used, which is standard for

multi-class classification problems. It combines LogSoftmax and NLLLoss in one

single class.

●

●

**Optimizer**: The Adam optimizer is employed to adjust the weights of the network with a

learning rate of 0.001. Adam is chosen for its adaptive learning rate capabilities, which

often leads to better performance on a wide range of problems.

**Training Loop**: The network is trained for a number of epochs (iterations over the entire

dataset), performing the following steps in each epoch:

○

○

○

○

Forward pass to compute the predictions (outputs) from the inputs.

Compute the loss using the criterion.

Backpropagate the errors through the network (loss.backward()).

Update the weights with the optimizer.step() function.

After each epoch, the loss is printed out to monitor the training process.



<a name="br4"></a> 

**Evaluation:**

Upon completion of training, the network is evaluated in inference mode (model.eval()) using

the validation set, where no gradients are computed. The accuracy is calculated by comparing

the predicted class (with the highest output probability) against the actual labels, giving an

indication of the model's generalization performance.



![image](https://github.com/aaqmarr2001/Enhanced-Vehicle-Positioning-Project/assets/91699635/5ad714b5-6573-4468-8993-523e3206e388)
![image](https://github.com/aaqmarr2001/Enhanced-Vehicle-Positioning-Project/assets/91699635/d456d6c1-b0af-4bfa-a04d-10c445e16929)
![image](https://github.com/aaqmarr2001/Enhanced-Vehicle-Positioning-Project/assets/91699635/5490e8de-ae82-473b-9ffb-cab447533bca)



**Position Estimation**

A regressor is used for the vehicle position estimation task because, unlike classification,

which predicts discrete labels, position estimation involves predicting continuous

numerical values that represent the vehicle's location in space.

1\. KNN Regressor

The K-Nearest Neighbors (KNN) algorithm can be used for regression as well as

classification. In the context of a regression problem, the KNN regressor

estimates the continuous output variable based on the average or median of the

K-nearest neighbors' values. The KNeighborsRegressor is initialized with

n\_neighbors=3, indicating that it will use the three nearest neighbors to make a

prediction.

The model is trained using the fit method on the training data and then makes

predictions on the validation data with the predict method. Model performance is

evaluated using Mean Squared Error (MSE), which assesses how close the

predicted positions are to the actual positions.

2\. ANN Model

**Model Definition**: The model is a sequential stack of layers with two

hidden layers and an output layer.



<a name="br6"></a> 

a. The first layer has 64 neurons and uses the ReLU activation

function. It also specifies the input shape to match the feature

space of the training data.

b. The second layer is identical to the first, with 64 neurons and ReLU

activation.

c. The output layer has 2 neurons corresponding to the x and y

coordinates for the vehicle positioning, without an activation

function as it's a regression task.

**Model Compilation**: The model uses the Adam optimizer and mean

squared error as the loss function, which is standard for regression

problems.

**Model Training**: The fit method trains the model for 100 epochs with a

batch size of 32. It also holds out 10% of the training data for validation.

**Prediction**: The model predicts the vehicle positions using the validation

dataset.

**Performance Metrics**: The Mean Squared Error (MSE) and the

R-squared value are computed to evaluate the model's performance on

the validation data. MSE measures the average squared difference

between predicted and actual values, while R-squared represents how

well the regression predictions approximate the real data points.

**Visualization**: A plot of the training and validation loss over epochs is

suggested to monitor the model's learning process and check for

overfitting or underfitting.



![image](https://github.com/aaqmarr2001/Enhanced-Vehicle-Positioning-Project/assets/91699635/5bff7d45-90d9-446f-967b-1e2b4d379cfa)

![image](https://github.com/aaqmarr2001/Enhanced-Vehicle-Positioning-Project/assets/91699635/7fd82b5f-eab1-4a6e-93b5-0a64d39e964d)



