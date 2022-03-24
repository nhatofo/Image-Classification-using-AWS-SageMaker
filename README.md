# Image Classification using AWS SageMaker
## This assignment is a part of AWS Machine Learning Engineer Nanodegree Program.

The following tasks are performed.
- A pretrained Resnet50 model from pytorch vision library is used in the project (https://pytorch.org/vision/master/generated/torchvision.models.resnet50.html)
- Fine-tune the model with hyperparameter tuning and Network Re-shaping
- Implement Profiling and Debugging with hooks
- Deploy the model and perform inference

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio.
Download the starter files.
Download/Make the dataset available.

## Dataset
Udacity's Dog Classification Data set is used to complete the task.

The dataset can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

### Dependencies

```
Python 3.7
Pytorch AWS Instance
```
## Files Used in the notebook

- `hpo.py` - This script file contains code that will be used by the hyperparameter tuning jobs to train and test/validate the models with different hyperparameters to find the best hyperparameter
- `train_model.py` - This script file contains the code that will be used by the training job to train and test/validate the model with the best hyperparameters that we got from hyperparameter tuning
- `endpoint_inference.py` - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations) , serialization- deserialization and predictions/inferences and post-processing using the saved model from the training job.
- `train_and_deploy.ipynb` - This jupyter notebook contains all the code and the steps performed in this project and their outputs.

## Hyperparameter Tuning
- The ResNet model represents the deep Residual Learning Framework to ease the training process.
- A pair of fully connected Neural Networks has been added on top of the pretrained model to perform the classification task with 133 output nodes.
- AdamW from torch.optm is used as an optimizer.
- The Following hyperparamets are used:
    - Learning rate-  0.01x to 100x
    - eps -  1e-09 to 1e-08
    - Weight decay -  0.1x to 10x
    - Batch size -  [ 64, 128 ]

The `hpo.py` script is used to perform hyperparameter tuning.

![Hyperparameters Tuning](Snapshots/Hyperparameter_Tuning_Job.png "Hyperparameters Tuning") ![Hyperparameters](Snapshots/Hyperparameters.png "Hyperparameters")

###Training Jobs
![Training Jobs](Snapshots/Training%20Jobs.png "Training Jobs")

## Debugging and Profiling
The Graphical representation of the Cross Entropy Loss is shown below.
![Cross Entropy Loss](Snapshots/Profiling%20and%20Debugging.png "Cross Entropy Loss")

Is there some anomalous behaviour in your debugging output? If so, what is the error and how will you fix it?
- There is no smooth output line and there are different highs and lows for the batch sets.
  If not, suppose there was an error. What would that error look like and how would you have fixed it?
- A proper mix of the batches with shuffling could help the model learn better
- Trying out different neural network architecture.

### Profiler Output
The profiler report can be found [here](profiler_report/profiler-output/profiler-report.html).

![Events](Snapshots/Events%20Logging.png "Events Bridge")
## Model Deployment
- Model was deployed to a "ml.t2.medium" instance type and "endpoint_inference.py" script is used to setup and deploy our working endpoint.
- For testing purposes ,few test images are stored in the "testImages" folder.
- Those images are fed to the endpoint for inference/
- The inference is performed using both the approaches. 
    1. Using the Predictor Object 
    2. Using the boto3 client.
  
![End Point Deployment](Snapshots/End%20Point.png "End Point")

