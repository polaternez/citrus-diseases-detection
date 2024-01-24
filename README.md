# Citrus Diseases Detection: Project Overview  
This tool was created to identify diseased citrus trees by classifying citrus leaf images based on disease type. 

* Take `citrus_leaves` dataset from TensorFlow Datasets
* Apply data preprocessing steps to the dataset
* Build a Convolutional Neural Network (CNN), then evaluate them on test dataset
* Built a client facing API using Flask 

Note: This project was made for educational purposes.

## Code and Resources Used 
**Python Version:** 3.10  
**Packages:** numpy, pandas, matplotlib, tensorflow, tensorflow_datasets, flask, pillow  
**Flask API Requirements:**  ```pip install -r requirements.txt```  
**Create Anaconda Environment for Flask API:**  ```conda env create -n <ENVNAME> -f environment.yaml```  
**Dataset:** https://www.tensorflow.org/datasets/catalog/citrus_leaves?hl=en

## Getting Data
We utilize the <a href="https://www.tensorflow.org/datasets/catalog/citrus_leaves?hl=en">citrus_leaves</a> dataset from TensorFlow Datasets. This dataset consists of 594 PNG images of citrus leaves, categorized into four labels: Black Spot, Canker, Greening, and Healthy. The images have a resolution of 256x256 pixels.

![alt text](https://github.com/polaternez/citrus-diseases-detection/blob/master/reports/figures/train_images.png "Train images")


## Data Preprocessing

* **Data Split**: We split the dataset into 80% training data and 20% testing data to ensure the model generalizes well to unseen data.
* **Image Preprocessing**: We reshape and normalize the images to a standard format and pixel range. 
* **Label Encoding**: We convert the labels to a one-hot encoded format for efficient processing and learning. This representation allows the model to handle multiple labels effectively.


## Model Building 

We built a Convolutional Neural Network (CNN) with the following architecture:

![alt text](https://github.com/polaternez/citrus-diseases-detection/blob/master/reports/figures/model.png "Convolutional Neural Network(CNN)")

## Model Evaluation 

We measure the model's loss using categorical cross-entropy and optimize it with the ADAM algorithm. After training, we obtain the following results:

![alt text](https://github.com/polaternez/citrus-diseases-detection/blob/master/reports/figures/model_evaluation.png "Model Performances")

## Productionization 
In this step, we developed the UI using Flask. The API endpoint help receives a request containing images and returns the predicted type of citrus disease for each image.

![alt text](https://github.com/polaternez/citrus-diseases-detection/blob/master/reports/figures/flask-api.png "Citrus Diseases Detection API")







