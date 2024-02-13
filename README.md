# Citrus Diseases Detection: Project Overview  
This tool identifies diseased citrus trees by classifying citrus leaf images based on disease type.

- Leverages the `citrus_leaves` dataset from TensorFlow Datasets.
- Performs data preprocessing for standardization and label encoding.
- Builds and evaluates a Convolutional Neural Network (CNN) on the dataset.
- Develops a user-friendly client-facing API using Flask.


## Code and Resources Used 
**Python Version:** 3.10  
**Packages:** numpy, pandas, matplotlib, tensorflow, tensorflow_datasets, flask, pillow  
**Flask API Setup:**
- ```pip install -r requirements.txt```  
- ```conda env create -n <ENVNAME> -f environment.yaml``` (Anaconda Environment)
  
**Dataset:** https://www.tensorflow.org/datasets/catalog/citrus_leaves?hl=en


## Getting Data
The project utilizes the <a href="https://www.tensorflow.org/datasets/catalog/citrus_leaves?hl=en">citrus_leaves</a> dataset from TensorFlow Datasets, containing 594 PNG images of citrus leaves categorized into four labels: Black Spot, Canker, Greening, and Healthy. The images have a resolution of 256x256 pixels.

![alt text](https://github.com/polaternez/citrus-diseases-detection/blob/master/reports/figures/train_images.png "Train images")


## Data Preprocessing
- **Data Split:** Divides the dataset into 80% training and 20% testing data for robust model generalization.
- **Image Preprocessing:** Reshapes and normalizes the images to a standard format and pixel range.
- **Label Encoding:** Converts labels to one-hot encoded format for efficient processing and multi-label handling.


## Model Building 
Constructs a Convolutional Neural Network (CNN) with the following architecture:


![alt text](https://github.com/polaternez/citrus-diseases-detection/blob/master/reports/figures/model.png "Convolutional Neural Network(CNN)")


## Model Evaluation 
Measures the model's loss using categorical cross-entropy and optimizes it with the ADAM algorithm. Achieved the following results:


![alt text](https://github.com/polaternez/citrus-diseases-detection/blob/master/reports/figures/model_evaluation.png "Model Performances")


## Productionization 
Develops a user-friendly UI using Flask. The API endpoint receives image requests and returns predicted citrus disease types for each image.


![alt text](https://github.com/polaternez/citrus-diseases-detection/blob/master/reports/figures/flask-api.png "Citrus Diseases Detection API")







