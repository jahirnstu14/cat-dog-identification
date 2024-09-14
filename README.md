
# Cat-Dog-Identification
Welcome to the **Cat-Dog-Identification** project! This project demonstrates how to classify images of cats and dogs using a TensorFlow Lite model. The model is trained using Google Colab, and the application is built with Streamlit for a simple and interactive user interface. The project also utilizes OpenCV for image preprocessing.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Technologies Used](#technologies-used)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Result](#result)

## Project Overview

This project provides a web application that allows users to upload an image and receive a prediction of whether the image contains a cat or a dog. The application uses a pre-trained TensorFlow Lite model to perform the classification and is built with Streamlit for a seamless user experience.

## Technologies Used

-   **TensorFlow Lite**: For model inference.
-   **OpenCV**: For image resizing and preprocessing.
-   **Streamlit**: For building the interactive web interface.
-   **Google Colab**: For training the TensorFlow model.
- 
 [Google Colab Notebook For creating tensorflow lite mode](https://colab.research.google.com/drive/1tDVjQ-Yaz_pNz4VbhSkxgh0cn43JleX0)
## Installation

To get started with this project, follow these steps:

1.  **Clone the Repository**
    
    git clone https://github.com/jahirnstu14/cat-dog-identification.git  
    
2.  **Set Up a Virtual Environment**

   `python -m venv venv`

   `venv/Scripts/activate`
    
    
4.  **Install Dependencies**
   
    `pip install streamlit
    opencv-python
    tensorflow
    numpy
    pillow` 
    
6.  **Download the Model**
    
    Make sure to download the TensorFlow Lite model file (`cats_dogs_cnn_model.tflite`) and place it in the project directory.
    

## Usage

1.  **Run the Streamlit App**
 
    `streamlit run app.py` 
    
2.  **Open Your Browser**
    
    Once the app is running, open your web browser and go to `http://localhost:8501` to access the application.
    
3.  **Upload an Image**
    
    -   Use the file uploader to upload an image in JPG, JPEG, or PNG format.
    -   Click on the **"Predict"** button to get the prediction.


## Result
![Cat Identify Successfully](https://github.com/jahirnstu14/cat-dog-identification/blob/main/output/cat_testing_result.jpg)



![Dog Identify Successfully ](https://github.com/jahirnstu14/cat-dog-identification/blob/main/output/dog_testing_result.jpg)




