# mask-detection-with-DL


# **Face Mask Detection using Deep Learning**

## **Introduction**
Face mask detection is a crucial application of deep learning and computer vision, especially in public health and safety. In this notebook, we implement a model to detect whether a person is wearing a mask or not using a Convolutional Neural Network (CNN) trained on labeled images.

## **Objectives**
- Build a deep learning model to classify masked and unmasked faces.
- Utilize OpenCV’s deep learning module for face detection.
- Apply the trained model to static images and pre-recorded videos.
- Save the trained model for future use.

## **Dataset**
We use a dataset containing images of individuals:
- **With Mask** 🟩
- **Without Mask** 🟥  

The dataset is preprocessed to ensure balanced representation and optimal model performance.

## **Workflow**
1. **Data Preprocessing**  
   - Load and augment image data.  
   - Convert images to numerical arrays and normalize pixel values.  
   - Split into training and validation sets.  

2. **Model Training**  
   - Train a CNN-based model or leverage a pre-trained network (e.g., MobileNetV2).  
   - Use appropriate loss functions and optimizers.  

3. **Evaluation & Model Saving**  
   - Assess performance using accuracy and loss metrics.  
   - Save the trained model to avoid re-training in future runs.  

4. **Face Mask Detection in Images & Videos**  
   - Use OpenCV’s DNN face detector to locate faces.  
   - Apply the trained model to classify mask usage.  
   - Display results with bounding boxes and labels.  

## **Implementation**
We implement the project using:
- **TensorFlow/Keras**: For model training and inference.
- **OpenCV**: For face detection.
- **Matplotlib**: For visualizing results.

## **Model Performance**
- **test_loss**:  0.054265428334474564
- **test_accuracy**:  0.979345977306366
- **Memory consumption** : 0 bytes
## **Using the Saved Model**

Once trained, the model is saved as a `.keras` file. To reuse it:
```python
from tensorflow.keras.models import load_model
model = load_model("mask_detector.keras")
```
example : 
![image](https://github.com/user-attachments/assets/69eae722-0e96-4997-ab28-54b43e1b7f66)
