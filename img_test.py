import tensorflow as tf
import numpy as np
from keras.api.preprocessing import image

#Load the trained model
model_path = "modelv0.8.0.keras"  #model's path
model = tf.keras.models.load_model(model_path)

#Function to preprocess an image
def preprocess_image(img_path, target_size=(880, 500)):  #Adjust target size
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  #Normalize to [0,1] range
    img_array = np.expand_dims(img_array, axis=0)  #Add batch dimension
    return img_array

#Function to make a prediction
def predict_image(img_path, class_labels=None):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    if class_labels:
        predicted_class = class_labels[np.argmax(predictions)]  #Get class name (once implemented)
        confidence = np.max(predictions)
        return f"Predicted: {predicted_class} with {confidence:.2%} confidence"
    else:
        return f"Predicted class index: {np.argmax(predictions)}, Confidence: {np.max(predictions):.2%}"

#Example usage
image_path = "michy.png"  #Replace with test image
class_labels = []  #Fill with classes (once implemented)

print(predict_image(image_path, class_labels))
