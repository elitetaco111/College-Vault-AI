import tensorflow as tf
import numpy as np
import pandas as pd
import os
from keras.api.preprocessing import image

#load the trained model
model_path = "modelv0.8.0.keras"  #Replace with your model's path
model = tf.keras.models.load_model(model_path)

#class labels
class_labels = []  #Adjust as needed

#preprocess an image
def preprocess_image(img_path, target_size=(880, 500)):  #Adjust img size
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  #Normalize to [0,1] range
    return img_array

#process all images in a folder
def process_images_in_folder(folder_path, output_csv="predictions.csv"):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print("No images found in the specified folder.")
        return
    
    #preprocess in a batch
    image_arrays = np.array([preprocess_image(img_path) for img_path in image_paths])
    
    #predict in a batch
    predictions = model.predict(image_arrays)
    
    #return class indices
    predicted_indices = [np.argmax(pred) for pred in predictions]
    
    #save to CSV
    df = pd.DataFrame({"Image Path": image_paths, "Predicted Class Index": predicted_indices})
    df.to_csv(output_csv, index=False)
    
    print(f"Predictions saved to {output_csv}")

#Run the batch processing
folder_path = "predict_images"  #image folder path (adjust as needed)
process_images_in_folder(folder_path, "predictions.csv") #output to csv named predictions.csv
