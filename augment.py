import tensorflow as tf
import os

# Define augmentation functions using tf.image
def augment_image(image):
    #image = tf.image.random_flip_left_right(image)  # Random horizontal flip
    image = tf.image.resize_with_crop_or_pad(image, target_height = 730, target_width = 730, preserve_aspect_ratio = True)
    #image = tf.image.random_brightness(image, max_delta=0.15)  # Random brightness
    #image = tf.image.random_contrast(image, lower=0.9, upper=1.1)  # Random contrast
    #image = tf.image.random_saturation(image, lower=0.9, upper=1.1)  # Adjust saturation
    #image = tf.image.resize_with_crop_or_pad(image, 730, 730)  # Ensure consistent size
    return image

def clean_image(image, target_size = (511, 730)):
    width, height = target_size
    image = tf.image.resize_with_pad(image, height, width)
    return image

def augment_and_save_images(input_dir, output_dir, num_augmentations=1):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        save_class_path = os.path.join(output_dir, class_name)
        os.makedirs(save_class_path, exist_ok=True)

        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                # Load image
                image = tf.io.read_file(img_path)
                image = tf.image.decode_jpeg(image, channels=3)  # Decode to tensor
                #image = tf.image.resize(image, (730, 730))  # Resize
                image = tf.cast(image, tf.uint8)  # Ensure correct type
                cln_image = clean_image(image)
                cln_image = tf.keras.utils.array_to_img(cln_image)
                cln_image.save(os.path.join(save_class_path, f"cln_{img_name}"))

                # Generate and save augmented images
                for i in range(num_augmentations):
                    aug_image = clean_image(image)
                    aug_image = augment_image(aug_image)
                    aug_image = tf.keras.utils.array_to_img(aug_image)  # Convert tensor to PIL image
                    aug_image.save(os.path.join(save_class_path, f"aug_{i}_{img_name}"))

# Define dataset paths
input_directory = "augmented_images"
output_directory = "augmented_images"

# Run augmentation and save images
augment_and_save_images(input_directory, output_directory, num_augmentations=1)

print("Augmented images saved successfully!")