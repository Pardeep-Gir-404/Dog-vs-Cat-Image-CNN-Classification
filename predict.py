import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/model.h5')

# Image size SAME as training
img_height = 150
img_width = 150

# Ask user for image filename
filename = input("Enter image filename (with extension, e.g., sample.jpg): ")
img = image.load_img(filename, target_size=(img_height, img_width))
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
class_index = np.argmax(prediction)

# Print class name
classes = ['cat','dog']
print("Predicted class:", classes[class_index])