print("SCRIPT STARTED")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

train_dir = "dataset/train"
test_dir = "dataset/test"

print("Train exists:", os.path.exists(train_dir))
print("Test exists:", os.path.exists(test_dir))

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=8,
    class_mode='categorical'
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=8,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(2,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("TRAINING STARTED")

model.fit(train_data, epochs=5, validation_data=test_data)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")

print("MODEL SAVED SUCCESSFULLY")
input("PRESS ENTER TO EXIT")
