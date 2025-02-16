import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check GPU availability
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
device = "GPU" if tf.config.experimental.list_physical_devices('GPU') else "CPU"
print(f"Using device: {device}")

# Dataset Paths (UPDATED validation folder)
train_dir = r"C:\Users\ashta\AI Doctor\dataset\archive (1)\lung_cancer_MRI_dataset\train"
val_dir = r"C:\Users\ashta\AI Doctor\dataset\archive (1)\lung_cancer_MRI_dataset\validate"

# Check if dataset paths exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training dataset not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation dataset not found: {val_dir}")
print("Dataset folders found!")

# Image Preprocessing
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Adjust brightness
    preprocessing_function=lambda img: tf.image.adjust_contrast(img, contrast_factor=1.5),  # Contrast Fix
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 🔹 Load MobileNetV2 Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

# 🔹 Build the Custom Model
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)  # Binary classification (Cancer/No Cancer)

model = Model(inputs=base_model.input, outputs=output_layer)

# 🔹 Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🔹 Train the Model
EPOCHS = 10  # Adjust based on dataset size
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# 🔹 Save the Model
model.save(r"C:\Users\ashta\AI Doctor\lung_cancer_model.h5")
print("Model saved successfully!")
