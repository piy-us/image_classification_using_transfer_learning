import os
import numpy as np
from tensorflow.keras.applications import MobileNet,Xception,EfficientNetB2,DenseNet121
import tensorflow 
from tensorflow.keras.applications import Xception  # Import Xception instead of MobileNet
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import Precision, Recall, AUC  # Import additional metrics
from tensorflow.keras.callbacks import EarlyStopping


# Define paths to your training and validation directories
train_data_dir = "D://farminc//datasett//train"  # Replace with your actual path
validation_data_dir = "D://farminc//datasett//validation"  # Replace with your actual path

# Set image dimensions (adjust if your images have different sizes)
img_width, img_height = 299, 299  # For Xception model

# Data Augmentation (optional but recommended)
channel_shift_range = 0.1 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Random rotation up to 20 degrees
    width_shift_range=0.2,  # Random width shift
    height_shift_range=0.2,  # Random height shift
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    brightness_range=[0.8, 1.2], 
    channel_shift_range=channel_shift_range
    )
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data using the generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

class MyTrainingStopper(tensorflow.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    # Check the difference between training and validation accuracy
    training_acc = logs.get('accuracy')
    val_acc = logs.get('val_accuracy')
    accuracy_diff = (training_acc - val_acc)

    # Stop training if accuracy difference is greater than 10
    if accuracy_diff > 0.12:
      self.model.stop_training = True
      print(f"Stopping training: Accuracy difference ({accuracy_diff:.2f}) is greater than 10%")

def build_and_evaluate_model(base_model):
    # Freeze base model layers
    base_model.trainable = False

    # Add a classifier head
    avg = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(avg)  # Additional dense layer
    x = Dropout(0.5)(x)  # Dropout regularization
    x = Dense(128, activation='relu')(x)  # Additional dense layer
    x = Dropout(0.5)(x)  # Dropout regularization
    # predictions = Dense(15, activation='softmax')(x)
    # x = Flatten()(avg)
    # x = Dropout(0.2)(x)    
    predictions = Dense(15, activation='softmax',kernel_regularizer=regularizers.l2(0.05))(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    class_weights = compute_class_weights(train_generator.classes)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=15, validation_data=validation_generator,class_weight=class_weights,callbacks=[early_stopper])

    # Evaluate the model on validation set
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f"Xception Base Accuracy (Frozen Layers): {test_acc:.4f}")

    # Unfreeze some layers for further training
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    # Compile the model again with a lower learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train the model again
    model.fit(train_generator, epochs=25, validation_data=validation_generator,class_weight=class_weights,callbacks=[early_stopper])

    # Evaluate the model on validation set
    test_loss, test_acc = model.evaluate(validation_generator)
    model_filename = "xception_insect_classifier.h5"
    model.save(model_filename)
    print(f"Model saved to: {model_filename}")

    return model, test_acc
def compute_class_weights(y):
    """Compute class weights for imbalanced dataset."""
    class_weights = {}
    n_classes = len(np.unique(y))
    class_counts = np.bincount(y)
    total_samples = np.sum(class_counts)
    for i in range(n_classes):
        class_weights[i] = total_samples / (n_classes * class_counts[i])
    return class_weights
# Evaluate Xception
xception_model, xception_acc = build_and_evaluate_model(Xception(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3)))
print(f"Xception Base Accuracy: {xception_acc:.4f}")
