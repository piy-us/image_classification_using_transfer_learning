import os
import numpy as np
from tensorflow.keras.applications import DenseNet121
import tensorflow
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import legacy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import Precision, Recall, AUC  # Import additional metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

# Define paths to your training and validation directories
train_data_dir = "/content/drive/MyDrive/datasett/train"  # Replace with your actual path
validation_data_dir = "/content/drive/MyDrive/datasett/validation"

# Set image dimensions (adjust if your images have different sizes)
img_width, img_height = 224, 224  # For DenseNet121 model

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
        if accuracy_diff > 0.06:
            self.model.stop_training = True
            print(f"Stopping training: Accuracy difference ({accuracy_diff:.2f}) is greater than 10%")


def build_and_evaluate_model(base_model):
    # Freeze all layers in the base model
    base_model.trainable = False

    # Define a function to freeze or unfreeze layers
    def freeze_or_unfreeze_layers(model, layers_to_unfreeze):
        """Freezes or unfreezes a specified number of layers in a model.

        Args:
            model: The model to modify (base model in this case).
            layers_to_unfreeze: The number of layers to unfreeze from the end.
        """
        for layer in model.layers[:-layers_to_unfreeze]:
            layer.trainable = False
        for layer in model.layers[-layers_to_unfreeze:]:
            layer.trainable = True

    # Add a classifier head
    avg = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(avg)  # Additional dense layer
    x = Dropout(0.4)(x)  # Dropout regularization
    x = BatchNormalization()(x)  # Add BatchNormalization
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)  # Additional dense layer
    x = Dropout(0.2)(x)  # Dropout regularization
    predictions = Dense(15, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Training loop with gradual unfreezing
    for stage in range(5):  # Define number of unfreezing stages
        layers_to_unfreeze = (stage + 1) * 10  # Example: Unfreeze 10 layers per stage

        # Freeze or unfreeze layers based on the current stage
        freeze_or_unfreeze_layers(model, layers_to_unfreeze)

        # Compile the model (important after changing trainable layers)
        if stage>0:
          val_acc = history.history['val_accuracy'][-1]  # Get last validation accuracy
          training_acc = history.history['accuracy'][-1]  # Get last training accuracy

          if val_acc > 0.7 and training_acc > 0.7:
              # Learning rate drop if both accuracies are above 70%
              print("Learning rate dropping to 0.00001")
              optimizer = keras.optimizers.Adam(learning_rate=0.00001)  # Adjust learning rate
              model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        else:
          optimizer = keras.optimizers.Adam(learning_rate=0.0001)



        #optimizer = keras.optimizers.Adam(learning_rate=0.0001)  # Adjust learning rate as needed
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Train the model for a specified number of epochs in this stage
        early_stopper = MyTrainingStopper()
        model.fit(train_generator, epochs=10, validation_data=validation_generator, class_weight=class_weights, callbacks=[early_stopper])  # Adjust epochs

        # Evaluate the model (optional)
        test_loss, test_acc = model.evaluate(validation_generator)
        print(f"Stage {stage+1} Accuracy: {test_acc:.4f}")

    # Unfreeze remaining layers (optional for further fine-tuning)
    # freeze_or_unfreeze_layers(model, 0)  # Unfreeze all layers (optional)

    # Compile the model again with a lower learning rate (optional)
    # optimizer = keras.optimizers.Adam(learning_rate=0.00001)  # Adjust learning rate
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train the model again (optional)
    # model.fit(train_generator, epochs=10, validation_data=validation_generator, class_weight=class_weights, callbacks=[early_stopper])

    # Save the model
    model_filename = "DenseNet121_insect_classifier.h5"
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


# Calculate class weights for imbalanced training data (if applicable)
class_weights = compute_class_weights(train_generator.classes)

# Build and evaluate the DenseNet121 model with gradual unfreezing
DenseNet121_model, test_acc = build_and_evaluate_model(DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3)))
print(f"DenseNet121 Final Accuracy: {test_acc:.4f}")

