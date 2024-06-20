import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet

# Define paths to your training and validation directories
train_data_dir = "D://farminc//datasett//train"  # Replace with your actual path
validation_data_dir = "D://farminc//datasett//validation"  # Replace with your actual path

# Set image dimensions (adjust if your images have different sizes)
img_width, img_height = 229, 229  # Common size for these models

# Data Augmentation (optional but recommended)
validation_datagen = ImageDataGenerator(rescale=1./255)  # Rescale for validation set

# Load validation data using the generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='categorical')

# Load the saved model
# model_filename = "70mobile_net_insect_classifier.h5"
# model = load_model(model_filename)
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
loaded_model = load_model('775b64xception_insect_classifier.h5')

# Load new data for evaluation
# Example: Load images for evaluation
new_data_dir = "D://farminc//farm_insects"
new_data_generator = ImageDataGenerator(rescale=1./255)  # You may need to adjust this based on your preprocessing
new_data_flow = new_data_generator.flow_from_directory(
    new_data_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='categorical',
    shuffle=False  # Ensure order is maintained for evaluation
)

# Get true labels
true_labels = new_data_flow.classes

# Get predicted probabilities
predicted_probs = loaded_model.predict(new_data_flow)

# Get predicted labels
predicted_labels = np.argmax(predicted_probs, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, target_names=new_data_flow.class_indices.keys())
print(report)
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=new_data_flow.class_indices, yticklabels=new_data_flow.class_indices)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
report = classification_report(true_labels, predicted_labels, target_names=new_data_flow.class_indices.keys())
print(report)



