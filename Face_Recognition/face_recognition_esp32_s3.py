

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import time

# Path for the face image database
dataset_path = 'Dataset_path'

# Ensure the trainer directory exists
if not os.path.exists('trainer'):
    os.makedirs('trainer')

# Function to preprocess data
def preprocess_data(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    labels = []

    for imagePath in imagePaths:
        label = int(os.path.split(imagePath)[-1].split(".")[1])  # Extract label from filename
        label -= 1  # Adjust labels to be in range [0, num_classes - 1]
        img = Image.open(imagePath).convert('RGB')  # Convert to RGB
        img = img.resize((128, 128))  # Resize to MobileNetV2 input size
        img = np.array(img, dtype='float32') / 255.0  # Normalize pixel values to [0, 1]
        faces.append(img)
        labels.append(label)

    return np.array(faces), np.array(labels)

# Load and preprocess data
print("[INFO] Loading and preprocessing data...")
faces, labels = preprocess_data(dataset_path)

# Split the dataset into training, validation, and testing sets
print("[INFO] Splitting dataset into training, validation, and testing sets...")
X_train, X_temp, y_train, y_temp = train_test_split(faces, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Data augmentation
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3), alpha=0.35)
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling
x = tf.keras.layers.Dropout(0.3)(x)  # Increased Dropout layer to reduce overfitting
predictions = Dense(3, activation='softmax')(x)  # Softmax output layer for 3 classes
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("[INFO] Training the model...")
history = model.fit(
    data_augmentation.flow(X_train, y_train, batch_size=2),
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Evaluate the model on the test set
print("[INFO] Evaluating the model on the test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=16)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Additional evaluation using dataset objects
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy using dataset object:', accuracy)

# Latency measurement for inference
start_time = time.time()
_ = model.predict(X_test[:1])  # Predict on one sample
end_time = time.time()
latency = end_time - start_time
print(f"Latency for one inference: {latency:.4f} seconds")

model_path = 'trainer/mobilenetv2_face_recognition.h5'
model.save(model_path)
print(f"[INFO] Model trained and saved to '{model_path}'")
# Model size (on disk)
model_size = os.path.getsize('trainer/mobilenetv2_face_recognition.h5') / (1024 * 1024)  # MB
print(f"Model size: {model_size:.2f} MB")

# Print model summary to get the number of parameters
model.summary()

# Classification Report (Precision, Recall, F1-score)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot predictions to class labels
print(classification_report(y_test, y_pred_classes))

# Save the trained model

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Quantitative Evaluation
def get_flops(model):
    # Create a concrete function from the Keras model
    concrete_func = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete_func.get_concrete_function(tf.TensorSpec([1, 128, 128, 3], tf.float32))

    # Convert to frozen graph
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    # Create a new graph and import the frozen graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    # Profile the graph
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

    run_meta = tf.compat.v1.RunMetadata()
    opts = ProfileOptionBuilder.float_operation()  # Options to count FLOPs
    flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, options=opts)

    return flops.total_float_ops


total_params = model.count_params()
flops = get_flops(model)
macs = flops / 2

# Latency
input_data = np.random.rand(1, 128, 128, 3).astype(np.float32)
start_time = time.time()
_ = model.predict(input_data)
latency = (time.time() - start_time) * 1000

# Log results
with open('trainer/test_metrics.txt', 'w') as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_accuracy}\n")
    f.write(f"Total Parameters: {total_params}\n")
    f.write(f"MACs: {macs}\n")
    f.write(f"Latency: {latency:.2f} ms\n")

print("[INFO] Test metrics saved to 'trainer/test_metrics.txt'")

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
# plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(2)
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy using dataset object:', accuracy)

import tensorflow as tf
import pathlib

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Dynamic range quantization
tflite_model = converter.convert()

# Ensure the directory exists
output_dir = pathlib.Path('model')
output_dir.mkdir(parents=True, exist_ok=True)

# Save the TFLite model
tflite_model_file = output_dir / 'model-float32.tflite'
tflite_model_file.write_bytes(tflite_model)

def representative_data_get():
    for input_value, _ in tf.data.Dataset.from_tensor_slices((faces, labels)).take(25):
        # Add a batch dimension to the input tensor
        input_value = tf.expand_dims(input_value, axis=0)  # Shape becomes (1, 96, 96, 3)
        yield [input_value]

# Step 2: Convert the model to TFLite format and apply INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Dynamic range quantization
converter.representative_dataset = representative_data_get
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Step 3: Convert the model
tflite_model_quant = converter.convert()

# Step 4: Save the quantized model
output_dir = pathlib.Path('model')
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the 'model' directory exists
tflite_model_quant_file = output_dir / 'model-int8.tflite'
tflite_model_quant_file.write_bytes(tflite_model_quant)

print("[INFO] Quantized model saved to 'model/model-int8.tflite'")

# Evaluate TFLite models
def evaluate_tflite_model(tflite_model_path, dataset):
    print(f"[INFO] Evaluating TFLite model: {tflite_model_path}")

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct_predictions = 0
    total_samples = 0

    for images, labels in dataset:
        # Preprocess input
        interpreter.set_tensor(input_details[0]['index'], images)

        # Run inference
        interpreter.invoke()

        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_classes = np.argmax(predictions, axis=1)

        # Update accuracy
        correct_predictions += np.sum(predicted_classes == labels)
        total_samples += labels.shape[0]

    accuracy = correct_predictions / total_samples
    print(f"Accuracy of {tflite_model_path}: {accuracy}")
    return accuracy

# Prepare test dataset
prepared_test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)

# Evaluate both models
evaluate_tflite_model('model/model-float32.tflite', prepared_test_dataset)
evaluate_tflite_model('model/model-int8.tflite', prepared_test_dataset)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf

# Function to evaluate TFLite models with additional metrics (Confusion Matrix, Precision, Recall, Latency)
def evaluate_tflite_model_with_metrics(tflite_model_path, dataset):
    print(f"[INFO] Evaluating TFLite model: {tflite_model_path}")

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predicted_classes = []

    # Latency measurement
    start_time = time.time()

    for images, labels in dataset:
        # Preprocess input
        interpreter.set_tensor(input_details[0]['index'], images)

        # Run inference
        interpreter.invoke()

        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_classes = np.argmax(predictions, axis=1)

        # Update accuracy
        correct_predictions += np.sum(predicted_classes == labels)
        total_samples += labels.shape[0]

        # Store labels and predictions for confusion matrix and metrics
        all_labels.extend(labels.numpy())
        all_predicted_classes.extend(predicted_classes)

    # Latency measurement for the entire dataset
    end_time = time.time()
    latency = end_time - start_time

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    print(f"Accuracy of {tflite_model_path}: {accuracy:.4f}")
    print(f"Total Latency: {latency:.4f} seconds")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predicted_classes)
    print("Confusion Matrix:")
    print(cm)

    # Plot the Confusion Matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 1', 'Class 2', 'Class 3'])
    plt.title(f'Confusion Matrix for {tflite_model_path}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Classification Report (Precision, Recall, F1-score)
    print(classification_report(all_labels, all_predicted_classes, target_names=['Class 1', 'Class 2', 'Class 3']))

    return accuracy, latency, cm

# Prepare test dataset
prepared_test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)

# Evaluate both models
accuracy_float32, latency_float32, cm_float32 = evaluate_tflite_model_with_metrics('model/model-float32.tflite', prepared_test_dataset)
accuracy_int8, latency_int8, cm_int8 = evaluate_tflite_model_with_metrics('model/model-int8.tflite', prepared_test_dataset)

# You can also compare the results:
print(f"Comparison of Results:")
print(f"Float32 Model - Accuracy: {accuracy_float32:.4f}, Latency: {latency_float32:.4f}")
print(f"Int8 Model - Accuracy: {accuracy_int8:.4f}, Latency: {latency_int8:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf

# Function to evaluate TFLite models with additional metrics (Confusion Matrix, Precision, Recall, Latency)
def evaluate_tflite_model_with_metrics(tflite_model_path, dataset):
    print(f"[INFO] Evaluating TFLite model: {tflite_model_path}")

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_predicted_classes = []

    # Latency measurement
    start_time = time.time()

    for images, labels in dataset:
        # Preprocess input
        interpreter.set_tensor(input_details[0]['index'], images)

        # Run inference
        interpreter.invoke()

        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_classes = np.argmax(predictions, axis=1)

        # Update accuracy
        correct_predictions += np.sum(predicted_classes == labels)
        total_samples += labels.shape[0]

        # Store labels and predictions for confusion matrix and metrics
        all_labels.extend(labels.numpy())
        all_predicted_classes.extend(predicted_classes)

    # Latency measurement for the entire dataset
    end_time = time.time()
    latency = end_time - start_time

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    print(f"Accuracy of {tflite_model_path}: {accuracy:.4f}")
    print(f"Total Latency: {latency:.4f} seconds")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predicted_classes)
    print("Confusion Matrix:")
    print(cm)

    # Plot the Confusion Matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 1', 'Class 2', 'Class 3'])
    plt.title(f'Confusion Matrix for {tflite_model_path}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Classification Report (Precision, Recall, F1-score)
    print(classification_report(all_labels, all_predicted_classes, target_names=['Class 1', 'Class 2', 'Class 3']))

    return accuracy, latency, cm

# Prepare test dataset
prepared_test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)

# Evaluate both models
accuracy_float32, latency_float32, cm_float32 = evaluate_tflite_model_with_metrics('model/model-float32.tflite', prepared_test_dataset)
accuracy_int8, latency_int8, cm_int8 = evaluate_tflite_model_with_metrics('model/model-int8.tflite', prepared_test_dataset)

# You can also compare the results:
print(f"Comparison of Results:")
print(f"Float32 Model - Accuracy: {accuracy_float32:.4f}, Latency: {latency_float32:.4f}")
print(f"Int8 Model - Accuracy: {accuracy_int8:.4f}, Latency: {latency_int8:.4f}")