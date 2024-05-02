from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Mapping for label indices to class names
label_mapping = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Vascular lesions',
    6: 'Melanoma'
}

def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert('L')
    # Resize image to 28x28
    image = image.resize((28, 28))
    # Convert image to numpy array
    image_array = np.array(image)
    # Normalize image
    image_array = image_array / 255.0
    # Convert image array to FLOAT32
    image_array = image_array.astype(np.float32)
    # Expand dimensions to match model input shape
    image_array = np.expand_dims(image_array, axis=0)
    # Add channel dimension (for grayscale)
    image_array = np.expand_dims(image_array, axis=-1)
    # Repeat grayscale channel to match RGB channel
    image_array = np.repeat(image_array, 3, axis=-1)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    # Read the image file
    image = request.files['image']
    
    try:
        # Open and preprocess the image
        img = Image.open(image)
        img_array = preprocess_image(img)
        
        # Load the TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path='D:/Flask/model3.tflite')
        interpreter.allocate_tensors()
        
        # Set the input tensor
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensor
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the predicted class label
        predicted_label_index = np.argmax(output_data)
        predicted_label = label_mapping[predicted_label_index]
        
        # Get confidence scores for each class
        confidence_scores = {label_mapping[i]: float(output_data[0][i]) for i in range(len(output_data[0]))}
        
        return jsonify({'predicted_class': predicted_label, 'confidence_scores': confidence_scores})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
