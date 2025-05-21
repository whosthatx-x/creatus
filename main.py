import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import zipfile
from io import BytesIO
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from chatbot import chatbot_response

# Set page config
st.set_page_config(page_title="Creatus", page_icon='logo.png', menu_items={
    'About': "# :red[Creator]:blue[:] :violet[Pranav Lejith(:green[Amphibiar])]"
}, layout='wide')

# Initialize session state keys
if 'labels' not in st.session_state:
    st.session_state['labels'] = {}
if 'num_classes' not in st.session_state:
    st.session_state['num_classes'] = 0
if 'label_mapping' not in st.session_state:
    st.session_state['label_mapping'] = {}
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = None
if 'is_developer' not in st.session_state:
    st.session_state['is_developer'] = False
if 'show_developer_splash' not in st.session_state:
    st.session_state['show_developer_splash'] = False
if 'initial_load' not in st.session_state:
    st.session_state['initial_load'] = True
if 'dev_command_entered' not in st.session_state:
    st.session_state['dev_command_entered'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'show_chat' not in st.session_state:
    st.session_state['show_chat'] = False

# Developer authentication (hidden from normal users)
developer_commands = [
    'override protocol-amphibiar', 'override command-amphibiar', 
    'command override-amphibiar', 'command override-amphibiar23', 
    'control override-amphibiar', 'system override-amphibiar', 'user:amphibiar'
]

# Custom HTML for splash screen with typewriter effect
def create_splash_html(text, color):
    return f"""
    <style>
    .typewriter h1 {{
      overflow: hidden;
      color: {color};
      white-space: nowrap;
      margin: 0 auto;
      letter-spacing: .15em;
      border-right: .15em solid orange;
      animation: typing 3.5s steps(30, end), blink-caret .5s step-end infinite;
    }}

    @keyframes typing {{
      from {{ width: 0 }}
      to {{ width: 100% }}
    }}

    @keyframes blink-caret {{
      from, to {{ border-color: transparent }}
      50% {{ border-color: orange }}
    }}
    </style>
    <div class="typewriter">
        <h1>{text}</h1>
    </div>
    """

# Chat interface
def chat_interface():
    st.sidebar.subheader("Chat with Creatus AI")
    
    # Display chat history
    for message in st.session_state['chat_history']:
        with st.sidebar.chat_message(message["role"]):
            st.sidebar.write(message["content"])

    # Chat input
    user_input = st.sidebar.chat_input("Ask about Creatus...")

    if user_input:
        # Add user message to chat history
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        
        # Get AI response
        ai_response = chatbot_response(user_input)
        
        # Add AI response to chat history
        st.session_state['chat_history'].append({"role": "assistant", "content": ai_response})
        
        # Rerun to update the chat display
    

# Main content
def main_content():
    st.title(":red[Creatus (Model Creator)]")

    # Sidebar for label input
    st.sidebar.title(":blue[Manage Labels]")

    # Add reset button for developer mode
    if st.session_state['is_developer']:
        if st.sidebar.button("Reset to Normal User"):
            st.session_state['is_developer'] = False
        

    label_input = st.sidebar.text_input("Enter a new label:")
    if st.sidebar.button("Add Label"):
        if label_input in developer_commands:
            st.session_state['is_developer'] = True
            st.session_state['show_developer_splash'] = True
        
        elif label_input and label_input not in st.session_state['labels']:
            st.session_state['labels'][label_input] = []
            st.session_state['num_classes'] += 1
            st.sidebar.success(f"Label '{label_input}' added!")
        else:
            st.sidebar.warning("Label already exists or is empty.")

    # Display labels with delete buttons
    st.sidebar.subheader("Existing Labels")
    for label in list(st.session_state['labels'].keys()):
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.write(label)
        if col2.button(":red[-]", key=f"delete_{label}"):
            del st.session_state['labels'][label]
            st.session_state['num_classes'] -= 1

    # Dropdown to select model export format
    export_format = st.sidebar.selectbox("Select model export format:", options=["tflite", "h5"])

    # Display the existing labels and allow image upload in rows
    if st.session_state['num_classes'] > 0:
        num_columns = 3  # Adjust this value for the number of columns you want
        cols = st.columns(num_columns)
        
        for i, label in enumerate(st.session_state['labels']):
            with cols[i % num_columns]:  # Wrap to the next line
                st.subheader(f"Upload images for label: {label}")
                uploaded_files = st.file_uploader(f"Upload images for {label}", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'], key=label)
                
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        image_data = image.load_img(uploaded_file, target_size=(64, 64))
                        image_array = image.img_to_array(image_data)
                        st.session_state['labels'][label].append(image_array)
                    st.success(f"Uploaded {len(uploaded_files)} images for label '{label}'.")

    # Advanced options in sidebar
    with st.sidebar.expander("Advanced Options"):
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10)
        if st.session_state['is_developer']:
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)
            model_architecture = st.selectbox("Model Architecture", ["Simple CNN", "VGG-like", "ResNet-like", "Custom"])
            if model_architecture == "Custom":
                num_conv_layers = st.number_input("Number of Convolutional Layers", min_value=1, max_value=10, value=3)
                num_dense_layers = st.number_input("Number of Dense Layers", min_value=1, max_value=5, value=2)
                activation_function = st.selectbox("Activation Function", ["relu", "leaky_relu", "elu", "selu"])
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
            data_augmentation = st.checkbox("Enable Data Augmentation")
            if data_augmentation:
                rotation_range = st.slider("Rotation Range", 0, 180, 20)
                zoom_range = st.slider("Zoom Range", 0.0, 1.0, 0.2)
                horizontal_flip = st.checkbox("Horizontal Flip")
                vertical_flip = st.checkbox("Vertical Flip")
        else:
            learning_rate = 0.001
            batch_size = 32
            model_architecture = "Simple CNN"
            optimizer = "Adam"
            data_augmentation = False

    # Button to train the model
    if st.session_state['num_classes'] > 1:
        if st.button("Train Model"):
            all_images = []
            all_labels = []
            st.session_state['label_mapping'] = {label: idx for idx, label in enumerate(st.session_state['labels'].keys())}
            
            for label, images in st.session_state['labels'].items():
                all_images.extend(images)
                all_labels.extend([st.session_state['label_mapping'][label]] * len(images))
            
            if len(all_images) > 0:
                st.write("Training thxp model...")
                progress_bar = st.progress(0)  # Initialize progress bar
                st.session_state['model'], st.session_state['metrics'] = train_model(
                    all_images, all_labels, st.session_state['num_classes'], epochs, progress_bar,
                    learning_rate=learning_rate, batch_size=batch_size, model_architecture=model_architecture,
                    optimizer=optimizer, data_augmentation=data_augmentation
                )
                st.toast('Model Trained Successfully',icon='✅')
                st.success("Model trained!")

                # Display model performance metrics
                if st.session_state['metrics'] is not None:
                    st.subheader("Model Performance Metrics")
                    metrics = st.session_state['metrics']
                    st.write(f"Accuracy: {metrics['accuracy']:.4f}")
                    st.write(f"Precision: {metrics['precision']:.4f}")
                    st.write(f"Recall: {metrics['recall']:.4f}")
                    st.write(f"F1 Score: {metrics['f1_score']:.4f}")

                    # Visualize metrics
                    fig, ax = plt.subplots()
                    metrics_names = list(metrics.keys())
                    metrics_values = list(metrics.values())
                    ax.bar(metrics_names, metrics_values)
                    ax.set_ylim(0, 1)
                    ax.set_title("Model Performance Metrics")
                    ax.set_ylabel("Score")
                    for i, v in enumerate(metrics_values):
                        ax.text(i, v, f"{v:.4f}", ha='center', va='bottom')
                    st.pyplot(fig)
            else:
                st.error("Please upload some images before training.")
    else:
        st.warning("At least two labels are required to train the model.")

    # Option to test the trained model
    if st.session_state['model'] is not None:
        st.subheader("Test the trained model with a new image")
        test_image = st.file_uploader("Upload an image to test", type=['jpg', 'jpeg', 'png','webp'], key="test")
        
        if test_image:
            # Show image preview
            test_image_data = image.load_img(test_image, target_size=(64, 64))
            st.image(test_image_data, caption="Uploaded Image", use_column_width=True)

            test_image_array = image.img_to_array(test_image_data)
            predicted_label, confidence = test_model(st.session_state['model'], test_image_array, st.session_state['label_mapping'])

            st.write(f"Predicted Label: {predicted_label}")
            st.slider("Confidence Level (%)", min_value=1, max_value=100, value=int(confidence * 100), disabled=True)

    # Button to download the model
    if st.session_state['model'] is not None and st.button("Download Model"):
        try:
            buffer = save_model(st.session_state['model'], export_format, st.session_state['label_mapping'])
            
            st.download_button(
                label="Download the trained model and usage code",
                data=buffer,
                file_name=f"trained_model_{export_format}.zip",
                mime="application/zip"
            )
        except Exception as e:
            st.error(f"Error: {e}")

    # Chat button
    if st.button("Ask AI", key="open_chat"):
        st.session_state['show_chat'] = not st.session_state['show_chat']

    if st.session_state['show_chat']:
        chat_interface()

    st.sidebar.write("This app was created by :red[Pranav Lejith](:violet[Amphibiar])")
    st.sidebar.subheader(":orange[Usage Instructions]")
    st.sidebar.write(""" 
    1) Manage Labels: Enter a new label and upload images for that label.
                     
    2) Train Model: After uploading images for at least two labels, you can train the model.
                     
    3) Test Model: Once the model is trained, you can test it with new images and see predictions along with confidence levels.
                     
    4) Download Model: Finally, you can download the trained model in TensorFlow Lite or .h5 format for use in other applications. Tensorflow lite model is better because it is smaller in size as compared to the .h5 model so it can be used in many applications which have a file size limit.
                     

    """, unsafe_allow_html=True)
    st.sidebar.subheader(":red[Warning]")
    st.sidebar.write('The code might produce a ghosting effect sometimes. Do not panic due to the Ghosting effect. It is caused due to delay in code execution.')

    st.sidebar.subheader(":blue[Note]  :green[ from]  :red[ Developer]:")
    st.sidebar.write('The Creatus model creator is slightly more efficient than the teachable machine model creator as Creatus provides more customizability. But, for beginners, teachable machine might be a more comfortable option due to its simplicity and user friendly interface. But for advanced developers, Creatus will be more preferred choice.')

# Define a function to train the model with progress
def train_model(images, labels, num_classes, epochs, progress_bar, **kwargs):
    X = np.array(images)
    y = np.array(labels)

    # Normalize the pixel values to be between 0 and 1
    X = X / 255.0

    # One-hot encode the labels
    y = to_categorical(y, num_classes)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the CNN model
    model_architecture = kwargs.get('model_architecture', 'Simple CNN')
    if model_architecture == "Simple CNN":
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    elif model_architecture == "VGG-like":
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    elif model_architecture == "ResNet-like":
        def residual_block(x, filters, kernel_size=3, stride=1):
            y = Conv2D(filters, kernel_size, padding='same', strides=stride)(x)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.Activation('relu')(y)
            y = Conv2D(filters, kernel_size, padding='same')(y)
            y = tf.keras.layers.BatchNormalization()(y)
            if stride != 1 or x.shape[-1] != filters:
                x = Conv2D(filters, 1, strides=stride, padding='same')(x)
            return tf.keras.layers.add([x, y])

        inputs = tf.keras.Input(shape=(64, 64, 3))
        x = Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = MaxPooling2D(3, strides=2, padding='same')(x)
        x = residual_block(x, 64)
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 256, stride=2)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs, outputs)
    elif model_architecture == "Custom":
        num_conv_layers = kwargs.get('num_conv_layers', 3)
        num_dense_layers = kwargs.get('num_dense_layers', 2)
        activation_function = kwargs.get('activation_function', 'relu')
        
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation=activation_function, input_shape=(64, 64, 3)))
        model.add(MaxPooling2D((2, 2)))
        
        for i in range(num_conv_layers - 1):
            model.add(Conv2D(64 * (2**i), (3, 3), activation=activation_function))
            model.add(MaxPooling2D((2, 2)))
        
        model.add(Flatten())
        
        for _ in range(num_dense_layers - 1):
            model.add(Dense(128, activation=activation_function))
        
        model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer_name = kwargs.get('optimizer', 'Adam')
    learning_rate = kwargs.get('learning_rate', 0.001)
    if optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation
    if kwargs.get('data_augmentation', False):
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=kwargs.get('rotation_range', 20),
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=kwargs.get('zoom_range', 0.2),
            horizontal_flip=kwargs.get('horizontal_flip', True),
            vertical_flip=kwargs.get('vertical_flip', False),
            fill_mode='nearest'
        )
        data_gen.fit(X_train)
        train_generator = data_gen.flow(X_train, y_train, batch_size=kwargs.get('batch_size', 32))
        steps_per_epoch = len(X_train) // kwargs.get('batch_size', 32)
        
        # Train the model with progress reporting
        for epoch in range(epochs):
            model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=1, validation_data=(X_test, y_test))
            progress_bar.progress((epoch + 1) / epochs)  # Update the progress bar
    else:
        # Train the model with progress reporting
        for epoch in range(epochs):
            model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size=kwargs.get('batch_size', 32))
            progress_bar.progress((epoch + 1) / epochs)  # Update the progress bar

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    metrics = {
        'accuracy': accuracy_score(y_true_classes, y_pred_classes),
        'precision': precision_score(y_true_classes, y_pred_classes, average='weighted'),
        'recall': recall_score(y_true_classes, y_pred_classes, average='weighted'),
        'f1_score': f1_score(y_true_classes, y_pred_classes, average='weighted')
    }

    return model, metrics

# Function to save the model in the specified format
def save_model(model, export_format, label_mapping):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        if export_format == 'tflite':
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            zf.writestr("model.tflite", tflite_model)
            
            usage_code = """
import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# Load the model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Prepare the image
image_path = input("Enter the path to your image: ")
img = preprocess_image(image_path)

# Test the model
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
predicted_label_index = np.argmax(output)
print(f"Predicted Label: {labels[predicted_label_index]}")
print(f"Confidence: {np.max(output):.2f}")
"""
        elif export_format == 'h5':
            model.save("model.h5")
            zf.write("model.h5")
            
            usage_code = """
import tensorflow as tf
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Prepare the image
image_path = input("Enter the path to your image: ")
img = preprocess_image(image_path)

# Test the model
prediction = model.predict(img)
predicted_label_index = np.argmax(prediction)
print(f"Predicted Label: {labels[predicted_label_index]}")
print(f"Confidence: {np.max(prediction):.2f}")
"""

        zf.writestr("main.py", usage_code)
        
        # Create and add labels.txt file
        labels_content = "\n".join(label_mapping.keys())
        zf.writestr("labels.txt", labels_content)

    buffer.seek(0)
    return buffer

# Function to test the model with a new image
def test_model(model, img_array, label_mapping):
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    prediction = model.predict(img_array)
    predicted_label_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Reverse mapping from index to label
    labels_reverse_map = {v: k for k, v in label_mapping.items()}

    predicted_label = labels_reverse_map[predicted_label_index]
    return predicted_label, confidence

# Main app logic
if st.session_state['initial_load']:
    splash = st.empty()
    splash.markdown(create_splash_html("Creatus", '#48CFCB'), unsafe_allow_html=True)
    time.sleep(4)
    splash.empty()
    st.session_state['initial_load'] = False
    main_content()
elif st.session_state['dev_command_entered']:
    dev_splash = st.empty()
    dev_splash.markdown(create_splash_html("Welcome, Pranav Lejith {Amphibiar] (Developer)... ", 'red'), unsafe_allow_html=True)
    time.sleep(4)
    dev_splash.empty()
    st.session_state['dev_command_entered'] = False
    main_content()
else:
    main_content()

# Add custom CSS to position the chat button
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div.stButton > button[kind="secondary"] {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999;
        border-radius: 20px;
        padding: 10px 20px;
        background-color: #4CAF50;
        color: white;
        border: none;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition-duration: 0.4s;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)