import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import zipfile
from io import BytesIO
import time

# Set page config
st.set_page_config(page_title="Creatus", page_icon='logo.png', menu_items={
    'About': "# :red[Creator]:blue[:] :violet[Pranav Lejith(:green[Amphibiar])]",
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
if 'is_developer' not in st.session_state:
    st.session_state['is_developer'] = False
if 'show_developer_splash' not in st.session_state:
    st.session_state['show_developer_splash'] = False
if 'initial_load' not in st.session_state:
    st.session_state['initial_load'] = True

# Developer authentication (hidden from normal users)
developer_commands = [
    'override protocol-amphibiar', 'override command-amphibiar', 
    'command override-amphibiar', 'command override-amphibiar23', 
    'control override-amphibiar', 'system override-amphibiar', 'user:amphibiar', 'XD'
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

# Main content
def main_content():
    st.title(":red[Creatus (Model Creator)]")

    # Sidebar for label input
    st.sidebar.title(":blue[Manage Labels]")

    # Add reset button for developer mode
    if st.session_state['is_developer']:
        if st.sidebar.button("Reset to Normal User"):
            st.session_state['is_developer'] = False
            st.rerun()

    label_input = st.sidebar.text_input("Enter a new label:")
    if st.sidebar.button("Add Label"):
        if label_input in developer_commands:
            st.session_state['is_developer'] = True
            st.session_state['show_developer_splash'] = True
            st.rerun()
        elif label_input and label_input not in st.session_state['labels']:
            st.session_state['labels'][label_input] = []
            st.session_state['num_classes'] += 1
            st.sidebar.success(f"Label '{label_input}' added!")
        else:
            st.sidebar.warning("Label already exists or is empty.")

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
        else:
            learning_rate = 0.001
            batch_size = 32

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
                st.write("Training the model...")
                progress_bar = st.progress(0)  # Initialize progress bar
                st.session_state['model'] = train_model(all_images, all_labels, st.session_state['num_classes'], epochs, progress_bar, learning_rate, batch_size)
                st.toast('Model Trained Successfully')
                st.success("Model trained!")
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
            predicted_label_code = ', '.join([f"'{label}'" for label in st.session_state['label_mapping']])
            
            if export_format == 'tflite':
                usage_code = f"""
    import tensorflow as tf
    import numpy as np

    # Load the model
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare the image (adjust this for your actual input)
    img = np.random.rand(1, 64, 64, 3).astype(np.float32)

    # Test the model
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output)
    predicted_label_code = [{predicted_label_code}]
    print(f"Predicted Label: {{predicted_label_code[predicted_label]}}")
    """
            elif export_format == 'h5':
                usage_code = f"""
    import tensorflow as tf

    # Load the model
    model = tf.keras.models.load_model('model.h5')

    # Prepare the image (adjust this for your actual input)
    img = np.random.rand(1, 64, 64, 3)

    # Test the model
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    predicted_label_code = [{predicted_label_code}]
    print(f"Predicted Label: {{predicted_label_code[predicted_label]}}")
    """
            
            buffer = save_model(st.session_state['model'], export_format, usage_code)
            
            st.download_button(
                label="Download the trained model and usage code",
                data=buffer,
                file_name=f"trained_model_{export_format}.zip",
                mime="application/zip"
            )
        except Exception as e:
            st.error(f"Error: {e}")

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
def train_model(images, labels, num_classes, epochs, progress_bar, learning_rate, batch_size):
    X = np.array(images)
    y = np.array(labels)

    # Normalize the pixel values to be between 0 and 1
    X = X / 255.0

    # One-hot encode the labels
    y = to_categorical(y, num_classes)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model with progress reporting
    for epoch in range(epochs):
        model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size=batch_size)
        progress_bar.progress((epoch + 1) / epochs)  # Update the progress bar

    return model

# Function to save the model in the specified format
def save_model(model, export_format, usage_code):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        if export_format == 'tflite':
            input_shape = (1, 64, 64, 3)  # Adjust this based on your actual input shape
            run_model = tf.function(lambda x: model(x))
            concrete_func = run_model.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))

            # Convert the model to TensorFlow Lite format
            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            tflite_model = converter.convert()
            zf.writestr("model.tflite", tflite_model)
        elif export_format == 'h5':
            model.save("model.h5")
            zf.write("model.h5")

        # Add the usage code to the zip file
        zf.writestr("main.py", usage_code)

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
    splash.markdown(create_splash_html("Creatus", '#6EC207'), unsafe_allow_html=True)
    time.sleep(1)
    splash.empty()
    st.session_state['initial_load'] = False
    main_content()
elif st.session_state['show_developer_splash']:
    # Clear the entire screen
    st.empty()
    # Show only the developer splash
    dev_splash = st.empty()
    dev_splash.markdown(create_splash_html("Welcome , Amphibiar (Developer)", 'red'), unsafe_allow_html=True)
    time.sleep(4)
    dev_splash.empty()
    st.session_state['show_developer_splash'] = False
    main_content()
else:
    main_content()