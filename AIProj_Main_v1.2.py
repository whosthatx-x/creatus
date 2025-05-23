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
import matplotlib.pyplot as plt

st.set_page_config(page_title="TechnoKreate", page_icon='logo.png', menu_items={
    'About': "# :red[Creator]:blue[:] :violet[Shreeya Prusty(:green[Developer])]"
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
if 'dev_command_entered' not in st.session_state:
    st.session_state['dev_command_entered'] = False

# Developer authentication (hidden from normal users)
developer_commands = [
    'override protocol-amphibiar', 'override command-amphibiar', 
    'command override-amphibiar', 'command override-amphibiar23', 
    'control override-amphibiar', 'system override-amphibiar', 'user:amphibiar',
    'user:amphibiar-developer', 'user:amphibiar-admin', 'user:amphibiar-root',
    'control-admin', 'control-amphibiar','initiate override-amphibiar','currentuser:amphibiar',
    'initiate control override-amphibiar', 'initiate control-amphibiar','switch control-amphibiar'
    'override emergency protocol-amphibiar', 'command override-shreeya', 'protocol 23'
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
      animation: typing 2s steps(30, end), blink-caret .5s step-end infinite;
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
    st.title(":red[TechnoKreate (Model Creator)]")

    # Sidebar for label input
    st.sidebar.title(":blue[Manage Labels]")

    label_input = st.sidebar.text_input(":red[Enter a new label:]")
    if st.sidebar.button(":violet[Add Label]"):
        if label_input in developer_commands:
            st.session_state['dev_command_entered'] = True
            st.session_state['is_developer'] = True
            st.session_state['show_developer_splash'] = True
        elif label_input and label_input not in st.session_state['labels']:
            st.session_state['labels'][label_input] = []
            st.session_state['num_classes'] += 1
            st.sidebar.success(f"Label '{label_input}' added!")
        else:
            st.sidebar.warning("Label already exists or is empty.")

    # Display the existing labels and allow image upload in rows
    if st.session_state['num_classes'] > 0:
        num_columns = 3
        cols = st.columns(num_columns)

        for i, label in enumerate(st.session_state['labels']):
            with cols[i % num_columns]:
                st.subheader(f"Upload images for label: {label}")
                uploaded_files = st.file_uploader(f"Upload images for {label}", accept_multiple_files=True, type=['jpg', 'jpeg', 'png', 'webp'], key=label)

                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        image_data = image.load_img(uploaded_file, target_size=(64, 64))
                        image_array = image.img_to_array(image_data)
                        st.session_state['labels'][label].append(image_array)
                    st.success(f"Uploaded {len(uploaded_files)} images for label '{label}'.")

    # Display labels with delete buttons
    st.sidebar.subheader(":orange[Existing Labels]")
    for label in list(st.session_state['labels'].keys()):
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.write(label)
        if col2.button(":red[-]", key=f"delete_{label}"):
            del st.session_state['labels'][label]
            st.session_state['num_classes'] -= 1

    # Dropdown to select model export format
    export_format = st.sidebar.selectbox(":blue[Select model export format:]", options=["tflite", "h5"])

    # Advanced options in sidebar
    with st.sidebar.expander("Advanced Options", expanded=st.session_state['is_developer']):
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)

        model_architecture = "Simple CNN"

        if st.session_state['is_developer']:
            st.subheader("Developer Options")

            # Theme customization
            theme = st.selectbox("Theme", ["Light", "Dark", "Custom"])
            if theme == "Custom":
                primary_color = st.color_picker("Primary Color", "#FF4B4B")
                secondary_color = st.color_picker("Secondary Color", "#0068C9")
                background_color = st.color_picker("Background Color", "#FFFFFF")
                text_color = st.color_picker("Text Color", "#262730")

                # Apply custom theme
                st.markdown(f"""
                    <style>
                    :root {{
                        --primary-color: {primary_color};
                        --secondary-color: {secondary_color};
                        --background-color: {background_color};
                        --text-color: {text_color};
                    }}
                    body {{
                        color: var(--text-color);
                        background-color: var(--background-color);
                    }}
                    .stButton > button {{
                        color: var(--background-color);
                        background-color: var(--primary-color);
                    }}
                    .stTextInput > div > div > input {{
                        color: var(--text-color);
                    }}
                    </style>
                """, unsafe_allow_html=True)

            # Model architecture options
            model_architecture = st.selectbox("Model Architecture", ["Simple CNN", "VGG-like", "ResNet-like", "Custom"])
            if model_architecture == "Custom":
                num_conv_layers = st.number_input("Number of Convolutional Layers", min_value=1, max_value=10, value=3)
                num_dense_layers = st.number_input("Number of Dense Layers", min_value=1, max_value=5, value=2)
                activation_function = st.selectbox("Activation Function", ["relu", "leaky_relu", "elu", "selu"])

            # Data augmentation options
            data_augmentation = st.checkbox("Enable Data Augmentation")
            if data_augmentation:
                rotation_range = st.slider("Rotation Range (Degrees)", 0, 180, 20)
                zoom_range = st.slider("Zoom Range", 0.0, 1.0, 0.2)
                horizontal_flip = st.checkbox("Horizontal Flip", value=True)
                vertical_flip = st.checkbox("Vertical Flip")

            # Training options
            early_stopping = st.checkbox("Enable Early Stopping")
            if early_stopping:
                patience = st.number_input("Early Stopping Patience", min_value=1, max_value=20, value=5)

            # Optimization options
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
            if optimizer == "SGD":
                momentum = st.slider("Momentum", 0.0, 1.0, 0.9)

            # Regularization options
            l2_regularization = st.checkbox("L2 Regularization")
            if l2_regularization:
                l2_lambda = st.number_input("L2 Lambda", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")

            dropout = st.checkbox("Dropout")
            if dropout:
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)

            # Advanced visualization options
            show_model_summary = st.checkbox("Show Model Summary")
            plot_training_history = st.checkbox("Plot Training History")

            # Export options
            export_tensorboard_logs = st.checkbox("Export TensorBoard Logs")

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
                progress_bar = st.progress(0)

                training_options = {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "model_architecture": model_architecture,
                    "data_augmentation": st.session_state['is_developer'] and data_augmentation,
                    "early_stopping": st.session_state['is_developer'] and early_stopping,
                }

                if st.session_state['is_developer']:
                    if model_architecture == "Custom":
                        training_options.update({
                            "num_conv_layers": num_conv_layers,
                            "num_dense_layers": num_dense_layers,
                            "activation_function": activation_function,
                        })

                    if data_augmentation:
                        training_options.update({
                            "rotation_range": rotation_range,
                            "zoom_range": zoom_range,
                            "horizontal_flip": horizontal_flip,
                            "vertical_flip": vertical_flip,
                        })

                    if early_stopping:
                        training_options["patience"] = patience

                    training_options["optimizer"] = optimizer
                    if optimizer == "SGD":
                        training_options["momentum"] = momentum

                    if l2_regularization:
                        training_options["l2_lambda"] = l2_lambda

                    if dropout:
                        training_options["dropout_rate"] = dropout_rate

                st.session_state['model'] = train_model(all_images, all_labels, st.session_state['num_classes'], epochs, progress_bar, **training_options)

                if st.session_state['is_developer']:
                    if show_model_summary:
                        st.subheader("Model Summary")
                        st.text(st.session_state['model'].summary())

                    if plot_training_history and hasattr(st.session_state['model'], 'history'):
                        st.subheader("Training History")
                        col1, col2 = st.columns([1, 3])
                        with col2:
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
                            ax1.plot(st.session_state['model'].history.history['accuracy'])
                            ax1.plot(st.session_state['model'].history.history['val_accuracy'])
                            ax1.set_title('Model Accuracy')
                            ax1.set_ylabel('Accuracy')
                            ax1.set_xlabel('Epoch')
                            ax1.legend(['Train', 'Validation'], loc='lower right')

                            ax2.plot(st.session_state['model'].history.history['loss'])
                            ax2.plot(st.session_state['model'].history.history['val_loss'])
                            ax2.set_title('Model Loss')
                            ax2.set_ylabel('Loss')
                            ax2.set_xlabel('Epoch')
                            ax2.legend(['Train', 'Validation'], loc='upper right')

                            plt.tight_layout()
                            st.pyplot(fig)

                    if export_tensorboard_logs:
                        pass

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

    st.sidebar.write("This app is created by :red[**Shreeya Prusty**](:violet[**Developer**])")
    st.sidebar.write(":green[Beginners are advised not to change any of the advanced options as it affects the model training process. Any doubts or errors or any suggestions to improve the app further may be discussed with the Developer.]")

    st.sidebar.header(":orange[**Usage Instructions**]")
    st.sidebar.write("""
    ### :red[Step 1: Add Labels]
    1. In the sidebar, enter the name of a label in the "Enter a new label" input field.
    2. Click the "Add Label" button to add the label.

    ### :red[Step 2: Upload Images]
    1. For each label, you will see a section to upload images.
    2. Click the "Upload images for [label]" button to open the file uploader.
    3. Select the images corresponding to the label and upload them.
    4. Repeat this process for all labels.

    ### :red[Step 3: Train the Model]
    1. Once you have uploaded images for at least two labels, click the "Train Model" button.
    2. The model will start training, and you can see the progress in real-time.
    3. After the training is complete, you will receive a success message.

    ### :red[Step 4: Test the Model]
    1. Upload an image to test the trained model by clicking the "Upload an image to test" button.
    2. The model will predict the label of the uploaded image and display the confidence score.

    ### :red[Step 5: Download the Model]
    1. After training, choose your desired export format (TensorFlow Lite or H5) from the sidebar.
    2. Click the "Download Model" button to download the model along with the usage code.
    """, unsafe_allow_html=True)
    
    st.sidebar.header(":blue[Note]  :green[ from]  :red[ Developer]")
    st.sidebar.write('The TechnoKreate model creator is slightly more efficient than the teachable machine model creator as TechnoKreate provides more customizability. But, for beginners, teachable machine might be a more comfortable option due to its simplicity and user friendly interface. But for advanced developers, TechnoKreate will be more preferred choice.')
    st.sidebar.header(':blue[Definitions]  ')
    st.sidebar.write("""
    
    **:orange[Epochs:]**
    An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model. Another way to define an epoch is the number of passes a training dataset takes around an algorithm.

    """)
    st.sidebar.subheader(":red[**Warning**]")
    st.sidebar.write('The code might produce a ghosting effect sometimes. Do not panic due to the Ghosting effect. It is caused due to delay in code execution and is compeletely normal.')
    if st.session_state['is_developer']:
        if st.sidebar.button("Reset to Normal User", key="reset_button"):
            st.session_state['is_developer'] = False
            st.session_state['dev_command_entered'] = False
            # TODO: Add a toast
            # st.toast("Exiting Developer Mode")
            # BUG: TOAST NOT WORKING


# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# import zipfile
# from io import BytesIO
# import time
# import matplotlib.pyplot as plt

# # Set page config
# st.set_page_config(page_title="Creatus", page_icon='logo.png', menu_items={
#     'About': "# :red[Creator]:blue[:] :violet[Pranav Lejith(:green[Amphibiar])]"
# }, layout='wide')

# # Initialize session state keys
# if 'labels' not in st.session_state:
#     st.session_state['labels'] = {}
# if 'num_classes' not in st.session_state:
#     st.session_state['num_classes'] = 0
# if 'label_mapping' not in st.session_state:
#     st.session_state['label_mapping'] = {}
# if 'model' not in st.session_state:
#     st.session_state['model'] = None
# if 'is_developer' not in st.session_state:
#     st.session_state['is_developer'] = False
# if 'show_developer_splash' not in st.session_state:
#     st.session_state['show_developer_splash'] = False
# if 'initial_load' not in st.session_state:
#     st.session_state['initial_load'] = True
# if 'dev_command_entered' not in st.session_state:
#     st.session_state['dev_command_entered'] = False

# # Developer authentication (hidden from normal users)
# developer_commands = [
#     'override protocol-amphibiar', 'override command-amphibiar', 
#     'command override-amphibiar', 'command override-amphibiar23', 
#     'control override-amphibiar', 'system override-amphibiar', 'user:amphibiar',
#     'user:amphibiar-developer', 'user:amphibiar-admin', 'user:amphibiar-root',
#     'control-admin', 'control-amphibiar','initiate override-amphibiar','currentuser:amphibiar',
#     'initiate control override-amphibiar', 'initiate control-amphibiar','switch control-amphibiar'
#     'override emergency protocol-amphibiar'
# ]

# # Custom HTML for splash screen with typewriter effect
# def create_splash_html(text, color):
#     return f"""
#     <style>
#     .typewriter h1 {{
#       overflow: hidden;
#       color: {color};
#       white-space: nowrap;
#       margin: 0 auto;
#       letter-spacing: .15em;
#       border-right: .15em solid orange;
#       animation: typing 2s steps(30, end), blink-caret .5s step-end infinite;
#     }}

#     @keyframes typing {{
#       from {{ width: 0 }}
#       to {{ width: 100% }}
#     }}

#     @keyframes blink-caret {{
#       from, to {{ border-color: transparent }}
#       50% {{ border-color: orange }}
#     }}
#     </style>
#     <div class="typewriter">
#         <h1>{text}</h1>
#     </div>
#     """

# # Main content
# def main_content():
#     st.title(":red[Creatus (Model Creator)]")

#     # Sidebar for label input
#     st.sidebar.title(":blue[Manage Labels]")

#     label_input = st.sidebar.text_input("Enter a new label:")
#     if st.sidebar.button("Add Label"):
#         if label_input in developer_commands:
#             st.session_state['dev_command_entered'] = True
#             st.session_state['is_developer'] = True
#             st.session_state['show_developer_splash'] = True
#         elif label_input and label_input not in st.session_state['labels']:
#             st.session_state['labels'][label_input] = []
#             st.session_state['num_classes'] += 1
#             st.sidebar.success(f"Label '{label_input}' added!")
#         else:
#             st.sidebar.warning("Label already exists or is empty.")

#     # Display the existing labels and allow image upload in rows
#     if st.session_state['num_classes'] > 0:
#         num_columns = 3
#         cols = st.columns(num_columns)

#         for i, label in enumerate(st.session_state['labels']):
#             with cols[i % num_columns]:
#                 st.subheader(f"Upload images for label: {label}")
#                 uploaded_files = st.file_uploader(f"Upload images for {label}", accept_multiple_files=True, type=['jpg', 'jpeg', 'png', 'webp'], key=label)

#                 if uploaded_files:
#                     for uploaded_file in uploaded_files:
#                         image_data = image.load_img(uploaded_file, target_size=(64, 64))
#                         image_array = image.img_to_array(image_data)
#                         st.session_state['labels'][label].append(image_array)
#                     st.success(f"Uploaded {len(uploaded_files)} images for label '{label}'.")

#     # Display labels with delete buttons
#     st.sidebar.subheader("Existing Labels")
#     for label in list(st.session_state['labels'].keys()):
#         col1, col2 = st.sidebar.columns([0.8, 0.2])
#         col1.write(label)
#         if col2.button(":red[-]", key=f"delete_{label}"):
#             del st.session_state['labels'][label]
#             st.session_state['num_classes'] -= 1

#     # Dropdown to select model export format
#     export_format = st.sidebar.selectbox("Select model export format:", options=["tflite", "h5"])

#     # Advanced options in sidebar
#     with st.sidebar.expander("Advanced Options", expanded=st.session_state['is_developer']):
#         epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10)
#         learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
#         batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)

#         model_architecture = "Simple CNN"

#         if st.session_state['is_developer']:
#             st.subheader("Developer Options")

#             # Theme customization
#             theme = st.selectbox("Theme", ["Light", "Dark", "Custom"])
#             if theme == "Custom":
#                 primary_color = st.color_picker("Primary Color", "#FF4B4B")
#                 secondary_color = st.color_picker("Secondary Color", "#0068C9")
#                 background_color = st.color_picker("Background Color", "#FFFFFF")
#                 text_color = st.color_picker("Text Color", "#262730")

#                 # Apply custom theme
#                 st.markdown(f"""
#                     <style>
#                     :root {{
#                         --primary-color: {primary_color};
#                         --secondary-color: {secondary_color};
#                         --background-color: {background_color};
#                         --text-color: {text_color};
#                     }}
#                     body {{
#                         color: var(--text-color);
#                         background-color: var(--background-color);
#                     }}
#                     .stButton > button {{
#                         color: var(--background-color);
#                         background-color: var(--primary-color);
#                     }}
#                     .stTextInput > div > div > input {{
#                         color: var(--text-color);
#                     }}
#                     </style>
#                 """, unsafe_allow_html=True)

#             # Model architecture options
#             model_architecture = st.selectbox("Model Architecture", ["Simple CNN", "VGG-like", "ResNet-like", "Custom"])
#             if model_architecture == "Custom":
#                 num_conv_layers = st.number_input("Number of Convolutional Layers", min_value=1, max_value=10, value=3)
#                 num_dense_layers = st.number_input("Number of Dense Layers", min_value=1, max_value=5, value=2)
#                 activation_function = st.selectbox("Activation Function", ["relu", "leaky_relu", "elu", "selu"])

#             # Data augmentation options
#             data_augmentation = st.checkbox("Enable Data Augmentation")
#             if data_augmentation:
#                 rotation_range = st.slider("Rotation Range (Degrees)", 0, 180, 20)
#                 zoom_range = st.slider("Zoom Range", 0.0, 1.0, 0.2)
#                 horizontal_flip = st.checkbox("Horizontal Flip", value=True)
#                 vertical_flip = st.checkbox("Vertical Flip")

#             # Training options
#             early_stopping = st.checkbox("Enable Early Stopping")
#             if early_stopping:
#                 patience = st.number_input("Early Stopping Patience", min_value=1, max_value=20, value=5)

#             # Optimization options
#             optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
#             if optimizer == "SGD":
#                 momentum = st.slider("Momentum", 0.0, 1.0, 0.9)

#             # Regularization options
#             l2_regularization = st.checkbox("L2 Regularization")
#             if l2_regularization:
#                 l2_lambda = st.number_input("L2 Lambda", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")

#             dropout = st.checkbox("Dropout")
#             if dropout:
#                 dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)

#             # Advanced visualization options
#             show_model_summary = st.checkbox("Show Model Summary")
#             plot_training_history = st.checkbox("Plot Training History")

#             # Export options
#             export_tensorboard_logs = st.checkbox("Export TensorBoard Logs")

#     # Button to train the model
#     if st.session_state['num_classes'] > 1:
#         if st.button("Train Model"):
#             all_images = []
#             all_labels = []
#             st.session_state['label_mapping'] = {label: idx for idx, label in enumerate(st.session_state['labels'].keys())}

#             for label, images in st.session_state['labels'].items():
#                 all_images.extend(images)
#                 all_labels.extend([st.session_state['label_mapping'][label]] * len(images))

#             if len(all_images) > 0:
#                 st.write("Training the model...")
#                 progress_bar = st.progress(0)

#                 training_options = {
#                     "learning_rate": learning_rate,
#                     "batch_size": batch_size,
#                     "model_architecture": model_architecture,
#                     "data_augmentation": st.session_state['is_developer'] and data_augmentation,
#                     "early_stopping": st.session_state['is_developer'] and early_stopping,
#                 }

#                 if st.session_state['is_developer']:
#                     if model_architecture == "Custom":
#                         training_options.update({
#                             "num_conv_layers": num_conv_layers,
#                             "num_dense_layers": num_dense_layers,
#                             "activation_function": activation_function,
#                         })

#                     if data_augmentation:
#                         training_options.update({
#                             "rotation_range": rotation_range,
#                             "zoom_range": zoom_range,
#                             "horizontal_flip": horizontal_flip,
#                             "vertical_flip": vertical_flip,
#                         })

#                     if early_stopping:
#                         training_options["patience"] = patience

#                     training_options["optimizer"] = optimizer
#                     if optimizer == "SGD":
#                         training_options["momentum"] = momentum

#                     if l2_regularization:
#                         training_options["l2_lambda"] = l2_lambda

#                     if dropout:
#                         training_options["dropout_rate"] = dropout_rate

#                 st.session_state['model'] = train_model(all_images, all_labels, st.session_state['num_classes'], epochs, progress_bar, **training_options)

#                 if st.session_state['is_developer']:
#                     if show_model_summary:
#                         st.subheader("Model Summary")
#                         st.text(st.session_state['model'].summary())

#                     if plot_training_history and hasattr(st.session_state['model'], 'history'):
#                         st.subheader("Training History")
#                         col1, col2 = st.columns([1, 3])
#                         with col2:
#                             fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
#                             ax1.plot(st.session_state['model'].history.history['accuracy'])
#                             ax1.plot(st.session_state['model'].history.history['val_accuracy'])
#                             ax1.set_title('Model Accuracy')
#                             ax1.set_ylabel('Accuracy')
#                             ax1.set_xlabel('Epoch')
#                             ax1.legend(['Train', 'Validation'], loc='lower right')

#                             ax2.plot(st.session_state['model'].history.history['loss'])
#                             ax2.plot(st.session_state['model'].history.history['val_loss'])
#                             ax2.set_title('Model Loss')
#                             ax2.set_ylabel('Loss')
#                             ax2.set_xlabel('Epoch')
#                             ax2.legend(['Train', 'Validation'], loc='upper right')

#                             plt.tight_layout()
#                             st.pyplot(fig)

#                     if export_tensorboard_logs:
#                         pass

#                 st.toast('Model Trained Successfully')
#                 st.success("Model trained!")
#             else:
#                 st.error("Please upload some images before training.")
#     else:
#         st.warning("At least two labels are required to train the model.")

#     # Option to test the trained model
#     if st.session_state['model'] is not None:
#         st.subheader("Test the trained model with a new image")
#         test_image = st.file_uploader("Upload an image to test", type=['jpg', 'jpeg', 'png','webp'], key="test")

#         if test_image:
#             test_image_data = image.load_img(test_image, target_size=(64, 64))
#             st.image(test_image_data, caption="Uploaded Image", use_column_width=True)

#             test_image_array = image.img_to_array(test_image_data)
#             predicted_label, confidence = test_model(st.session_state['model'], test_image_array, st.session_state['label_mapping'])

#             st.write(f"Predicted Label: {predicted_label}")
#             st.slider("Confidence Level (%)", min_value=1, max_value=100, value=int(confidence * 100), disabled=True)

#     # Button to download the model
#     if st.session_state['model'] is not None and st.button("Download Model"):
#         try:
#             buffer = save_model(st.session_state['model'], export_format, st.session_state['label_mapping'])

#             st.download_button(
#                 label="Download the trained model and usage code",
#                 data=buffer,
#                 file_name=f"trained_model_{export_format}.zip",
#                 mime="application/zip"
#             )
#         except Exception as e:
#             st.error(f"Error: {e}")

#     st.sidebar.write("This app was created by :red[**Pranav Lejith**](:violet[**Amphibiar**])")
#     st.sidebar.write(":green[Beginners are advised not to change any of the advanced options as it affects the model training process. Any doubts or errors or any suggestions to improve the app further can be discussed with the Developer.]")

#     st.sidebar.subheader(":orange[**Usage Instructions**]")
#     st.sidebar.write("""
#     ### Step 1: Add Labels
#     1. In the sidebar, enter the name of a label in the "Enter a new label" input field.
#     2. Click the "Add Label" button to add the label.

#     ### Step 2: Upload Images
#     1. For each label, you will see a section to upload images.
#     2. Click the "Upload images for [label]" button to open the file uploader.
#     3. Select the images corresponding to the label and upload them.
#     4. Repeat this process for all labels.

#     ### Step 3: Train the Model
#     1. Once you have uploaded images for at least two labels, click the "Train Model" button.
#     2. The model will start training, and you can see the progress in real-time.
#     3. After the training is complete, you will receive a success message.

#     ### Step 4: Test the Model
#     1. Upload an image to test the trained model by clicking the "Upload an image to test" button.
#     2. The model will predict the label of the uploaded image and display the confidence score.

#     ### Step 5: Download the Model
#     1. After training, choose your desired export format (TensorFlow Lite or H5) from the sidebar.
#     2. Click the "Download Model" button to download the model along with the usage code.
#     """, unsafe_allow_html=True)
    
#     st.sidebar.subheader(":blue[Note]  :green[ from]  :red[ Developer]")
#     st.sidebar.write('The Creatus model creator is slightly more efficient than the teachable machine model creator as Creatus provides more customizability. But, for beginners, teachable machine might be a more comfortable option due to its simplicity and user friendly interface. But for advanced developers, Creatus will be more preferred choice.')
#     st.sidebar.subheader(':blue[Definitions]  ')
#     st.sidebar.write("""
    
#     **:orange[Epochs:]**
#     An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model. Another way to define an epoch is the number of passes a training dataset takes around an algorithm.

#     """)
#     st.sidebar.subheader(":red[**Warning**]")
#     st.sidebar.write('The code might produce a ghosting effect sometimes. Do not panic due to the Ghosting effect. It is caused due to delay in code execution.')
#     if st.session_state['is_developer']:
#         if st.sidebar.button("Reset to Normal User", key="reset_button"):
#             st.session_state['is_developer'] = False
#             st.session_state['dev_command_entered'] = False

# def train_model(images, labels, num_classes, epochs, progress_bar, **kwargs):
#     X = np.array(images)
#     y = np.array(labels)

#     X = X / 255.0

#     y = to_categorical(y, num_classes)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     model_architecture = kwargs.get('model_architecture', 'Simple CNN')
#     if model_architecture == "Simple CNN":
#         model = Sequential([
#             Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#             MaxPooling2D((2, 2)),
#             Flatten(),
#             Dense(128, activation='relu'),
#             Dense(num_classes, activation='softmax')
#         ])
#     elif model_architecture == "VGG-like":
#         model = Sequential([
#             Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
#             Conv2D(64, (3, 3), activation='relu', padding='same'),
#             MaxPooling2D((2, 2)),
#             Conv2D(128, (3, 3), activation='relu', padding='same'),
#             Conv2D(128, (3, 3), activation='relu', padding='same'),
#             MaxPooling2D((2, 2)),
#             Flatten(),
#             Dense(256, activation='relu'),
#             Dense(num_classes, activation='softmax')
#         ])
#     elif model_architecture == "ResNet-like":
#         def residual_block(x, filters, kernel_size=3, stride=1):
#             y = Conv2D(filters, kernel_size, padding='same', strides=stride)(x)
#             y = tf.keras.layers.BatchNormalization()(y)
#             y = tf.keras.layers.Activation('relu')(y)
#             y = Conv2D(filters, kernel_size, padding='same')(y)
#             y = tf.keras.layers.BatchNormalization()(y)
#             if stride != 1 or x.shape[-1] != filters:
#                 x = Conv2D(filters, 1, strides=stride, padding='same')(x)
#             return tf.keras.layers.add([x, y])

#         inputs = tf.keras.Input(shape=(64, 64, 3))
#         x = Conv2D(64, 7, strides=2, padding='same')(inputs)
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Activation('relu')(x)
#         x = MaxPooling2D(3, strides=2, padding='same')(x)
#         x = residual_block(x, 64)
#         x = residual_block(x, 128, stride=2)
#         x = residual_block(x, 256, stride=2)
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)
#         outputs = Dense(num_classes, activation='softmax')(x)
#         model = tf.keras.Model(inputs, outputs)
#     elif model_architecture == "Custom":
#         num_conv_layers = kwargs.get('num_conv_layers', 3)
#         num_dense_layers = kwargs.get('num_dense_layers', 2)
#         activation_function = kwargs.get('activation_function', 'relu')

#         model = Sequential()
#         model.add(Conv2D(32, (3, 3), activation=activation_function, padding='same', input_shape=(64, 64, 3)))
#         model.add(MaxPooling2D((2, 2)))

#         for i in range(num_conv_layers - 1):
#             model.add(Conv2D(64 * (2**i), (3, 3), activation=activation_function, padding='same'))
#             if i < num_conv_layers - 2:
#                 model.add(MaxPooling2D((2, 2)))

#         model.add(Flatten())

#         for _ in range(num_dense_layers - 1):
#             model.add(Dense(128, activation=activation_function))

#         model.add(Dense(num_classes, activation='softmax'))

#     optimizer = kwargs.get('optimizer', 'Adam')
#     learning_rate = kwargs.get('learning_rate', 0.001)

#     if optimizer == 'Adam':
#         opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     elif optimizer == 'SGD':
#         momentum = kwargs.get('momentum', 0.9)
#         opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
#     elif optimizer == 'RMSprop':
#         opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#     if kwargs.get('data_augmentation', False):
#         data_augmentation = tf.keras.Sequential([
#             tf.keras.layers.RandomFlip("horizontal"),
#             tf.keras.layers.RandomRotation(kwargs.get('rotation_range', 20) / 180.0),
#             tf.keras.layers.RandomZoom(kwargs.get('zoom_range', 0.2)),
#         ])
#         if kwargs.get('vertical_flip', False):
#             data_augmentation.add(tf.keras.layers.RandomFlip("vertical"))

#         X_train = data_augmentation(X_train)

#     callbacks = []
#     if kwargs.get('early_stopping', False):
#         early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=kwargs.get('patience', 5))
#         callbacks.append(early_stop)

#     class ProgressBarCallback(tf.keras.callbacks.Callback):
#         def __init__(self, epochs, progress_bar):
#             self.epochs = epochs
#             self.progress_bar = progress_bar

#         def on_epoch_end(self, epoch, logs=None):
#             current_progress = (epoch + 1) / self.epochs
#             self.progress_bar.progress(current_progress)

#     progress_callback = ProgressBarCallback(epochs, progress_bar)
#     callbacks.append(progress_callback)

#     history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
#                         batch_size=kwargs.get('batch_size', 32), callbacks=callbacks)

#     model.history = history
#     return model

# def save_model(model, export_format, label_mapping):
#     buffer = BytesIO()
#     with zipfile.ZipFile(buffer, "w") as zf:
#         if export_format == 'tflite':
#             converter = tf.lite.TFLiteConverter.from_keras_model(model)
#             tflite_model = converter.convert()
#             zf.writestr("model.tflite", tflite_model)
            
#             usage_code = """
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# def preprocess_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize((64, 64))
#     img_array = np.array(img) / 255.0
#     return np.expand_dims(img_array, axis=0).astype(np.float32)

# # Load the model
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Prepare the image
# image_path = input("Enter the path to your image: ")
# img = preprocess_image(image_path)

# # Test the model
# interpreter.set_tensor(input_details[0]['index'], img)
# interpreter.invoke()

# output = interpreter.get_tensor(output_details[0]['index'])
# predicted_label = np.argmax(output)
# label_mapping = {label_mapping[label]: label for label in label_mapping}
# print(f"Predicted Label: {label_mapping[predicted_label]}")
# print(f"Confidence: {np.max(output):.2f}")
# """
#         elif export_format == 'h5':
#             model.save("model.h5")
#             zf.write("model.h5")
            
#             usage_code = """
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# def preprocess_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     img = img.resize((64, 64))
#     img_array = np.array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

# # Load the model
# model = tf.keras.models.load_model('model.h5')

# # Prepare the image
# image_path = input("Enter the path to your image: ")
# img = preprocess_image(image_path)

# # Test the model
# prediction = model.predict(img)
# predicted_label = np.argmax(prediction)
# label_mapping = {label_mapping[label]: label for label in label_mapping}
# print(f"Predicted Label: {label_mapping[predicted_label]}")
# print(f"Confidence: {np.max(prediction):.2f}")
# """

#         zf.writestr("main.py", usage_code)
#         zf.writestr("label_mapping.py", f"label_mapping = {label_mapping}")

#     buffer.seek(0)
#     return buffer

# def test_model(model, img_array, label_mapping):
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0

#     prediction = model.predict(img_array)
#     predicted_label_index = np.argmax(prediction)
#     confidence = np.max(prediction)

#     labels_reverse_map = {v: k for k, v in label_mapping.items()}

#     predicted_label = labels_reverse_map[predicted_label_index]
#     return predicted_label, confidence

# # Main app logic
# if st.session_state['initial_load']:
#     splash = st.empty()
#     splash.markdown(create_splash_html("Creatus", '#48CFCB'), unsafe_allow_html=True)
#     time.sleep(4)
#     splash.empty()
#     st.session_state['initial_load'] = False
#     main_content()
# elif st.session_state['dev_command_entered']:
#     dev_splash = st.empty()
#     dev_splash.markdown(create_splash_html("Welcome,Amphibiar(Developer)... ", 'red'), unsafe_allow_html=True)
#     time.sleep(4)
#     dev_splash.empty()
#     st.session_state['dev_command_entered'] = False
#     main_content()
# else:
#     main_content()



def train_model(images, labels, num_classes, epochs, progress_bar, **kwargs):
    X = np.array(images)
    y = np.array(labels)

    X = X / 255.0

    y = to_categorical(y, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        model.add(Conv2D(32, (3, 3), activation=activation_function, padding='same', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D((2, 2)))

        for i in range(num_conv_layers - 1):
            model.add(Conv2D(64 * (2**i), (3, 3), activation=activation_function, padding='same'))
            if i < num_conv_layers - 2:
                model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        for _ in range(num_dense_layers - 1):
            model.add(Dense(128, activation=activation_function))

        model.add(Dense(num_classes, activation='softmax'))

    optimizer = kwargs.get('optimizer', 'Adam')
    learning_rate = kwargs.get('learning_rate', 0.001)

    if optimizer == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        momentum = kwargs.get('momentum', 0.9)
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == 'RMSprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if kwargs.get('data_augmentation', False):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(kwargs.get('rotation_range', 20) / 180.0),
            tf.keras.layers.RandomZoom(kwargs.get('zoom_range', 0.2)),
        ])
        if kwargs.get('vertical_flip', False):
            data_augmentation.add(tf.keras.layers.RandomFlip("vertical"))

        X_train = data_augmentation(X_train)

    callbacks = []
    if kwargs.get('early_stopping', False):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=kwargs.get('patience', 5))
        callbacks.append(early_stop)

    class ProgressBarCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs, progress_bar):
            self.epochs = epochs
            self.progress_bar = progress_bar

        def on_epoch_end(self, epoch, logs=None):
            current_progress = (epoch + 1) / self.epochs
            self.progress_bar.progress(current_progress)

    progress_callback = ProgressBarCallback(epochs, progress_bar)
    callbacks.append(progress_callback)

    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
                        batch_size=kwargs.get('batch_size', 32), callbacks=callbacks)

    model.history = history
    return model

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

# Prepare the image
image_path = input("Enter the path to your image: ")
img = preprocess_image(image_path)

# Test the model
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output)
label_mapping = {label_mapping[label]: label for label in label_mapping}
print(f"Predicted Label: {label_mapping[predicted_label]}")
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

# Prepare the image
image_path = input("Enter the path to your image: ")
img = preprocess_image(image_path)

# Test the model
prediction = model.predict(img)
predicted_label = np.argmax(prediction)
label_mapping = {label_mapping[label]: label for label in label_mapping}
print(f"Predicted Label: {label_mapping[predicted_label]}")
print(f"Confidence: {np.max(prediction):.2f}")
"""

        zf.writestr("main.py", usage_code)
        zf.writestr("label_mapping.py", f"label_mapping = {label_mapping}")

    buffer.seek(0)
    return buffer

def test_model(model, img_array, label_mapping):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_label_index = np.argmax(prediction)
    confidence = np.max(prediction)

    labels_reverse_map = {v: k for k, v in label_mapping.items()}

    predicted_label = labels_reverse_map[predicted_label_index]
    return predicted_label, confidence

# Main app logic
if st.session_state['initial_load']:
    splash = st.empty()
    splash.markdown(create_splash_html("TechnoKreate", '#48CFCB'), unsafe_allow_html=True)
    time.sleep(4)
    splash.empty()
    st.session_state['initial_load'] = False
    main_content()
elif st.session_state['dev_command_entered']:
    dev_splash = st.empty()
    dev_splash.markdown(create_splash_html("Welcome,Shreeya(Developer)... ", 'red'), unsafe_allow_html=True)
    time.sleep(4)
    dev_splash.empty()
    st.session_state['dev_command_entered'] = False
    main_content()
else:
    main_content()
