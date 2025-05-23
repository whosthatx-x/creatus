# The graphical app test of TechnoKreate
# Formerly known as Creatus
#
#         
#      
     

import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QListWidget, QListWidgetItem, QFrame,
                             QFileDialog, QInputDialog, QMessageBox, QTabWidget, QTextEdit,
                             QSpinBox, QDoubleSpinBox, QComboBox)
from PyQt5.QtGui import QDrag, QPixmap, QIcon
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, QSize
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from io import BytesIO

class DraggableButton(QPushButton):
    def __init__(self, text, icon_path):
        super().__init__(text)
        self.setIcon(QIcon(icon_path))
        self.setIconSize(QSize(32, 32))

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.text())
            drag.setMimeData(mime)
            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.exec_(Qt.MoveAction)

class DropArea(QFrame):
    itemDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Sunken | QFrame.StyledPanel)
        self.setMinimumHeight(100)

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        self.itemDropped.emit(event.mimeData().text())

class ModelBuilder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Creatus Model Builder')
        self.setGeometry(100, 100, 1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Components panel
        components_panel = QWidget()
        components_layout = QVBoxLayout(components_panel)
        components_layout.addWidget(QLabel("Components"))

        components = [
            ("Data Source", "icons/data.png"),
            ("Preprocessing", "icons/preprocess.png"),
            ("Conv2D", "icons/conv2d.png"),
            ("MaxPooling2D", "icons/maxpool.png"),
            ("Dense", "icons/dense.png"),
            ("Dropout", "icons/dropout.png"),
            ("Flatten", "icons/flatten.png")
        ]
        for component, icon_path in components:
            btn = DraggableButton(component, icon_path)
            components_layout.addWidget(btn)

        components_layout.addStretch(1)
        main_layout.addWidget(components_panel)

        # Pipeline area
        pipeline_widget = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_widget)
        pipeline_layout.addWidget(QLabel("Model Pipeline"))

        self.pipeline_list = QListWidget()
        self.drop_area = DropArea()
        self.drop_area.itemDropped.connect(self.addToPipeline)

        pipeline_layout.addWidget(self.drop_area)
        pipeline_layout.addWidget(self.pipeline_list)

        # Buttons
        button_layout = QHBoxLayout()
        load_data_button = QPushButton("Load Data")
        load_data_button.clicked.connect(self.loadData)
        train_button = QPushButton("Train Model")
        train_button.clicked.connect(self.trainModel)
        save_button = QPushButton("Save Model")
        save_button.clicked.connect(self.saveModel)
        button_layout.addWidget(load_data_button)
        button_layout.addWidget(train_button)
        button_layout.addWidget(save_button)
        pipeline_layout.addLayout(button_layout)

        main_layout.addWidget(pipeline_widget, stretch=2)

        # Training parameters
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.addWidget(QLabel("Training Parameters"))

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(10)
        params_layout.addWidget(QLabel("Epochs:"))
        params_layout.addWidget(self.epochs_input)

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.0001, 0.1)
        self.lr_input.setSingleStep(0.0001)
        self.lr_input.setValue(0.001)
        params_layout.addWidget(QLabel("Learning Rate:"))
        params_layout.addWidget(self.lr_input)

        self.optimizer_input = QComboBox()
        self.optimizer_input.addItems(["Adam", "SGD", "RMSprop"])
        params_layout.addWidget(QLabel("Optimizer:"))
        params_layout.addWidget(self.optimizer_input)

        main_layout.addWidget(params_widget)

        # Output tabs
        self.tabs = QTabWidget()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.tabs.addTab(self.console_output, "Console")
        self.visualization = QLabel("Model visualization will appear here after training")
        self.tabs.addTab(self.visualization, "Visualization")
        main_layout.addWidget(self.tabs, stretch=2)

    def addToPipeline(self, component):
        item = QListWidgetItem(component)
        if component == "Conv2D":
            filters, ok = QInputDialog.getInt(self, "Conv2D", "Number of filters:", 32, 1, 1024)
            if ok:
                item.setData(Qt.UserRole, {"filters": filters})
        elif component == "Dense":
            units, ok = QInputDialog.getInt(self, "Dense", "Number of units:", 64, 1, 1024)
            if ok:
                item.setData(Qt.UserRole, {"units": units})
        elif component == "Dropout":
            rate, ok = QInputDialog.getDouble(self, "Dropout", "Dropout rate:", 0.5, 0, 1, 2)
            if ok:
                item.setData(Qt.UserRole, {"rate": rate})
        self.pipeline_list.addItem(item)

    def loadData(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "NumPy Files (*.npy);;All Files (*)", options=options)
        if fileName:
            self.data = np.load(fileName)
            self.console_output.append(f"Data loaded: {self.data.shape}")

    def trainModel(self):
        if self.data is None:
            QMessageBox.warning(self, "Error", "Please load data first!")
            return

        pipeline = []
        for i in range(self.pipeline_list.count()):
            item = self.pipeline_list.item(i)
            component = item.text()
            params = item.data(Qt.UserRole)
            pipeline.append((component, params))

        self.model = Sequential()
        input_shape = self.data.shape[1:]
        for component, params in pipeline:
            if component == "Conv2D":
                self.model.add(Conv2D(params['filters'], (3, 3), activation='relu', input_shape=input_shape))
            elif component == "MaxPooling2D":
                self.model.add(MaxPooling2D((2, 2)))
            elif component == "Dense":
                self.model.add(Dense(params['units'], activation='relu'))
            elif component == "Dropout":
                self.model.add(Dropout(params['rate']))
            elif component == "Flatten":
                self.model.add(Flatten())

        self.model.add(Dense(10, activation='softmax'))  # Assuming 10 classes, adjust as needed

        optimizer = self.optimizer_input.currentText().lower()
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        X_train, X_test, y_train, y_test = train_test_split(self.data, np.zeros((self.data.shape[0], 10)), test_size=0.2)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs_input.value(),
            validation_data=(X_test, y_test),
            callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=self.update_console)]
        )

        self.visualize_training(history)

    def update_console(self, epoch, logs):
        self.console_output.append(f"Epoch {epoch+1}: loss = {logs['loss']:.4f}, accuracy = {logs['accuracy']:.4f}")

    def visualize_training(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.visualization.setPixmap(QPixmap.fromImage(QImage.fromData(buf.getvalue())))
        self.tabs.setCurrentIndex(1)  # Switch to visualization tab

    def saveModel(self):
        if self.model is None:
            QMessageBox.warning(self, "Error", "No model to save!")
            return

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "H5 Files (*.h5);;All Files (*)", options=options)
        if fileName:
            self.model.save(fileName)
            self.console_output.append(f"Model saved to {fileName}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ModelBuilder()
    ex.show()
    sys.exit(app.exec_())
