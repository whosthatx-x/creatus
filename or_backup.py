# import sys
# import operator
# import os
# import importlib
# import functools
# import random
# import math
# import datetime
# import json
# import re
# import collections
# import itertools
# import statistics
# import urllib.request
# import xml.etree.ElementTree as ET
# import csv
# import sqlite3
# import hashlib
# import base64
# import zlib
# import threading
# import multiprocessing
# import asyncio
# import typing
# from termcolor import colored
# import math
# import cv2
# import librosa
# import scapy.all as scapy
# # import nmap
# import webbrowser
# import subprocess
# import socket
# import ssl
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sympy
# import qiskit
# import pygame
# import pyaudio
# import wave
# import cryptography
# from cryptography.fernet import Fernet
# import requests




# try:
#     import numpy as np
#     NUMPY_AVAILABLE = True
# except ImportError:
#     NUMPY_AVAILABLE = False

# printOptions = ["Created by Pranav Lejith (Amphibiar)", """Created by Pranav "Amphibiar" Lejith"""]
# devCommands = ['amphibiar', 'developer', 'command override-amphibiar', 'emergency override-amphibiar']

# class OrionInterpreter:
#     def __init__(self, filename=None, interactive=False):
#         self.filename = filename
#         self.variables = {}
#         self.functions = {}
#         self.classes = {}
#         self.lines = []
#         self.current_line = 0
#         self.interactive = interactive
#         self.ops = {
#             '+': operator.add,
#             '-': operator.sub,
#             '*': operator.mul,
#             '/': operator.truediv,
#             '%': operator.mod,
#             '**': operator.pow,
#             '//': operator.floordiv,
#             '==': operator.eq,
#             '!=': operator.ne,
#             '<': operator.lt,
#             '<=': operator.le,
#             '>': operator.gt,
#             '>=': operator.ge,
#             'and': operator.and_,
#             'or': operator.or_,
#             'not': operator.not_,
#         }

#     # Quantum Computing Features
#     def create_quantum_circuit(self, args):
#         """Create a quantum circuit with specified number of qubits."""
#         num_qubits = args[0]
#         return qiskit.QuantumCircuit(num_qubits)

#     def run_quantum_circuit(self, args):
#         """Run a quantum circuit on a simulator or real quantum computer."""
#         circuit = args[0]
#         backend = args[1] if len(args) > 1 else qiskit.Aer.get_backend('qasm_simulator')
#         job = qiskit.execute(circuit, backend)
#         result = job.result()
#         return result.get_counts()

#     def quantum_gates(self, args):
#         """Apply quantum gates to a circuit."""
#         circuit = args[0]
#         gate_type = args[1]
#         qubits = args[2]
        
#         if gate_type == 'h':  # Hadamard
#             circuit.h(qubits)
#         elif gate_type == 'x':  # Pauli-X
#             circuit.x(qubits)
#         elif gate_type == 'y':  # Pauli-Y
#             circuit.y(qubits)
#         elif gate_type == 'z':  # Pauli-Z
#             circuit.z(qubits)
#         elif gate_type == 'cx':  # CNOT
#             circuit.cx(qubits[0], qubits[1])
#         elif gate_type == 'measure':  # Measurement
#             circuit.measure(qubits, qubits)
#         return circuit

#     # Robotics Features
#     def create_robot(self, args):
#         """Create a robot instance with specified parameters."""
#         robot_type = args[0]
#         if robot_type == 'mobile':
#             return {'type': 'mobile', 'position': [0, 0], 'orientation': 0}
#         elif robot_type == 'manipulator':
#             return {'type': 'manipulator', 'joints': [0]*6}
#         return None

#     def move_robot(self, args):
#         """Move the robot to specified coordinates."""
#         robot = args[0]
#         x, y = args[1], args[2]
#         if robot['type'] == 'mobile':
#             robot['position'] = [x, y]
#         return robot

#     def rotate_robot(self, args):
#         """Rotate the robot by specified angle."""
#         robot = args[0]
#         angle = args[1]
#         if robot['type'] == 'mobile':
#             robot['orientation'] += angle
#         elif robot['type'] == 'manipulator':
#             for i in range(len(robot['joints'])):
#                 robot['joints'][i] += angle
#         return robot

#     # Computer Vision Features
#     def load_image(self, args):
#         """Load an image from file."""
#         image_path = args[0]
#         return cv2.imread(image_path)

#     def detect_faces(self, args):
#         """Detect faces in an image."""
#         image = args[0]
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#         return faces

#     def object_detection(self, args):
#         """Perform object detection on an image."""
#         image = args[0]
#         net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
#         blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#         net.setInput(blob)
#         detections = net.forward()
#         return detections

#     def image_segmentation(self, args):
#         """Perform image segmentation."""
#         image = args[0]
#         segmented = cv2.pyrMeanShiftFiltering(image, 21, 51)
#         return segmented

#     # Audio Processing Features
#     def load_audio(self, args):
#         """Load an audio file."""
#         audio_path = args[0]
#         return librosa.load(audio_path)

#     def analyze_audio(self, args):
#         """Analyze audio features."""
#         audio = args[0]
#         sr = args[1]
#         features = {
#             'chroma': librosa.feature.chroma_stft(y=audio, sr=sr),
#             'mfcc': librosa.feature.mfcc(y=audio, sr=sr),
#             'spectral_contrast': librosa.feature.spectral_contrast(y=audio, sr=sr)
#         }
#         return features

#     def generate_audio(self, args):
#         """Generate audio based on parameters."""
#         duration = args[0]
#         sample_rate = args[1]
#         frequency = args[2]
#         t = np.linspace(0, duration, int(sample_rate * duration))
#         audio = np.sin(2 * np.pi * frequency * t)
#         return audio, sample_rate

#     # Network Security Features
#     def scan_network(self, args):
#         """Network scanning is not available in this version."""
#         return []

#     def capture_packets(self, args):
#         """Capture network packets."""
#         interface = args[0]
#         count = args[1] if len(args) > 1 else 10
#         packets = scapy.sniff(iface=interface, count=count)
#         return packets

#     def analyze_packets(self, args):
#         """Analyze captured packets."""
#         packets = args[0]
#         analysis = []
#         for packet in packets:
#             if packet.haslayer(scapy.IP):
#                 analysis.append({
#                     'src': packet[scapy.IP].src,
#                     'dst': packet[scapy.IP].dst,
#                     'proto': packet[scapy.IP].proto
#                 })
#         return analysis

#     # VSCode Extension Features
#     def create_vscode_extension(self, args):
#         """Create a new VSCode extension."""
#         extension_name = args[0]
#         extension_path = f"extensions/{extension_name}"
#         os.makedirs(extension_path, exist_ok=True)
        
#         # Create package.json
#         package_json = {
#             "name": extension_name,
#             "displayName": extension_name,
#             "version": "0.0.1",
#             "engines": {"vscode": "^1.60.0"},
#             "activationEvents": ["onCommand:extension.helloWorld"],
#             "main": "./extension.js",
#             "contributes": {
#                 "commands": [{
#                     "command": "extension.helloWorld",
#                     "title": "Hello World"
#                 }]
#             }
#         }
        
#         with open(f"{extension_path}/package.json", 'w') as f:
#             json.dump(package_json, f, indent=2)
            
#         # Create extension.js
#         with open(f"{extension_path}/extension.js", 'w') as f:
#             f.write("""
# const vscode = require('vscode');

# function activate(context) {
#     let disposable = vscode.commands.registerCommand('extension.helloWorld', () => {
#         vscode.window.showInformationMessage('Hello World!');
#     });
#     context.subscriptions.push(disposable);
# }

# function deactivate() {}

# module.exports = {
#     activate,
#     deactivate
# };
# """)
        
#         return extension_path

#     def add_vscode_command(self, args):
#         """Add a new command to a VSCode extension."""
#         extension_path = args[0]
#         command_name = args[1]
#         command_title = args[2]
        
#         # Update package.json
#         with open(f"{extension_path}/package.json", 'r') as f:
#             package = json.load(f)
            
#         if "contributes" not in package:
#             package["contributes"] = {"commands": []}
        
#         package["contributes"]["commands"].append({
#             "command": f"extension.{command_name}",
#             "title": command_title
#         })
        
#         with open(f"{extension_path}/package.json", 'w') as f:
#             json.dump(package, f, indent=2)
            
#         # Add command to extension.js
#         with open(f"{extension_path}/extension.js", 'r') as f:
#             content = f.read()
        
#         command_code = f"""
#     let {command_name} = vscode.commands.registerCommand('extension.{command_name}', () => {{
#         vscode.window.showInformationMessage('{command_title}');
#     }});
#     context.subscriptions.push({command_name});
# """
        
#         content = content.replace("function activate(context) {", 
#                                 f"function activate(context) {{\n{command_code}\n")
        
#         with open(f"{extension_path}/extension.js", 'w') as f:
#             f.write(content)
            
#         return True

#     def register_vscode_listener(self, args):
#         """Register a document or workspace listener."""
#         extension_path = args[0]
#         listener_type = args[1]
#         event = args[2]
        
#         # Add listener to extension.js
#         with open(f"{extension_path}/extension.js", 'r') as f:
#             content = f.read()
        
#         listener_code = f"""
#     let {event}Listener = vscode.workspace.on{listener_type}({event}, (e) => {{
#         vscode.window.showInformationMessage('Event triggered: {event}');
#     }});
#     context.subscriptions.push({event}Listener);
# """
        
#         content = content.replace("function activate(context) {", 
#                                 f"function activate(context) {{\n{listener_code}\n")
        
#         with open(f"{extension_path}/extension.js", 'w') as f:
#             f.write(content)
            
#         return True

#     # Embedded Systems Features
#     def setup_arduino(self, args):
#         """Setup Arduino board."""
#         board_type = args[0]
#         port = args[1]
#         baud_rate = args[2] if len(args) > 2 else 9600
        
#         import serial
#         try:
#             arduino = serial.Serial(port, baud_rate)
#             arduino.close()
#             return True
#         except:
#             return False

#     def setup_raspberry_pi(self, args):
#         """Setup Raspberry Pi GPIO."""
#         import RPi.GPIO as GPIO
#         GPIO.setmode(GPIO.BCM)
        
#         # Set up pins
#         for pin in args:
#             GPIO.setup(pin, GPIO.OUT)
            
#         return True

#     def gpio_control(self, args):
#         """Control GPIO pins."""
#         pin = args[0]
#         state = args[1]
        
#         import RPi.GPIO as GPIO
#         GPIO.output(pin, state)
#         return True

#     def read_sensor(self, args):
#         """Read from a sensor."""
#         sensor_type = args[0]
#         pin = args[1]
        
#         if sensor_type == 'temperature':
#             import adafruit_dht
#             sensor = adafruit_dht.DHT22(pin)
#             return sensor.temperature
#         elif sensor_type == 'humidity':
#             import adafruit_dht
#             sensor = adafruit_dht.DHT22(pin)
#             return sensor.humidity
#         elif sensor_type == 'light':
#             import RPi.GPIO as GPIO
#             GPIO.setup(pin, GPIO.IN)
#             return GPIO.input(pin)
#         return None

#     def control_motor(self, args):
#         """Control a motor."""
#         motor_type = args[0]
#         direction = args[1]
#         speed = args[2]
        
#         if motor_type == 'dc':
#             import RPi.GPIO as GPIO
#             enable_pin = args[3]
#             direction_pin = args[4]
            
#             GPIO.output(enable_pin, True)
#             GPIO.output(direction_pin, direction)
#             return True
#         elif motor_type == 'servo':
#             import RPi.GPIO as GPIO
#             pwm_pin = args[3]
#             pwm = GPIO.PWM(pwm_pin, 50)
#             pwm.start(0)
#             pwm.ChangeDutyCycle(speed)
#             return True
#         return False

#     def setup_i2c(self, args):
#         """Setup I2C communication."""
#         import smbus
#         bus = smbus.SMBus(1)
#         address = args[0]
        
#         try:
#             bus.read_byte(address)
#             return True
#         except:
#             return False

#     def read_i2c(self, args):
#         """Read from I2C device."""
#         import smbus
#         bus = smbus.SMBus(1)
#         address = args[0]
#         register = args[1]
        
#         try:
#             return bus.read_byte_data(address, register)
#         except:
#             return None

#     def write_i2c(self, args):
#         """Write to I2C device."""
#         import smbus
#         bus = smbus.SMBus(1)
#         address = args[0]
#         register = args[1]
#         value = args[2]
        
#         try:
#             bus.write_byte_data(address, register, value)
#             return True
#         except:
#             return False

#     # Low-Level Programming Features
#     def compile_c(self, args):
#         """Compile C code."""
#         source_file = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['gcc', source_file, '-o', output_file], check=True)
#             return True
#         except:
#             return False

#     def compile_cpp(self, args):
#         """Compile C++ code."""
#         source_file = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['g++', source_file, '-o', output_file], check=True)
#             return True
#         except:
#             return False

#     def compile_rust(self, args):
#         """Compile Rust code."""
#         source_file = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['rustc', source_file, '-o', output_file], check=True)
#             return True
#         except:
#             return False

#     def generate_assembly(self, args):
#         """Generate assembly code from C."""
#         source_file = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['gcc', '-S', source_file, '-o', output_file], check=True)
#             return True
#         except:
#             return False

#     def analyze_binary(self, args):
#         """Analyze binary file."""
#         binary_file = args[0]
        
#         import lief
#         binary = lief.parse(binary_file)
        
#         analysis = {
#             'type': binary.format.name,
#             'entry_point': binary.entrypoint,
#             'sections': [s.name for s in binary.sections],
#             'symbols': [s.name for s in binary.symbols]
#         }
        
#         return analysis

#     def disassemble_code(self, args):
#         """Disassemble binary code."""
#         binary_file = args[0]
        
#         import capstone
        
#         with open(binary_file, 'rb') as f:
#             code = f.read()
        
#         md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
#         disassembly = list(md.disasm(code, 0x1000))
        
#         return [{'address': hex(i.address), 'mnemonic': i.mnemonic, 'op_str': i.op_str} 
#                 for i in disassembly]

#     def create_shared_library(self, args):
#         """Create shared library from C code."""
#         source_file = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['gcc', '-shared', '-fPIC', source_file, '-o', output_file], check=True)
#             return True
#         except:
#             return False

#     def create_static_library(self, args):
#         """Create static library from object files."""
#         object_files = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['ar', 'rcs', output_file] + object_files, check=True)
#             return True
#         except:
#             return False

#     def optimize_code(self, args):
#         """Optimize C/C++ code."""
#         source_file = args[0]
#         output_file = args[1]
#         optimization_level = args[2] if len(args) > 2 else 3
        
#         import subprocess
#         try:
#             subprocess.run(['gcc', '-O' + str(optimization_level), source_file, '-o', output_file], check=True)
#             return True
#         except:
#             return False

#     def profile_code(self, args):
#         """Profile C/C++ code execution."""
#         executable = args[0]
        
#         import subprocess
#         try:
#             subprocess.run(['gprof', executable], check=True)
#             return True
#         except:
#             return False

#     def create_kernel_module(self, args):
#         """Create Linux kernel module."""
#         source_file = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['make', '-C', '/lib/modules/$(uname -r)/build', 'M=' + os.path.dirname(source_file)], check=True)
#             return True
#         except:
#             return False

#     def create_firmware(self, args):
#         """Create firmware image."""
#         source_files = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['mkimage', '-A', 'arm', '-O', 'linux', '-T', 'kernel', '-C', 'none', 
#                           '-a', '0x00008000', '-e', '0x00008000', '-n', 'Firmware', 
#                           '-d', source_files[0], output_file], check=True)
#             return True
#         except:
#             return False

#     def create_bootloader(self, args):
#         """Create bootloader."""
#         source_file = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['nasm', '-f', 'bin', source_file, '-o', output_file], check=True)
#             return True
#         except:
#             return False

#     def create_driver(self, args):
#         """Create device driver."""
#         source_file = args[0]
#         output_file = args[1]
        
#         import subprocess
#         try:
#             subprocess.run(['gcc', '-D', '__KERNEL__', '-DMODULE', '-Wall', '-Wstrict-prototypes', 
#                           '-O2', '-c', source_file], check=True)
#             subprocess.run(['ld', '-r', '-o', output_file, source_file.replace('.c', '.o')], check=True)
#             return True
#         except:
#             return False

#     def create_embedded_system(self, args):
#         """Create embedded system project."""
#         project_name = args[0]
#         target_platform = args[1]
        
#         import os
#         os.makedirs(project_name, exist_ok=True)
        
#         # Create Makefile
#         with open(f"{project_name}/Makefile", 'w') as f:
#             f.write(f"""
# CC = gcc
# CFLAGS = -Wall -Werror -O2
# LDFLAGS = 
# TARGET = {project_name}
# SOURCES = $(wildcard *.c)
# OBJECTS = $(SOURCES:.c=.o)

# all: $(TARGET)

# $(TARGET): $(OBJECTS)
# 	$(CC) $(LDFLAGS) -o $@ $^

# clean:
# 	rm -f $(OBJECTS) $(TARGET)
# """)
        
#         # Create main.c
#         with open(f"{project_name}/main.c", 'w') as f:
#             f.write("""
# #include <stdio.h>

# int main() {
#     printf("Hello, Embedded World!\n");
#     return 0;
# }
# """)
        
#         return True

#     def create_rtos(self, args):
#         """Create real-time operating system."""
#         project_name = args[0]
#         os.makedirs(project_name, exist_ok=True)
        
#         # Create kernel.c
#         with open(f"{project_name}/kernel.c", 'w') as f:
#             f.write("""
# #include <stdint.h>
# #include <stdbool.h>

# #define MAX_TASKS 10

# typedef struct {
#     void (*task)(void);
#     uint32_t priority;
#     bool active;
# } task_t;

# task_t tasks[MAX_TASKS];
# uint32_t current_task = 0;

# void add_task(void (*task)(void), uint32_t priority) {
#     for (uint32_t i = 0; i < MAX_TASKS; i++) {
#         if (!tasks[i].active) {
#             tasks[i].task = task;
#             tasks[i].priority = priority;
#             tasks[i].active = true;
#             break;
#         }
#     }
# }

# void schedule() {
#     uint32_t highest_priority = 0;
#     uint32_t next_task = 0;
    
#     for (uint32_t i = 0; i < MAX_TASKS; i++) {
#         if (tasks[i].active && tasks[i].priority > highest_priority) {
#             highest_priority = tasks[i].priority;
#             next_task = i;
#         }
#     }
    
#     current_task = next_task;
# }

# void start_kernel() {
#     while (true) {
#         schedule();
#         tasks[current_task].task();
#     }
# }
# """)
        
#         return True

#     # Server Technologies Features
#     def setup_nginx(self, args):
#         """Setup Nginx web server."""
#         server_name = args[0]
#         root_dir = args[1]
        
#         import os
        
#         # Create Nginx configuration
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     root {root_dir};
#     index index.html index.htm;
    
#     location / {{
#         try_files $uri $uri/ =404;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/default', 'w') as f:
#             f.write(config)
        
#         # Create root directory
#         os.makedirs(root_dir, exist_ok=True)
        
#         # Create default index.html
#         with open(f"{root_dir}/index.html", 'w') as f:
#             f.write("""
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Welcome to {server_name}</title>
# </head>
# <body>
#     <h1>Welcome to {server_name}</h1>
#     <p>Powered by Nginx</p>
# </body>
# </html>
# """)
        
#         return True

#     def setup_apache(self, args):
#         """Setup Apache web server."""
#         server_name = args[0]
#         root_dir = args[1]
        
#         import os
        
#         # Create Apache configuration
#         config = f"""
# <VirtualHost *:80>
#     ServerName {server_name}
#     DocumentRoot {root_dir}
    
#     <Directory {root_dir}>
#         Options Indexes FollowSymLinks MultiViews
#         AllowOverride All
#         Require all granted
#     </Directory>
# </VirtualHost>
# """
        
#         with open('/etc/apache2/sites-available/000-default.conf', 'w') as f:
#             f.write(config)
        
#         # Create root directory
#         os.makedirs(root_dir, exist_ok=True)
        
#         # Create default index.html
#         with open(f"{root_dir}/index.html", 'w') as f:
#             f.write("""
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Welcome to {server_name}</title>
# </head>
# <body>
#     <h1>Welcome to {server_name}</h1>
#     <p>Powered by Apache</p>
# </body>
# </html>
# """)
        
#         return True

#     def configure_web_server(self, args):
#         """Configure web server settings."""
#         server_type = args[0]
#         setting = args[1]
#         value = args[2]
        
#         if server_type == 'nginx':
#             with open('/etc/nginx/nginx.conf', 'r') as f:
#                 config = f.read()
            
#             config = config.replace(f'#{setting} ', f'{setting} {value};')
            
#             with open('/etc/nginx/nginx.conf', 'w') as f:
#                 f.write(config)
            
#         elif server_type == 'apache':
#             with open('/etc/apache2/apache2.conf', 'r') as f:
#                 config = f.read()
            
#             config = config.replace(f'#{setting} ', f'{setting} {value}')
            
#             with open('/etc/apache2/apache2.conf', 'w') as f:
#                 f.write(config)
                
#         return True

#     def setup_reverse_proxy(self, args):
#         """Setup reverse proxy."""
#         server_name = args[0]
#         backend_servers = args[1]
        
#         config = f"""
# upstream {server_name} {{
#     {"".join([f"    server {server};\n" for server in backend_servers])}
# }}

# server {{
#     listen 80;
#     server_name {server_name};
    
#     location / {{
#         proxy_pass http://{server_name};
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/reverse-proxy', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_load_balancer(self, args):
#         """Setup load balancer."""
#         server_name = args[0]
#         backend_servers = args[1]
        
#         config = f"""
# upstream {server_name} {{
#     {"".join([f"    server {server};\n" for server in backend_servers])}
    
#     # Load balancing method
#     least_conn;
# }}

# server {{
#     listen 80;
#     server_name {server_name};
    
#     location / {{
#         proxy_pass http://{server_name};
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/load-balancer', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_cache(self, args):
#         """Setup caching."""
#         server_type = args[0]
#         cache_type = args[1]
        
#         if server_type == 'nginx':
#             if cache_type == 'browser':
#                 config = """
# add_header Cache-Control "public, max-age=31536000";
# add_header Expires "1y";
# """
#             elif cache_type == 'proxy':
#                 config = """
# proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=STATIC:10m inactive=7d;
# proxy_cache_key "$scheme$request_method$host$request_uri";
# proxy_cache STATIC;
# """
            
#             with open('/etc/nginx/nginx.conf', 'a') as f:
#                 f.write(config)
                
#         return True

#     def setup_ssl(self, args):
#         """Setup SSL/TLS."""
#         server_name = args[0]
#         cert_file = args[1]
#         key_file = args[2]
        
#         config = f"""
# server {{
#     listen 443 ssl;
#     server_name {server_name};
    
#     ssl_certificate {cert_file};
#     ssl_certificate_key {key_file};
    
#     ssl_protocols TLSv1.2 TLSv1.3;
#     ssl_ciphers HIGH:!aNULL:!MD5;
    
#     location / {{
#         try_files $uri $uri/ =404;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/ssl', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_websocket(self, args):
#         """Setup WebSocket."""
#         server_name = args[0]
#         websocket_path = args[1]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     location {websocket_path} {{
#         proxy_pass http://websocket_backend;
#         proxy_http_version 1.1;
#         proxy_set_header Upgrade $http_upgrade;
#         proxy_set_header Connection "upgrade";
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/websocket', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_rate_limiting(self, args):
#         """Setup rate limiting."""
#         server_name = args[0]
#         limit = args[1]
        
#         config = f"""
# limit_req_zone $binary_remote_addr zone=one:10m rate={limit}r/s;

# server {{
#     listen 80;
#     server_name {server_name};
    
#     location / {{
#         limit_req zone=one burst=5;
#         try_files $uri $uri/ =404;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/rate-limit', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_logging(self, args):
#         """Setup logging."""
#         server_name = args[0]
#         log_format = args[1]
        
#         config = f"""
# log_format custom '{log_format}';

# server {{
#     listen 80;
#     server_name {server_name};
    
#     access_log /var/log/nginx/access.log custom;
#     error_log /var/log/nginx/error.log;
    
#     location / {{
#         try_files $uri $uri/ =404;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/logging', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_monitoring(self, args):
#         """Setup monitoring."""
#         server_name = args[0]
#         metrics_path = args[1]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     location {metrics_path} {{
#         stub_status;
#         allow 127.0.0.1;
#         deny all;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/monitoring', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_caching(self, args):
#         """Setup caching."""
#         server_name = args[0]
#         cache_time = args[1]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     location / {{
#         proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=STATIC:10m inactive={cache_time};
#         proxy_cache_key "$scheme$request_method$host$request_uri";
#         proxy_cache STATIC;
        
#         try_files $uri $uri/ =404;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/caching', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_compression(self, args):
#         """Setup compression."""
#         server_name = args[0]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     gzip on;
#     gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
#     location / {{
#         try_files $uri $uri/ =404;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/compression', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_security_headers(self, args):
#         """Setup security headers."""
#         server_name = args[0]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     add_header X-Frame-Options "SAMEORIGIN";
#     add_header X-XSS-Protection "1; mode=block";
#     add_header X-Content-Type-Options "nosniff";
#     add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
#     location / {{
#         try_files $uri $uri/ =404;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/security', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_cors(self, args):
#         """Setup CORS."""
#         server_name = args[0]
#         allowed_origins = args[1]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     location / {{
#         add_header 'Access-Control-Allow-Origin' '{" ".join(allowed_origins)}';
#         add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
#         add_header 'Access-Control-Allow-Headers' 'DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type';
        
#         if ($request_method = 'OPTIONS') {{
#             return 204;
#         }}
        
#         try_files $uri $uri/ =404;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/cors', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_websocket_proxy(self, args):
#         """Setup WebSocket proxy."""
#         server_name = args[0]
#         websocket_path = args[1]
#         backend_server = args[2]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     location {websocket_path} {{
#         proxy_pass {backend_server};
#         proxy_http_version 1.1;
#         proxy_set_header Upgrade $http_upgrade;
#         proxy_set_header Connection "upgrade";
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/websocket-proxy', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_api_gateway(self, args):
#         """Setup API gateway."""
#         server_name = args[0]
#         api_routes = args[1]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     {"".join([f"    location {route['path']} {{\n        proxy_pass {route['backend']};\n    }}\n" for route in api_routes])}
# }}
# """
        
#         with open('/etc/nginx/sites-available/api-gateway', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_microservices(self, args):
#         """Setup microservices architecture."""
#         server_name = args[0]
#         services = args[1]
        
#         config = f"""
# upstream services {{
#     {"".join([f"    server {service['name']}:{service['port']};\n" for service in services])}
# }}

# server {{
#     listen 80;
#     server_name {server_name};
    
#     location / {{
#         proxy_pass http://services;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/microservices', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_service_discovery(self, args):
#         """Setup service discovery."""
#         server_name = args[0]
#         discovery_service = args[1]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     location /service-discovery {{
#         proxy_pass {discovery_service};
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/service-discovery', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_load_balancer_with_health_check(self, args):
#         """Setup load balancer with health check."""
#         server_name = args[0]
#         backend_servers = args[1]
#         health_check_path = args[2]
        
#         config = f"""
# upstream {server_name} {{
#     {"".join([f"    server {server};\n" for server in backend_servers])}
    
#     # Health check
#     health_check uri={health_check_path} interval=5s rise=2 fall=2 timeout=1s;
# }}

# server {{
#     listen 80;
#     server_name {server_name};
    
#     location / {{
#         proxy_pass http://{server_name};
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#     }}
# }}
# """
        
#         with open('/etc/nginx/sites-available/load-balancer-with-health-check', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_api_rate_limiting(self, args):
#         """Setup API rate limiting."""
#         server_name = args[0]
#         api_routes = args[1]
        
#         config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     {"".join([f"    location {route['path']} {{\n        limit_req zone=api burst=5;\n        proxy_pass {route['backend']};\n    }}\n" for route in api_routes])}
# }}
# """
        
#         with open('/etc/nginx/sites-available/api-rate-limiting', 'w') as f:
#             f.write(config)
            
#         return True

#     def setup_api_security(self, args):
#         """Setup API security."""
#         try:
#             server_name = args[0]
#             api_routes = args[1]
            
#             config = f"""
# server {{
#     listen 80;
#     server_name {server_name};
    
#     {"".join([f"    location {route['path']} {{\n        add_header X-Frame-Options \"SAMEORIGIN\";\n        add_header X-XSS-Protection \"1; mode=block\";\n        add_header X-Content-Type-Options \"nosniff\";\n        proxy_pass {route['backend']};\n    }}\n" for route in api_routes])}
# }}
# """
            
#             with open('/etc/nginx/sites-available/api-security', 'w') as f:
#                 f.write(config)
#             return True
#         except Exception:
#             return False

#     # Built-in functions
#     def len_func(self, args):
#         return len(args[0])

#     def max_func(self, args):
#         return max(args[0])

#     def min_func(self, args):
#         return min(args[0])

#     def sum_func(self, args):
#         return sum(args[0])

#     def abs_func(self, args):
#         return abs(args[0])

#     def round_func(self, args):
#         return round(*args)

#     def type_func(self, args):
#         return type(args[0]).__name__

#     def int_func(self, args):
#         return int(args[0])

#     def float_func(self, args):
#         return float(args[0])

#     def str_func(self, args):
#         return str(args[0])

#     def bool_func(self, args):
#         return bool(args[0])

#     def list_func(self, args):
#         return list(args[0])

#     def tuple_func(self, args):
#         return tuple(args[0])

#     def set_func(self, args):
#         return set(args[0])

#     def dict_func(self, args):
#         return dict(args[0])

#     def range_func(self, args):
#         return list(range(*args))

#     def enumerate_func(self, args):
#         return list(enumerate(*args))

#     def zip_func(self, args):
#         return list(zip(*args))

#     def map_func(self, args):
#         return list(map(*args))

#     def filter_func(self, args):
#         return list(filter(*args))

#     def reduce_func(self, args):
#         return functools.reduce(*args)

#     def sorted_func(self, args):
#         return sorted(*args)

#     def reversed_func(self, args):
#         return list(reversed(args[0]))

#     def any_func(self, args):
#         return any(args[0])

#     def all_func(self, args):
#         return all(args[0])

#     def chr_func(self, args):
#         return chr(args[0])

#     def ord_func(self, args):
#         return ord(args[0])

#     def bin_func(self, args):
#         return bin(args[0])

#     def oct_func(self, args):
#         return oct(args[0])

#     def hex_func(self, args):
#         return hex(args[0])

#     def id_func(self, args):
#         return id(args[0])

#     def isinstance_func(self, args):
#         return isinstance(args[0], args[1])

#     def issubclass_func(self, args):
#         return issubclass(args[0], args[1])

#     def callable_func(self, args):
#         return callable(args[0])

#     def getattr_func(self, args):
#         return getattr(*args)

#     def setattr_func(self, args):
#         setattr(*args)

#     def hasattr_func(self, args):
#         return hasattr(*args)

#     def delattr_func(self, args):
#         delattr(*args)

#     def open_func(self, args):
#         return open(*args)

#     def input_func(self, args):
#         return input(*args)

#     def print_func(self, args):
#         print(*args)

#     def len_func(self, args):
#         return len(args[0])

#     def upper_func(self, args):
#         return args[0].upper()

#     def lower_func(self, args):
#         return args[0].lower()

#     def capitalize_func(self, args):
#         return args[0].capitalize()

#     def title_func(self, args):
#         return args[0].title()

#     def strip_func(self, args):
#         return args[0].strip()

#     def split_func(self, args):
#         return args[0].split(*args[1:])

#     def join_func(self, args):
#         return args[0].join(args[1])

#     def replace_func(self, args):
#         return args[0].replace(*args[1:])

#     def startswith_func(self, args):
#         return args[0].startswith(args[1])

#     def endswith_func(self, args):
#         return args[0].endswith(args[1])

#     def find_func(self, args):
#         return args[0].find(*args[1:])

#     def count_func(self, args):
#         return args[0].count(args[1])

#     def isalpha_func(self, args):
#         return args[0].isalpha()

#     def isdigit_func(self, args):
#         return args[0].isdigit()

#     def isalnum_func(self, args):
#         return args[0].isalnum()

#     def islower_func(self, args):
#         return args[0].islower()

#     def isupper_func(self, args):
#         return args[0].isupper()

#     def append_func(self, args):
#         args[0].append(args[1])

#     def extend_func(self, args):
#         args[0].extend(args[1])

#     def insert_func(self, args):
#         args[0].insert(args[1], args[2])

#     def remove_func(self, args):
#         args[0].remove(args[1])

#     def pop_func(self, args):
#         return args[0].pop(*args[1:])

#     def clear_func(self, args):
#         args[0].clear()

#     def index_func(self, args):
#         return args[0].index(*args[1:])

#     def reverse_func(self, args):
#         args[0].reverse()

#     def copy_func(self, args):
#         return args[0].copy()

#     def deepcopy_func(self, args):
#         import copy
#         return copy.deepcopy(args[0])

#     def keys_func(self, args):
#         return list(args[0].keys())

#     def values_func(self, args):
#         return list(args[0].values())

#     def items_func(self, args):
#         return list(args[0].items())

#     def get_func(self, args):
#         return args[0].get(*args[1:])

#     def update_func(self, args):
#         args[0].update(args[1])

#     def math_sin_func(self, args):
#         return math.sin(args[0])

#     def math_cos_func(self, args):
#         return math.cos(args[0])

#     def math_tan_func(self, args):
#         return math.tan(args[0])

#     def math_sqrt_func(self, args):
#         return math.sqrt(args[0])

#     def math_log_func(self, args):
#         return math.log(*args)

#     def math_exp_func(self, args):
#         return math.exp(args[0])

#     def math_floor_func(self, args):
#         return math.floor(args[0])

#     def math_ceil_func(self, args):
#         return math.ceil(args[0])

#     def random_randint_func(self, args):
#         return random.randint(*args)

#     def random_choice_func(self, args):
#         return random.choice(args[0])

#     def random_shuffle_func(self, args):
#         random.shuffle(args[0])

#     def datetime_now_func(self, args):
#         return datetime.datetime.now()

#     def datetime_date_func(self, args):
#         return datetime.date(*args)

#     def datetime_time_func(self, args):
#         return datetime.time(*args)

#     def json_dumps_func(self, args):
#         return json.dumps(*args)

#     def json_loads_func(self, args):
#         return json.loads(*args)

#     def re_search_func(self, args):
#         return re.search(*args)

#     def re_match_func(self, args):
#         return re.match(*args)

#     def re_findall_func(self, args):
#         return re.findall(*args)

#     def re_sub_func(self, args):
#         return re.sub(*args)

#     def collections_counter_func(self, args):
#         return collections.Counter(args[0])

#     def collections_defaultdict_func(self, args):
#         return collections.defaultdict(args[0])

#     def itertools_permutations_func(self, args):
#         return list(itertools.permutations(*args))

#     def itertools_combinations_func(self, args):
#         return list(itertools.combinations(*args))

#     def statistics_mean_func(self, args):
#         return statistics.mean(args[0])

#     def statistics_median_func(self, args):
#         return statistics.median(args[0])

#     def statistics_mode_func(self, args):
#         return statistics.mode(args[0])

#     def statistics_stdev_func(self, args):
#         return statistics.stdev(args[0])

#     def urllib_request_urlopen_func(self, args):
#         return urllib.request.urlopen(*args)

#     def xml_parse_func(self, args):
#         return ET.parse(*args)

#     def csv_reader_func(self, args):
#         return csv.reader(*args)

#     def csv_writer_func(self, args):
#         return csv.writer(*args)

#     def sqlite3_connect_func(self, args):
#         return sqlite3.connect(*args)

#     def hashlib_md5_func(self, args):
#         return hashlib.md5(args[0].encode()).hexdigest()

#     def hashlib_sha256_func(self, args):
#         return hashlib.sha256(args[0].encode()).hexdigest()

#     def base64_encode_func(self, args):
#         return base64.b64encode(args[0].encode()).decode()

#     def base64_decode_func(self, args):
#         return base64.b64decode(args[0]).decode()

#     def zlib_compress_func(self, args):
#         return zlib.compress(args[0].encode())

#     def zlib_decompress_func(self, args):
#         return zlib.decompress(args[0]).decode()

#     def threading_thread_func(self, args):
#         return threading.Thread(*args)

#     def multiprocessing_process_func(self, args):
#         return multiprocessing.Process(*args)

#     def asyncio_run_func(self, args):
#         return asyncio.run(*args)

#     def typing_get_type_hints_func(self, args):
#         return typing.get_type_hints(*args)

#     # TensorFlow functions (if available)
#     def tf_constant_func(self, args):
#         if TENSORFLOW_AVAILABLE:
#             return tf.constant(*args)
#         else:
#             raise Exception("TensorFlow is not available")

#     def tf_variable_func(self, args):
#         if TENSORFLOW_AVAILABLE:
#             return tf.Variable(*args)
#         else:
#             raise Exception("TensorFlow is not available")

#     def tf_matmul_func(self, args):
#         if TENSORFLOW_AVAILABLE:
#             return tf.matmul(*args)
#         else:
#             raise Exception("TensorFlow is not available")

#     def np_array_func(self, args):
#         if TENSORFLOW_AVAILABLE:
#             return np.array(*args)
#         else:
#             raise Exception("NumPy is not available")

#     def np_zeros_func(self, args):
#         if TENSORFLOW_AVAILABLE:
#             return np.zeros(*args)
#         else:
#             raise Exception("NumPy is not available")

#     def np_ones_func(self, args):
#         if TENSORFLOW_AVAILABLE:
#             return np.ones(*args)
#         else:
#             raise Exception("NumPy is not available")

#     # Custom functions
#     def isSreekuttyIdiot(self, args):
#         if len(args) != 1:
#             raise ValueError("isSreekuttyIdiot() takes exactly one argument")
#         arg = args[0]
#         if arg not in [0, 1]:
#             raise ValueError("isSreekuttyIdiot() argument must be either 0 or 1")
#         return "Yes, Sreekutty is an idiot" if arg == 1 else "No, she is not an idiot"

#     def isSaiIdiot(self, args):
#         if len(args) != 1:
#             raise ValueError("isSaiIdiot() takes exactly one argument")
#         arg = args[0]
#         if arg not in [0, 1]:
#             raise ValueError("isSaiIdiot() argument must be either 0 or 1")
#         return "Yes, Sai is an idiot" if arg == 1 else "No, Sai is not an idiot"

#     def add(self, args):
#         if len(args) != 2:
#             raise ValueError("add() takes 2 arguments only")
#         return args[0] + args[1]

#     def pickRandom(self, args):
#         return random.choice(args)

#     def calculator(self, args):
#         if len(args) != 3:
#             raise ValueError("calculator() takes 3 arguments: two numbers and an operation")
#         n1, n2, op = args
#         if op == 'add':
#             return n1 + n2
#         elif op == 'sub':
#             return n1 - n2
#         elif op == 'div':
#             return n1 / n2
#         elif op == 'mul':
#             return n1 * n2
#         else:
#             raise ValueError('Supported operations are add, sub, mul, div')

#     def hypotenuse(self,args):
#         if len(args) != 2:
#             raise ValueError("hypotenuse() takes 2 arguments (side1,side2)")
#         return math.sqrt((args[0]*args[0])+(args[1]*args[1]))

#     def coloredText(self,args):
#         if len(args) != 2:
#             raise ValueError("coloredText() taks 2 arguments: text, color")
#         return colored(args[0],args[1])

#     def displayColoredText(self,args):
#         if len(args) != 2:
#             raise ValueError("displayColoredText() taks 2 arguments: text, color")
#         print(colored(args[0],args[1]))
#     # Register built-in functions
#     # Built-in functions dictionary
#     builtin_functions = {
#         # Quantum Computing
#         'create_quantum_circuit': create_quantum_circuit,
#         'run_quantum_circuit': run_quantum_circuit,
#         'quantum_gates': quantum_gates,
        
#         # Robotics
#         'create_robot': create_robot,
#         'move_robot': move_robot,
#         'rotate_robot': rotate_robot,
        
#         # Computer Vision
#         'load_image': load_image,
#         'detect_faces': detect_faces,
#         'object_detection': object_detection,
#         'image_segmentation': image_segmentation,
        
#         # Audio Processing
#         'load_audio': load_audio,
#         'analyze_audio': analyze_audio,
#         'generate_audio': generate_audio,
        
#         # Network Security
#         'scan_network': scan_network,
#         'capture_packets': capture_packets,
#         'analyze_packets': analyze_packets,
        
#         # VSCode Extension
#         'create_vscode_extension': create_vscode_extension,
#         'add_vscode_command': add_vscode_command,
#         'register_vscode_listener': register_vscode_listener,
        
#         # Embedded Systems
#         'setup_arduino': setup_arduino,
#         'setup_raspberry_pi': setup_raspberry_pi,
#         'gpio_control': gpio_control,
        
#         # Low-Level Programming
#         'compile_c': compile_c,
#         'compile_cpp': compile_cpp,
#         'compile_rust': compile_rust,
        
#         # Server Technologies
#         'setup_nginx': setup_nginx,
#         'setup_apache': setup_apache,
#         'configure_web_server': configure_web_server,
        
#         # Blockchain
#         'deploy_contract': deploy_contract,
#         'create_nft': create_nft,
        
#         # Game Development
#         'create_unity_game': create_unity_game,
#         'create_unreal_game': create_unreal_game,
#         'add_game_feature': add_game_feature,
        
#         # Basic Functions (existing)
#         'len': len_func,
#         'max': max_func,
#         'min': min_func,
#         'len': len_func,
#         'max': max_func,
#         'min': min_func,
#         'sum': sum_func,
#         'abs': abs_func,
#         'round': round_func,
#         'type': type_func,
#         'int': int_func,
#         'float': float_func,
#         'str': str_func,
#         'bool': bool_func,
#         'list': list_func,
#         'tuple': tuple_func,
#         'set': set_func,
#         'dict': dict_func,
#         'range': range_func,
#         'enumerate': enumerate_func,
#         'zip': zip_func,
#         'map': map_func,
#         'filter': filter_func,
#         'reduce': reduce_func,
#         'sorted': sorted_func,
#         'reversed': reversed_func,
#         'any': any_func,
#         'all': all_func,
#         'chr': chr_func,
#         'ord': ord_func,
#         'bin': bin_func,
#         'oct': oct_func,
#         'hex': hex_func,
#         'id': id_func,
#         'isinstance': isinstance_func,
#         'issubclass': issubclass_func,
#         'callable': callable_func,
#         'getattr': getattr_func,
#         'setattr': setattr_func,
#         'hasattr': hasattr_func,
#         'delattr': delattr_func,
#         'open': open_func,
#         'input': input_func,
#         'print': print_func,
#         'upper': upper_func,
#         'lower': lower_func,
#         'capitalize': capitalize_func,
#         'title': title_func,
#         'strip': strip_func,
#         'split': split_func,
#         'join': join_func,
#         'replace': replace_func,
#         'startswith': startswith_func,
#         'endswith': endswith_func,
#         'find': find_func,
#         'count': count_func,
#         'isalpha': isalpha_func,
#         'isdigit': isdigit_func,
#         'isalnum': isalnum_func,
#         'islower': islower_func,
#         'isupper': isupper_func,
#         'append': append_func,
#         'extend': extend_func,
#         'insert': insert_func,
#         'remove': remove_func,
#         'pop': pop_func,
#         'clear': clear_func,
#         'index': index_func,
#         'reverse': reverse_func,
#         'copy': copy_func,
#         'deepcopy': deepcopy_func,
#         'keys': keys_func,
#         'values': values_func,
#         'items': items_func,
#         'get': get_func,
#         'update': update_func,
#         'sin': math_sin_func,
#         'cos': math_cos_func,
#         'tan': math_tan_func,
#         'sqrt': math_sqrt_func,
#         'log': math_log_func,
#         'exp': math_exp_func,
#         'floor': math_floor_func,
#         'ceil': math_ceil_func,
#         'randint': random_randint_func,
#         'choice': random_choice_func,
#         'shuffle': random_shuffle_func,
#         'now': datetime_now_func,
#         'date': datetime_date_func,
#         'time': datetime_time_func,
#         'json_dumps': json_dumps_func,
#         'json_loads': json_loads_func,
#         're_search': re_search_func,
#         're_match': re_match_func,
#         're_findall': re_findall_func,
#         're_sub': re_sub_func,
#         'counter': collections_counter_func,
#         'defaultdict': collections_defaultdict_func,
#         'permutations': itertools_permutations_func,
#         'combinations': itertools_combinations_func,
#         'mean': statistics_mean_func,
#         'median': statistics_median_func,
#         'mode': statistics_mode_func,
#         'stdev': statistics_stdev_func,
#         'urlopen': urllib_request_urlopen_func,
#         'xml_parse': xml_parse_func,
#         'csv_reader': csv_reader_func,
#         'csv_writer': csv_writer_func,
#         'sqlite_connect': sqlite3_connect_func,
#         'md5': hashlib_md5_func,
#         'sha256': hashlib_sha256_func,
#         'base64_encode': base64_encode_func,
#         'base64_decode': base64_decode_func,
#         'zlib_compress': zlib_compress_func,
#         'zlib_decompress': zlib_decompress_func,
#         'thread': threading_thread_func,
#         'process': multiprocessing_process_func,
#         'asyncio_run': asyncio_run_func,
#         'get_type_hints': typing_get_type_hints_func,
#         'tf_constant': tf_constant_func,
#         'tf_variable': tf_variable_func,
#         'tf_matmul': tf_matmul_func,
#         'np_array': np_array_func,
#         'np_zeros': np_zeros_func,
#         'np_ones': np_ones_func,
#         'isSreekuttyIdiot': isSreekuttyIdiot,
#         'isSaiIdiot': isSaiIdiot,
#         'add': add,
#         'pickRandom': pickRandom,
#         'calculator': calculator,
#         'hypotenuse': hypotenuse,
#         'coloredText': coloredText,
#         'displayColoredText': displayColoredText
#     }

#     def evaluate_expression(self, expression):
#         try:
#             if expression.startswith('[') and expression.endswith(']'):
#                 # Handle list creation and list comprehension
#                 if ' for ' in expression:
#                     return eval(f"[{expression[1:-1]}]", {"__builtins__": None}, self.variables)
#                 return [self.evaluate_expression(item.strip()) for item in expression[1:-1].split(',')]
#             elif expression.startswith('{') and expression.endswith('}'):
#                 # Handle dictionary creation
#                 items = expression[1:-1].split(',')
#                 return {k.strip(): self.evaluate_expression(v.strip()) for k, v in (item.split(':') for item in items)}
#             elif expression.startswith('lambda'):
#                 # Handle lambda functions
#                 parts = expression.split(':')
#                 args = parts[0].split()[1:]
#                 body = ':'.join(parts[1:]).strip()
#                 return lambda *a: self.evaluate_expression(body)
#             elif '(' in expression and ')' in expression:
#                 func_name, args = expression.split('(', 1)
#                 args = args.rsplit(')', 1)[0].split(',')
#                 args = [self.evaluate_expression(arg.strip()) for arg in args]
#                 func_name = func_name.strip()
#                 if func_name in self.builtin_functions:
#                     return self.builtin_functions[func_name](self, args)
#                 return self.execute_function(func_name, args)
#             else:
#                 # Handle strings with both single and double quotes
#                 if (expression.startswith('"') and expression.endswith('"')) or \
#                         (expression.startswith("'") and expression.endswith("'")):
#                     return expression[1:-1]  # Return the string without quotes
#                 return eval(expression, {"__builtins__": None}, self.variables)
#         except Exception as e:
#             raise Exception(f"Invalid expression: {expression}")

#     def get_func(self, prompt):
#         user_input = input(prompt)  # Get user input
#         return user_input

#     def parse_block(self):
#         block = []
#         self.current_line += 1
#         while self.current_line < len(self.lines):
#             line = self.lines[self.current_line].strip()
#             if not line.startswith("    "):  # End of block
#                 break
#             block.append(line[4:])  # Remove indentation
#             self.current_line += 1
#         return block

#     def execute_block(self, block):
#         for line in block:
#             self.parse_line(line)

#     def parse_condition(self):
#         """Parse if, elif, else blocks."""
#         while self.current_line < len(self.lines):
#             line = self.lines[self.current_line].strip()
#             if line.startswith("if "):
#                 condition = self.evaluate_expression(line[3:].strip())
#                 block = self.parse_block()
#                 if condition:
#                     return block
#             elif line.startswith("elif "):
#                 condition = self.evaluate_expression(line[5:].strip())
#                 block = self.parse_block()
#                 if condition:
#                     return block
#             elif line == "else":
#                 block = self.parse_block()
#                 return block
#             else:
#                 break
#         return []

#     def parse_while(self):
#         condition = self.evaluate_expression(self.lines[self.current_line][6:].strip())
#         block = self.parse_block()
#         while condition:
#             self.execute_block(block)
#             condition = self.evaluate_expression(self.lines[self.current_line][6:].strip())

#     def parse_for(self):
#         var_name, range_expr = self.lines[self.current_line][4:].split(" in ")
#         var_name = var_name.strip()
#         range_expr = range_expr.strip()
#         block = self.parse_block()
#         for i in eval(range_expr, {}, self.variables):
#             self.variables[var_name] = i
#             self.execute_block(block)

#     def parse_function(self):
#         function_def = self.lines[self.current_line][4:].strip()
#         func_name, args = function_def.split("(")
#         func_name = func_name.strip()
#         args = args.replace(")", "").strip().split(",")
#         block = self.parse_block()
#         self.functions[func_name] = (args, block)

#     def execute_function(self, func_name, args_values):
#         if func_name not in self.functions:
#             raise Exception(f"Unknown function: {func_name}")
#         arg_names, block = self.functions[func_name]
#         if len(arg_names) != len(args_values):
#             raise Exception(
#                 f"Function {func_name} expects {len(arg_names)} arguments, but {len(args_values)} were provided")
#         # Save the current variables and functions context
#         original_variables = self.variables.copy()
#         original_functions = self.functions.copy()
#         try:
#             # Set the function arguments in the variables context
#             for i, arg in enumerate(arg_names):
#                 self.variables[arg] = args_values[i]
#             # Execute the function block
#             return_value = None
#             for line in block:
#                 if line.startswith("return "):
#                     return_value = self.evaluate_expression(line[7:].strip())
#                     break
#                 else:
#                     self.parse_line(line)
#             return return_value
#         finally:
#             # Restore the original variables and functions context
#             self.variables = original_variables
#             self.functions = original_functions

#     def parse_class(self):
#         class_def = self.lines[self.current_line][6:].strip()
#         class_name = class_def.split('(')[0].strip()
#         block = self.parse_block()
#         class_dict = {}
#         for line in block:
#             if line.startswith('def '):
#                 func_name = line[4:].split('(')[0].strip()
#                 args = line.split('(')[1].split(')')[0].split(',')
#                 args = [arg.strip() for arg in args]
#                 method_block = self.parse_block()
#                 class_dict[func_name] = (args, method_block)
#         self.classes[class_name] = class_dict

#     def create_object(self, class_name, *args):
#         if class_name not in self.classes:
#             raise Exception(f"Unknown class: {class_name}")
#         class_dict = self.classes[class_name]
#         obj = {'__class__': class_name}
#         if '__init__' in class_dict:
#             init_args, init_block = class_dict['__init__']
#             self.execute_method(obj, '__init__', init_args, init_block, args)
#         return obj

#     def execute_method(self, obj, method_name, args, block, arg_values):
#         original_variables = self.variables.copy()
#         try:
#             self.variables['self'] = obj
#             for i, arg in enumerate(args[1:]):  # Skip 'self'
#                 self.variables[arg] = arg_values[i]
#             self.execute_block(block)
#         finally:
#             self.variables = original_variables

#     def parse_import(self):
#         import_statement = self.lines[self.current_line][7:].strip()
#         module_name = import_statement.split(' as ')[0] if ' as ' in import_statement else import_statement
#         alias = import_statement.split(' as ')[1] if ' as ' in import_statement else module_name
#         try:
#             module = importlib.import_module(module_name)
#             self.variables[alias] = module
#         except ImportError:
#             raise Exception(f"Unable to import module: {module_name}")

#     def parse_with(self):
#         with_statement = self.lines[self.current_line][5:].strip()
#         context_expr, var_name = with_statement.split(' as ')
#         context_manager = self.evaluate_expression(context_expr)
#         block = self.parse_block()
#         with context_manager as cm:
#             self.variables[var_name.strip()] = cm
#             self.execute_block(block)

#     def parse_decorator(self):
#         decorator_name = self.lines[self.current_line][1:].strip()
#         self.current_line += 1
#         function_def = self.lines[self.current_line][4:].strip()
#         func_name, args = function_def.split("(")
#         func_name = func_name.strip()
#         args = args.replace(")", "").strip().split(",")
#         block = self.parse_block()
#         decorator = self.evaluate_expression(decorator_name)
#         decorated_func = decorator(lambda *args: self.execute_block(block))
#         self.functions[func_name] = (args, decorated_func)

#     def parse_line(self, line):
#         line = line.strip()

#         try:
#             if line.startswith('"""'):
#                 # Handle multi-line comment
#                 while not line.endswith('"""'):
#                     self.current_line += 1
#                     if self.current_line >= len(self.lines):
#                         raise Exception("Unterminated multi-line comment")
#                     line += self.lines[self.current_line].strip()
#                 return  # Ignore multi-line comments

#             elif line.startswith("display "):
#                 content = line[8:].strip()
#                 if content.startswith('"') and content.endswith('"') or \
#                         content.startswith("'") and content.endswith("'"):
#                     print(content[1:-1])
#                 elif content in self.variables:
#                     print(self.variables[content])
#                 else:
#                     print(self.evaluate_expression(content))

#             elif line.startswith("get "):
#                 parts = line[4:].strip().split('"', 1)  # Split on the first quotation mark
#                 if len(parts) == 2 and (parts[1].endswith('"') or parts[1].endswith("'")):
#                     var_name = parts[0].strip()  # Variable name before prompt
#                     prompt = parts[1][:-1]  # Remove trailing quote
#                     user_input = self.get_func(prompt)  # Get user input
#                     self.variables[var_name] = user_input  # Store user input in the variable
#                 else:
#                     raise Exception(f"Invalid get statement: {line}")

#             elif "=" in line:  # Handle variable assignment without 'let'
#                 parts = line.split("=")
#                 if len(parts) == 2:
#                     var_name = parts[0].strip()
#                     var_value = self.evaluate_expression(parts[1].strip())
#                     self.variables[var_name] = var_value
#                 else:
#                     raise Exception(f"Invalid assignment statement: {line}")

#             elif line.startswith("if "):
#                 block = self.parse_condition()
#                 self.execute_block(block)

#             elif line.startswith("while "):
#                 self.parse_while()

#             elif line.startswith("for "):
#                 self.parse_for()

#             elif line.startswith("def "):
#                 self.parse_function()

#             elif line.startswith("class "):
#                 self.parse_class()

#             elif line.startswith("import "):
#                 self.parse_import()

#             elif line.startswith("with "):
#                 self.parse_with()

#             elif line.startswith("@"):
#                 self.parse_decorator()

#             elif line.startswith("return "):
#                 # This will be handled in the execute_function method
#                 pass

#             elif line == "" or line.startswith("#"):
#                 pass

#             elif line == "help":
#                 self.display_help()

#             elif line == "about":
#                 self.display_about()

#             elif line.startswith("shell ") or line == "shell" or line == 'pl shell':
#                 self.display_Warning()

#             elif line in devCommands:  # If the command is in the devCommands list then, greet the developer
#                 self.greet_developer()

#             else:
#                 result = self.evaluate_expression(line)
#                 if result is not None:
#                     print(result)

#         except Exception as e:
#             raise Exception(f"Error on line {self.current_line + 1}: {str(e)}")

#     def run(self):
#         if self.interactive:
#             print(colored("Welcome to the Orion interactive shell. Orion Version 2.0.0",'red'))
#             logo = """
#             {cyan}██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗{reset}    
#             {cyan}██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║{reset}    
#             {cyan}██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║{reset}    
#             {cyan}██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║{reset}    
#             {cyan}╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║{reset}    
#             {cyan}╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝{reset}    
            
#             {magenta}The celestial hunter of the night sky{reset}    
            
#                               {green}-Amphibiar{reset}              
#             """

#             # Define color codes
#             yellow = '\033[93m'
#             cyan = '\033[96m'
#             magenta = '\033[95m'
#             reset = '\033[0m'
#             green = '\033[32m'

#             # Replace color placeholders
#             logo = logo.format(yellow=yellow, cyan=cyan, magenta=magenta, reset=reset, green=green)

#             print(logo)
#             print(colored("Type 'help' for help and 'about' for information.",'yellow'))
#             while True:
#                 try:
#                     line = input(colored("$ ",'blue'))
#                     if line == "exit":
#                         break
#                     self.parse_line(line)
#                 except Exception as e:
#                     print(f"Error: {e}")
#         else:
#             if self.filename:
#                 try:
#                     with open(self.filename, 'r') as file:
#                         self.lines = file.readlines()
#                     while self.current_line < len(self.lines):
#                         try:
#                             self.parse_line(self.lines[self.current_line])
#                         except Exception as e:
#                             print(f"Error on line {self.current_line + 1}: {str(e)}")
#                             break
#                         self.current_line += 1
#                 except FileNotFoundError:
#                     print(f"Error: File '{self.filename}' not found.")
#                 except Exception as e:
#                     print(f"Error: {str(e)}")

#     def display_help(self):
#         help_text = """
# PL Language Help:

# Basic Syntax:
# - display: Output something, e.g., display 'Hello'
# - get: Get input from the user, e.g., get var 'Enter your name'
# - Variables: Use = for assignment, e.g., x = 10
# - Functions: Define functions using def, e.g., def my_function(x): ...
# - Classes: Define classes using class, e.g., class MyClass: ...
# - Control structures: Use if, elif, else, while, for
# - Imports: Import modules using import, e.g., import math
# - List comprehensions: [x for x in range(10) if x % 2 == 0]
# - Lambda functions: lambda x: x * 2
# - Decorators: Use @ symbol, e.g., @my_decorator
# - Context managers: Use with statement, e.g., with open('file.txt', 'r') as f: ...

# Built-in Functions:
# - Math: abs, round, sum, max, min, sin, cos, tan, sqrt, log, exp, floor, ceil
# - Type conversion: int, float, str, bool, list, tuple, set, dict
# - Sequences: len, range, enumerate, zip, map, filter, reduce, sorted, reversed
# - String operations: upper, lower, capitalize, title, strip, split, join, replace
# - List operations: append, extend, insert, remove, pop, clear, index, reverse, copy
# - Dictionary operations: keys, values, items, get, update
# - File operations: open, read, write
# - Random: randint, choice, shuffle
# - Date and time: now, date, time
# - JSON: json_dumps, json_loads
# - Regular expressions: re_search, re_match, re_findall, re_sub
# - Collections: counter, defaultdict
# - Itertools: permutations, combinations
# - Statistics: mean, median, mode, stdev
# - Web: urlopen
# - XML: xml_parse
# - CSV: csv_reader, csv_writer
# - Database: sqlite_connect
# - Cryptography: md5, sha256, base64_encode, base64_decode
# - Compression: zlib_compress, zlib_decompress
# - Concurrency: thread, process, asyncio_run

# TensorFlow and NumPy (if available):
# - tf_constant, tf_variable, tf_matmul
# - np_array, np_zeros, np_ones

# Custom Functions:
# - isSreekuttyIdiot, isSaiIdiot, add, pickRandom, calculator

# Type 'exit' to quit the interactive shell.
# """
#         print(help_text)

#     def greet_developer(self):
#         print("Welcome, developer")

#     def display_Warning(self):
#         print('Already in shell')

#     def display_about(self):
#         print("Orion Interpreter v2.0.0")
#         print(random.choice(printOptions))
#         print("")

# if __name__ == "__main__":
#     if len(sys.argv) == 1:
#         print("Usage: orion run <filename.or> or orion shell for interactive mode")
#     elif len(sys.argv) == 2 and sys.argv[1] == "shell":
#         interpreter = OrionInterpreter(interactive=True)
#         interpreter.run()
#     elif len(sys.argv) == 3 and sys.argv[1] == "run":
#         filename = sys.argv[2]
#         interpreter = OrionInterpreter(filename=filename)
#         interpreter.run()
#     else:
#         print("Usage: orion run <filename.or> or orion shell for interactive mode")





import sys
import operator
import os
import importlib
import functools
import random
import math
import datetime
import json
import re
import collections
import itertools
import statistics
import urllib.request
import xml.etree.ElementTree as ET
import csv
import sqlite3
import hashlib
import base64
import zlib
import threading
import multiprocessing
import asyncio
import typing
from termcolor import colored
import math


try:
    import numpy as np
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

printOptions = ["Created by Pranav Lejith (Amphibiar)", """Created by Pranav "Amphibiar" Lejith"""]
devCommands = ['amphibiar', 'developer', 'command override-amphibiar', 'emergency override-amphibiar']

class OrionInterpreter:
    def __init__(self, filename=None, interactive=False):
        self.filename = filename
        self.variables = {}
        self.functions = {}
        self.classes = {}
        self.lines = []
        self.current_line = 0
        self.interactive = interactive
        self.ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '%': operator.mod,
            '**': operator.pow,
            '//': operator.floordiv,
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '<=': operator.le,
            '>': operator.gt,
            '>=': operator.ge,
            'and': operator.and_,
            'or': operator.or_,
            'not': operator.not_,
        }

    # Built-in functions
    def len_func(self, args):
        return len(args[0])

    def max_func(self, args):
        return max(args[0])

    def min_func(self, args):

        return min(args[0])

    def sum_func(self, args):
        return sum(args[0])

    def abs_func(self, args):
        return abs(args[0])

    def round_func(self, args):
        return round(*args)

    def type_func(self, args):
        return type(args[0]).__name__

    def int_func(self, args):
        return int(args[0])

    def float_func(self, args):
        return float(args[0])

    def str_func(self, args):
        return str(args[0])

    def bool_func(self, args):
        return bool(args[0])

    def list_func(self, args):
        return list(args[0])

    def tuple_func(self, args):
        return tuple(args[0])

    def set_func(self, args):
        return set(args[0])

    def dict_func(self, args):
        return dict(args[0])

    def range_func(self, args):
        return list(range(*args))

    def enumerate_func(self, args):
        return list(enumerate(*args))

    def zip_func(self, args):
        return list(zip(*args))

    def map_func(self, args):
        return list(map(*args))

    def filter_func(self, args):
        return list(filter(*args))

    def reduce_func(self, args):
        return functools.reduce(*args)

    def sorted_func(self, args):
        return sorted(*args)

    def reversed_func(self, args):
        return list(reversed(args[0]))

    def any_func(self, args):
        return any(args[0])

    def all_func(self, args):
        return all(args[0])

    def chr_func(self, args):
        return chr(args[0])

    def ord_func(self, args):
        return ord(args[0])

    def bin_func(self, args):
        return bin(args[0])

    def oct_func(self, args):
        return oct(args[0])

    def hex_func(self, args):
        return hex(args[0])

    def id_func(self, args):
        return id(args[0])

    def isinstance_func(self, args):
        return isinstance(args[0], args[1])

    def issubclass_func(self, args):
        return issubclass(args[0], args[1])

    def callable_func(self, args):
        return callable(args[0])

    def getattr_func(self, args):
        return getattr(*args)

    def setattr_func(self, args):
        setattr(*args)

    def hasattr_func(self, args):
        return hasattr(*args)

    def delattr_func(self, args):
        delattr(*args)

    def open_func(self, args):
        return open(*args)

    def input_func(self, args):
        return input(*args)

    def print_func(self, args):
        print(*args)

    def len_func(self, args):
        return len(args[0])

    def upper_func(self, args):
        return args[0].upper()

    def lower_func(self, args):
        return args[0].lower()

    def capitalize_func(self, args):
        return args[0].capitalize()

    def title_func(self, args):
        return args[0].title()

    def strip_func(self, args):
        return args[0].strip()

    def split_func(self, args):
        return args[0].split(*args[1:])

    def join_func(self, args):
        return args[0].join(args[1])

    def replace_func(self, args):
        return args[0].replace(*args[1:])

    def startswith_func(self, args):
        return args[0].startswith(args[1])

    def endswith_func(self, args):
        return args[0].endswith(args[1])

    def find_func(self, args):
        return args[0].find(*args[1:])

    def count_func(self, args):
        return args[0].count(args[1])

    def isalpha_func(self, args):
        return args[0].isalpha()

    def isdigit_func(self, args):
        return args[0].isdigit()

    def isalnum_func(self, args):
        return args[0].isalnum()

    def islower_func(self, args):
        return args[0].islower()

    def isupper_func(self, args):
        return args[0].isupper()

    def append_func(self, args):
        args[0].append(args[1])

    def extend_func(self, args):
        args[0].extend(args[1])

    def insert_func(self, args):
        args[0].insert(args[1], args[2])

    def remove_func(self, args):
        args[0].remove(args[1])

    def pop_func(self, args):
        return args[0].pop(*args[1:])

    def clear_func(self, args):
        args[0].clear()

    def index_func(self, args):
        return args[0].index(*args[1:])

    def reverse_func(self, args):
        args[0].reverse()

    def copy_func(self, args):
        return args[0].copy()

    def deepcopy_func(self, args):
        import copy
        return copy.deepcopy(args[0])

    def keys_func(self, args):
        return list(args[0].keys())

    def values_func(self, args):
        return list(args[0].values())

    def items_func(self, args):
        return list(args[0].items())

    def get_func(self, args):
        return args[0].get(*args[1:])

    def update_func(self, args):
        args[0].update(args[1])

    def math_sin_func(self, args):
        return math.sin(args[0])

    def math_cos_func(self, args):
        return math.cos(args[0])

    def math_tan_func(self, args):
        return math.tan(args[0])

    def math_sqrt_func(self, args):
        return math.sqrt(args[0])

    def math_log_func(self, args):
        return math.log(*args)

    def math_exp_func(self, args):
        return math.exp(args[0])

    def math_floor_func(self, args):
        return math.floor(args[0])

    def math_ceil_func(self, args):
        return math.ceil(args[0])

    def random_randint_func(self, args):
        return random.randint(*args)

    def random_choice_func(self, args):
        return random.choice(args[0])

    def random_shuffle_func(self, args):
        random.shuffle(args[0])

    def datetime_now_func(self, args):
        return datetime.datetime.now()

    def datetime_date_func(self, args):
        return datetime.date(*args)

    def datetime_time_func(self, args):
        return datetime.time(*args)

    def json_dumps_func(self, args):
        return json.dumps(*args)

    def json_loads_func(self, args):
        return json.loads(*args)

    def re_search_func(self, args):
        return re.search(*args)

    def re_match_func(self, args):
        return re.match(*args)

    def re_findall_func(self, args):
        return re.findall(*args)

    def re_sub_func(self, args):
        return re.sub(*args)

    def collections_counter_func(self, args):
        return collections.Counter(args[0])

    def collections_defaultdict_func(self, args):
        return collections.defaultdict(args[0])

    def itertools_permutations_func(self, args):
        return list(itertools.permutations(*args))

    def itertools_combinations_func(self, args):
        return list(itertools.combinations(*args))

    def statistics_mean_func(self, args):
        return statistics.mean(args[0])

    def statistics_median_func(self, args):
        return statistics.median(args[0])

    def statistics_mode_func(self, args):
        return statistics.mode(args[0])

    def statistics_stdev_func(self, args):
        return statistics.stdev(args[0])

    def urllib_request_urlopen_func(self, args):
        return urllib.request.urlopen(*args)

    def xml_parse_func(self, args):
        return ET.parse(*args)

    def csv_reader_func(self, args):
        return csv.reader(*args)

    def csv_writer_func(self, args):
        return csv.writer(*args)

    def sqlite3_connect_func(self, args):
        return sqlite3.connect(*args)

    def hashlib_md5_func(self, args):
        return hashlib.md5(args[0].encode()).hexdigest()

    def hashlib_sha256_func(self, args):
        return hashlib.sha256(args[0].encode()).hexdigest()

    def base64_encode_func(self, args):
        return base64.b64encode(args[0].encode()).decode()

    def base64_decode_func(self, args):
        return base64.b64decode(args[0]).decode()

    def zlib_compress_func(self, args):
        return zlib.compress(args[0].encode())

    def zlib_decompress_func(self, args):
        return zlib.decompress(args[0]).decode()

    def threading_thread_func(self, args):
        return threading.Thread(*args)

    def multiprocessing_process_func(self, args):
        return multiprocessing.Process(*args)

    def asyncio_run_func(self, args):
        return asyncio.run(*args)

    def typing_get_type_hints_func(self, args):
        return typing.get_type_hints(*args)

    # TensorFlow functions (if available)
    def tf_constant_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return tf.constant(*args)
        else:
            raise Exception("TensorFlow is not available")

    def tf_variable_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return tf.Variable(*args)
        else:
            raise Exception("TensorFlow is not available")

    def tf_matmul_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return tf.matmul(*args)
        else:
            raise Exception("TensorFlow is not available")

    def np_array_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return np.array(*args)
        else:
            raise Exception("NumPy is not available")

    def np_zeros_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return np.zeros(*args)
        else:
            raise Exception("NumPy is not available")

    def np_ones_func(self, args):
        if TENSORFLOW_AVAILABLE:
            return np.ones(*args)
        else:
            raise Exception("NumPy is not available")

    # Custom functions
    def isSreekuttyIdiot(self, args):
        if len(args) != 1:
            raise ValueError("isSreekuttyIdiot() takes exactly one argument")
        arg = args[0]
        if arg not in [0, 1]:
            raise ValueError("isSreekuttyIdiot() argument must be either 0 or 1")
        return "Yes, Sreekutty is an idiot" if arg == 1 else "No, she is not an idiot"

    def isSaiIdiot(self, args):
        if len(args) != 1:
            raise ValueError("isSaiIdiot() takes exactly one argument")
        arg = args[0]
        if arg not in [0, 1]:
            raise ValueError("isSaiIdiot() argument must be either 0 or 1")
        return "Yes, Sai is an idiot" if arg == 1 else "No, Sai is not an idiot"

    def add(self, args):
        if len(args) != 2:
            raise ValueError("add() takes 2 arguments only")
        return args[0] + args[1]

    def pickRandom(self, args):
        return random.choice(args)

    def calculator(self, args):
        if len(args) != 3:
            raise ValueError("calculator() takes 3 arguments: two numbers and an operation")
        n1, n2, op = args
        if op == 'add':
            return n1 + n2
        elif op == 'sub':
            return n1 - n2
        elif op == 'div':
            return n1 / n2
        elif op == 'mul':
            return n1 * n2
        else:
            raise ValueError('Supported operations are add, sub, mul, div')

    def hypotenuse(self,args):
        if len(args) != 2:
            raise ValueError("hypotenuse() takes 2 arguments (side1,side2)")
        return math.sqrt((args[0]*args[0])+(args[1]*args[1]))

    def coloredText(self,args):
        if len(args) != 2:
            raise ValueError("coloredText() taks 2 arguments: text, color")
        return colored(args[0],args[1])

    def displayColoredText(self,args):
        if len(args) != 2:
            raise ValueError("displayColoredText() taks 2 arguments: text, color")
        print(colored(args[0],args[1]))
    # Register built-in functions
    builtin_functions = {
        'len': len_func,
        'max': max_func,
        'min': min_func,
        'sum': sum_func,
        'abs': abs_func,
        'round': round_func,
        'type': type_func,
        'int': int_func,
        'float': float_func,
        'str': str_func,
        'bool': bool_func,
        'list': list_func,
        'tuple': tuple_func,
        'set': set_func,
        'dict': dict_func,
        'range': range_func,
        'enumerate': enumerate_func,
        'zip': zip_func,
        'map': map_func,
        'filter': filter_func,
        'reduce': reduce_func,
        'sorted': sorted_func,
        'reversed': reversed_func,
        'any': any_func,
        'all': all_func,
        'chr': chr_func,
        'ord': ord_func,
        'bin': bin_func,
        'oct': oct_func,
        'hex': hex_func,
        'id': id_func,
        'isinstance': isinstance_func,
        'issubclass': issubclass_func,
        'callable': callable_func,
        'getattr': getattr_func,
        'setattr': setattr_func,
        'hasattr': hasattr_func,
        'delattr': delattr_func,
        'open': open_func,
        'input': input_func,
        'print': print_func,
        'upper': upper_func,
        'lower': lower_func,
        'capitalize': capitalize_func,
        'title': title_func,
        'strip': strip_func,
        'split': split_func,
        'join': join_func,
        'replace': replace_func,
        'startswith': startswith_func,
        'endswith': endswith_func,
        'find': find_func,
        'count': count_func,
        'isalpha': isalpha_func,
        'isdigit': isdigit_func,
        'isalnum': isalnum_func,
        'islower': islower_func,
        'isupper': isupper_func,
        'append': append_func,
        'extend': extend_func,
        'insert': insert_func,
        'remove': remove_func,
        'pop': pop_func,
        'clear': clear_func,
        'index': index_func,
        'reverse': reverse_func,
        'copy': copy_func,
        'deepcopy': deepcopy_func,
        'keys': keys_func,
        'values': values_func,
        'items': items_func,
        'get': get_func,
        'update': update_func,
        'sin': math_sin_func,
        'cos': math_cos_func,
        'tan': math_tan_func,
        'sqrt': math_sqrt_func,
        'log': math_log_func,
        'exp': math_exp_func,
        'floor': math_floor_func,
        'ceil': math_ceil_func,
        'randint': random_randint_func,
        'choice': random_choice_func,
        'shuffle': random_shuffle_func,
        'now': datetime_now_func,
        'date': datetime_date_func,
        'time': datetime_time_func,
        'json_dumps': json_dumps_func,
        'json_loads': json_loads_func,
        're_search': re_search_func,
        're_match': re_match_func,
        're_findall': re_findall_func,
        're_sub': re_sub_func,
        'counter': collections_counter_func,
        'defaultdict': collections_defaultdict_func,
        'permutations': itertools_permutations_func,
        'combinations': itertools_combinations_func,
        'mean': statistics_mean_func,
        'median': statistics_median_func,
        'mode': statistics_mode_func,
        'stdev': statistics_stdev_func,
        'urlopen': urllib_request_urlopen_func,
        'xml_parse': xml_parse_func,
        'csv_reader': csv_reader_func,
        'csv_writer': csv_writer_func,
        'sqlite_connect': sqlite3_connect_func,
        'md5': hashlib_md5_func,
        'sha256': hashlib_sha256_func,
        'base64_encode': base64_encode_func,
        'base64_decode': base64_decode_func,
        'zlib_compress': zlib_compress_func,
        'zlib_decompress': zlib_decompress_func,
        'thread': threading_thread_func,
        'process': multiprocessing_process_func,
        'asyncio_run': asyncio_run_func,
        'get_type_hints': typing_get_type_hints_func,
        'tf_constant': tf_constant_func,
        'tf_variable': tf_variable_func,
        'tf_matmul': tf_matmul_func,
        'np_array': np_array_func,
        'np_zeros': np_zeros_func,
        'np_ones': np_ones_func,
        'isSreekuttyIdiot': isSreekuttyIdiot,
        'isSaiIdiot': isSaiIdiot,
        'add': add,
        'pickRandom': pickRandom,
        'calculator': calculator,
        'hypotenuse': hypotenuse,
        'coloredText': coloredText,
        'displayColoredText': displayColoredText
    }

    def evaluate_expression(self, expression):
        try:
            if expression.startswith('[') and expression.endswith(']'):
                # Handle list creation and list comprehension
                if ' for ' in expression:
                    return eval(f"[{expression[1:-1]}]", {"__builtins__": None}, self.variables)
                return [self.evaluate_expression(item.strip()) for item in expression[1:-1].split(',')]
            elif expression.startswith('{') and expression.endswith('}'):
                # Handle dictionary creation
                items = expression[1:-1].split(',')
                return {k.strip(): self.evaluate_expression(v.strip()) for k, v in (item.split(':') for item in items)}
            elif expression.startswith('lambda'):
                # Handle lambda functions
                parts = expression.split(':')
                args = parts[0].split()[1:]
                body = ':'.join(parts[1:]).strip()
                return lambda *a: self.evaluate_expression(body)
            elif '(' in expression and ')' in expression:
                func_name, args = expression.split('(', 1)
                args = args.rsplit(')', 1)[0].split(',')
                args = [self.evaluate_expression(arg.strip()) for arg in args]
                func_name = func_name.strip()
                if func_name in self.builtin_functions:
                    return self.builtin_functions[func_name](self, args)
                return self.execute_function(func_name, args)
            else:
                # Handle strings with both single and double quotes
                if (expression.startswith('"') and expression.endswith('"')) or \
                        (expression.startswith("'") and expression.endswith("'")):
                    return expression[1:-1]  # Return the string without quotes
                return eval(expression, {"__builtins__": None, **self.ops}, self.variables)
        except Exception as e:
            raise Exception(f"Invalid expression: {expression}")

    def get_func(self, prompt):
        user_input = input(prompt)  # Get user input
        return user_input

    def parse_block(self):
        block = []
        self.current_line += 1
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].strip()
            if not line.startswith("    "):  # End of block
                break
            block.append(line[4:])  # Remove indentation
            self.current_line += 1
        return block

    def execute_block(self, block):
        for line in block:
            self.parse_line(line)

    def parse_condition(self):
        """Parse if, elif, else blocks."""
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].strip()
            if line.startswith("if "):
                condition = self.evaluate_expression(line[3:].strip())
                block = self.parse_block()
                if condition:
                    return block
            elif line.startswith("elif "):
                condition = self.evaluate_expression(line[5:].strip())
                block = self.parse_block()
                if condition:
                    return block
            elif line == "else":
                block = self.parse_block()
                return block
            else:
                break
        return []

    def parse_while(self):
        condition = self.evaluate_expression(self.lines[self.current_line][6:].strip())
        block = self.parse_block()
        while condition:
            self.execute_block(block)
            condition = self.evaluate_expression(self.lines[self.current_line][6:].strip())

    def parse_for(self):
        var_name, range_expr = self.lines[self.current_line][4:].split(" in ")
        var_name = var_name.strip()
        range_expr = range_expr.strip()
        block = self.parse_block()
        for i in eval(range_expr, {}, self.variables):
            self.variables[var_name] = i
            self.execute_block(block)

    def parse_function(self):
        function_def = self.lines[self.current_line][4:].strip()
        func_name, args = function_def.split("(")
        func_name = func_name.strip()
        args = args.replace(")", "").strip().split(",")
        block = self.parse_block()
        self.functions[func_name] = (args, block)

    def execute_function(self, func_name, args_values):
        if func_name not in self.functions:
            raise Exception(f"Unknown function: {func_name}")
        arg_names, block = self.functions[func_name]
        if len(arg_names) != len(args_values):
            raise Exception(
                f"Function {func_name} expects {len(arg_names)} arguments, but {len(args_values)} were provided")
        # Save the current variables and functions context
        original_variables = self.variables.copy()
        original_functions = self.functions.copy()
        try:
            # Set the function arguments in the variables context
            for i, arg in enumerate(arg_names):
                self.variables[arg] = args_values[i]
            # Execute the function block
            return_value = None
            for line in block:
                if line.startswith("return "):
                    return_value = self.evaluate_expression(line[7:].strip())
                    break
                else:
                    self.parse_line(line)
            return return_value
        finally:
            # Restore the original variables and functions context
            self.variables = original_variables
            self.functions = original_functions

    def parse_class(self):
        class_def = self.lines[self.current_line][6:].strip()
        class_name = class_def.split('(')[0].strip()
        block = self.parse_block()
        class_dict = {}
        for line in block:
            if line.startswith('def '):
                func_name = line[4:].split('(')[0].strip()
                args = line.split('(')[1].split(')')[0].split(',')
                args = [arg.strip() for arg in args]
                method_block = self.parse_block()
                class_dict[func_name] = (args, method_block)
        self.classes[class_name] = class_dict

    def create_object(self, class_name, *args):
        if class_name not in self.classes:
            raise Exception(f"Unknown class: {class_name}")
        class_dict = self.classes[class_name]
        obj = {'__class__': class_name}
        if '__init__' in class_dict:
            init_args, init_block = class_dict['__init__']
            self.execute_method(obj, '__init__', init_args, init_block, args)
        return obj

    def execute_method(self, obj, method_name, args, block, arg_values):
        original_variables = self.variables.copy()
        try:
            self.variables['self'] = obj
            for i, arg in enumerate(args[1:]):  # Skip 'self'
                self.variables[arg] = arg_values[i]
            self.execute_block(block)
        finally:
            self.variables = original_variables

    def parse_import(self):
        import_statement = self.lines[self.current_line][7:].strip()
        module_name = import_statement.split(' as ')[0] if ' as ' in import_statement else import_statement
        alias = import_statement.split(' as ')[1] if ' as ' in import_statement else module_name
        try:
            module = importlib.import_module(module_name)
            self.variables[alias] = module
        except ImportError:
            raise Exception(f"Unable to import module: {module_name}")

    def parse_with(self):
        with_statement = self.lines[self.current_line][5:].strip()
        context_expr, var_name = with_statement.split(' as ')
        context_manager = self.evaluate_expression(context_expr)
        block = self.parse_block()
        with context_manager as cm:
            self.variables[var_name.strip()] = cm
            self.execute_block(block)

    def parse_decorator(self):
        decorator_name = self.lines[self.current_line][1:].strip()
        self.current_line += 1
        function_def = self.lines[self.current_line][4:].strip()
        func_name, args = function_def.split("(")
        func_name = func_name.strip()
        args = args.replace(")", "").strip().split(",")
        block = self.parse_block()
        decorator = self.evaluate_expression(decorator_name)
        decorated_func = decorator(lambda *args: self.execute_block(block))
        self.functions[func_name] = (args, decorated_func)

    def parse_line(self, line):
        line = line.strip()

        try:
            if line.startswith('"""'):
                # Handle multi-line comment
                while not line.endswith('"""'):
                    self.current_line += 1
                    if self.current_line >= len(self.lines):
                        raise Exception("Unterminated multi-line comment")
                    line += self.lines[self.current_line].strip()
                return  # Ignore multi-line comments

            elif line.startswith("display "):
                content = line[8:].strip()
                if content.startswith('"') and content.endswith('"') or \
                        content.startswith("'") and content.endswith("'"):
                    print(content[1:-1])
                elif content in self.variables:
                    print(self.variables[content])
                else:
                    print(self.evaluate_expression(content))

            elif line.startswith("get "):
                parts = line[4:].strip().split('"', 1)  # Split on the first quotation mark
                if len(parts) == 2 and (parts[1].endswith('"') or parts[1].endswith("'")):
                    var_name = parts[0].strip()  # Variable name before prompt
                    prompt = parts[1][:-1]  # Remove trailing quote
                    user_input = self.get_func(prompt)  # Get user input
                    self.variables[var_name] = user_input  # Store user input in the variable
                else:
                    raise Exception(f"Invalid get statement: {line}")

            elif (
                "=" in line
                and not line.startswith("if ")
                and not line.startswith("elif ")
                and not line.startswith("while ")
                and not line.startswith("for ")
                and not line.startswith("def ")
                and not line.startswith("class ")
                and not line.startswith("return ")
            ):
                parts = line.split("=")
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_value = self.evaluate_expression(parts[1].strip())
                    self.variables[var_name] = var_value
                else:
                    raise Exception(f"Invalid assignment statement: {line}")

            elif line.startswith("if "):
                block = self.parse_condition()
                self.execute_block(block)

            elif line.startswith("while "):
                self.parse_while()

            elif line.startswith("for "):
                self.parse_for()

            elif line.startswith("def "):
                self.parse_function()

            elif line.startswith("class "):
                self.parse_class()

            elif line.startswith("import "):
                self.parse_import()

            elif line.startswith("with "):
                self.parse_with()

            elif line.startswith("@"):
                self.parse_decorator()

            elif line.startswith("return "):
                # This will be handled in the execute_function method
                pass

            elif line == "" or line.startswith("#"):
                pass

            elif line == "help":
                self.display_help()

            elif line == "about":
                self.display_about()

            elif line.startswith("shell ") or line == "shell" or line == 'pl shell':
                self.display_Warning()

            elif line in devCommands:  # If the command is in the devCommands list then, greet the developer
                self.greet_developer()

            else:
                result = self.evaluate_expression(line)
                if result is not None:
                    print(result)

        except Exception as e:
            raise Exception(f"Error on line {self.current_line + 1}: {str(e)}")

    def run(self):
        if self.interactive:
            print(colored("Welcome to the Orion interactive shell. Orion Version 2.0.0",'red'))
            logo = """
            {cyan}██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗{reset}    
            {cyan}██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║{reset}    
            {cyan}██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║{reset}    
            {cyan}██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║{reset}    
            {cyan}╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║{reset}    
            {cyan}╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝{reset}    
            
            {magenta}The celestial hunter of the night sky{reset}    
            
                              {green}-Amphibiar{reset}              
            """

            # Define color codes
            yellow = '\033[93m'
            cyan = '\033[96m'
            magenta = '\033[95m'
            reset = '\033[0m'
            green = '\033[32m'

            # Replace color placeholders
            logo = logo.format(yellow=yellow, cyan=cyan, magenta=magenta, reset=reset, green=green)

            print(logo)
            print(colored("Type 'help' for help and 'about' for information.",'yellow'))
            while True:
                try:
                    line = input(colored("$ ",'blue'))
                    if line == "exit":
                        break
                    self.parse_line(line)
                except Exception as e:
                    print(f"Error: {e}")
        else:
            if self.filename:
                try:
                    with open(self.filename, 'r') as file:
                        self.lines = file.readlines()
                    while self.current_line < len(self.lines):
                        try:
                            self.parse_line(self.lines[self.current_line])
                        except Exception as e:
                            print(f"Error on line {self.current_line + 1}: {str(e)}")
                            break
                        self.current_line += 1
                except FileNotFoundError:
                    print(f"Error: File '{self.filename}' not found.")
                except Exception as e:
                    print(f"Error: {str(e)}")

    def display_help(self):
        help_text = """
PL Language Help:

Basic Syntax:
- display: Output something, e.g., display 'Hello'
- get: Get input from the user, e.g., get var 'Enter your name'
- Variables: Use = for assignment, e.g., x = 10
- Functions: Define functions using def, e.g., def my_function(x): ...
- Classes: Define classes using class, e.g., class MyClass: ...
- Control structures: Use if, elif, else, while, for
- Imports: Import modules using import, e.g., import math
- List comprehensions: [x for x in range(10) if x % 2 == 0]
- Lambda functions: lambda x: x * 2
- Decorators: Use @ symbol, e.g., @my_decorator
- Context managers: Use with statement, e.g., with open('file.txt', 'r') as f: ...

Built-in Functions:
- Math: abs, round, sum, max, min, sin, cos, tan, sqrt, log, exp, floor, ceil
- Type conversion: int, float, str, bool, list, tuple, set, dict
- Sequences: len, range, enumerate, zip, map, filter, reduce, sorted, reversed
- String operations: upper, lower, capitalize, title, strip, split, join, replace
- List operations: append, extend, insert, remove, pop, clear, index, reverse, copy
- Dictionary operations: keys, values, items, get, update
- File operations: open, read, write
- Random: randint, choice, shuffle
- Date and time: now, date, time
- JSON: json_dumps, json_loads
- Regular expressions: re_search, re_match, re_findall, re_sub
- Collections: counter, defaultdict
- Itertools: permutations, combinations
- Statistics: mean, median, mode, stdev
- Web: urlopen
- XML: xml_parse
- CSV: csv_reader, csv_writer
- Database: sqlite_connect
- Cryptography: md5, sha256, base64_encode, base64_decode
- Compression: zlib_compress, zlib_decompress
- Concurrency: thread, process, asyncio_run

TensorFlow and NumPy (if available):
- tf_constant, tf_variable, tf_matmul
- np_array, np_zeros, np_ones

Custom Functions:
- add, pickRandom, calculator

Type 'exit' to quit the interactive shell.
"""
        print(help_text)

    def greet_developer(self):
        print("Welcome, developer")

    def display_Warning(self):
        print('Already in shell')

    def display_about(self):
        print("Orion Interpreter v2.0.0")
        print(random.choice(printOptions))
        print("")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: orion run <filename.or> or orion shell for interactive mode")
    elif len(sys.argv) == 2 and sys.argv[1] == "shell":
        interpreter = OrionInterpreter(interactive=True)
        interpreter.run()
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        filename = sys.argv[2]
        interpreter = OrionInterpreter(filename=filename)
        interpreter.run()
    else:
        print("Usage: orion run <filename.or> or orion shell for interactive mode")
