import socket
import time
import pickle
import ctypes
import tkinter as tk
import numpy as np
import pandas as pd
import multiprocessing as mp
from tkinter import ttk, filedialog, messagebox, simpledialog
from threading import Thread, Condition, Event
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from collections import deque

# Start or stop OSC streaming to UE5 and Max
is_streaming = False
streaming_condition = Condition()

# Event for graceful thread termination
stop_event = Event()
threads = []

# Initialize variables
gestures = {}
X = []
y = []

# Broadcast OSC data to each device on the same network.
# Change the remote_IP to send it to a specific device.
LOCAL_IP = "127.0.0.1"
REMOTE_IP = "192.168.0.255"

UDP_RECEIVE_PORT = 48001
UDP_SEND_PORT = 48002
OSC_RECEIVE_PORT = 48003
OSC_SEND_PORT = 48004

# Bind to a socket and allow for communication even if it is being used
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("", UDP_RECEIVE_PORT))
except Exception as e:
    print(f"Socket error: {e}")

# Create OSC client and dispatcher
client = udp_client.SimpleUDPClient(LOCAL_IP, OSC_SEND_PORT)
dispatcher = dispatcher.Dispatcher()


# Thread that sends OSC data to other devices on the network
def osc_listen_thread(glove_ref):
    while True:
        client.send_message("/acceleration", glove_ref.acceleration)
        client.send_message("/rotation", glove_ref.rotation)
        client.send_message("/flex", glove_ref.flex_sensors)
        client.send_message("/joystick", glove_ref.joystick)
        client.send_message("/button", glove_ref.buttons)
        client.send_message("/battery", glove_ref.battery)
        client.send_message("/temperature", glove_ref.temperature)


# Thread that listens to incoming OSC data
def osc_dispatch_thread():
    server = osc_server.ThreadingOSCUDPServer((LOCAL_IP, OSC_RECEIVE_PORT), dispatcher)
    server.serve_forever()


# Attempt to gracefully quit the program
def exit_handler():
    global is_streaming, sock
    prediction_status.value = False
    is_streaming = False

    for thread in threads:
        terminate_thread(thread)

    sock.shutdown(socket.SHUT_RDWR)
    sock.close()
    exit()


class HandsOnGlove:
    def __init__(self):
        # Register handlers on dispatcher
        # Call corresponding functions whenever a message is sent to one of the addresses
        dispatcher.map("/rgb", self.rgb_handler)
        dispatcher.map("/haptic", self.haptic_handler)

        self.udp_data = None
        self.acceleration = None
        self.rotation = None
        self.flex_sensors = None
        self.joystick = None
        self.buttons = None
        self.battery = None
        self.temperature = None

        self.joystick_min_x = 0
        self.joystick_max_x = 4095
        self.joystick_min_y = 0
        self.joystick_max_y = 4095
        self.joystick_centre_x = 2047
        self.joystick_centre_y = 2047

        self.is_flex_calibrated = False
        self.is_joystick_calibrated = False
        self.deadzone_radius = 125
        self.edge_margin = 100

        self.remote_IP = REMOTE_IP
        self.udp_send_port = UDP_SEND_PORT

    # Take OSC data regarding the onboard LED from UE5 or Max,
    # convert them into UDP messages in the format the glove can parse
    def rgb_handler(self, unused_addr, red, green, blue):
        sock.sendto(f"/rgb|{red}|{green}|{blue}|0|0|".encode("utf-8"), (self.remote_IP, self.udp_send_port))
        return

    # Take OSC data regarding the onboard vibration motor from UE5 or Max,
    # convert them into UDP messages in the format the glove can parse
    def haptic_handler(self, unused_addr, value):
        sock.sendto(f"/haptic|{value}|0|0|0|0|".encode("utf-8"), (self.remote_IP, self.udp_send_port))
        return

    # Receive sensor data over UDP
    def get_udp_data(self):
        self.udp_data = (sock.recv(255).decode("utf-8")).split()

    # Read sensor data and update all sensor types
    def update_sensors(self):
        self.get_udp_data()
        self.get_acceleration()
        self.get_rotation()
        self.get_flex_sensors()
        self.get_joystick()
        self.get_buttons()
        self.get_battery()
        self.get_temperature()

    # Parse acceleration readings
    # |------------------------------------------------------------|
    # |   x   |   y   |   z   |  calibration status (3 is the max) |
    # | float | float | float |               int                  |
    # |------------------------------------------------------------|
    def get_acceleration(self):
        self.acceleration = [float(x) for x in self.udp_data[1:4]] + [int(self.udp_data[4])]

    # Parse rotation readings
    # |------------------------------------------------------------------------------------|
    # |   i   |   j   |   k   |   w   | error (radian) | calibration status (3 is the max) |
    # | float | float | float | float |     float      |              int                  |
    # |------------------------------------------------------------------------------------|
    def get_rotation(self):
        self.rotation = [float(x) for x in self.udp_data[6:11]] + [int(self.udp_data[11])]

    # Parse flex sensors
    # ADC on the other end is 12-bit
    def get_flex_sensors(self):
        self.flex_sensors = [float(x) for x in self.udp_data[13:21]]

    # Parse joystick readings
    # ADC on the other end is 12-bit
    def get_joystick(self):
        self.joystick = [float(x) for x in self.udp_data[22:24]]

    # Parse buttons
    # 0 is off, 1 is on
    def get_buttons(self):
        self.buttons = [int(x) for x in self.udp_data[25:27]]

    # Parse battery readings
    # ADC on the other end is 12-bit
    def get_battery(self):
        self.battery = int(self.udp_data[28])

    # Parse temperature readings
    # Data from the onboard temperature sensor
    def get_temperature(self):
        self.temperature = float(self.udp_data[30][:7])


class HandsOnGloveGUI:
    def __init__(self, glove_ref, root, queue, prediction_status):
        self.glove = glove_ref

        self.root = root
        root.title("Hands-on Glove")

        self.svm_model_tuple = ()
        self.is_model_ready = False

        # Call the update_variable function initially to start the updates
        self.get_updates = False
        self._update_variables()

        self.main_frame = tk.Frame(self.root, padx=10, pady=10, borderwidth=2, relief="groove")
        self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.frame = ttk.Frame(self.main_frame, padding=10, borderwidth=2, relief="groove")
        self.frame.pack(side="left", padx=10, pady=10, fill="x", expand=True)

        self.sensor_button_frame = tk.Frame(self.main_frame, padx=10, pady=10, borderwidth=2, relief="groove")
        self.sensor_button_frame.pack(side="right", padx=(10, 10), pady=10, fill="x", expand=True)

        self.ml_button_frame = tk.Frame(self.main_frame, padx=10, pady=10, borderwidth=2, relief="groove")
        self.ml_button_frame.pack(side="right", padx=(10, 10), pady=10, fill="x", expand=True)

        self.plot_frame = tk.Frame(self.main_frame, borderwidth=2, relief="groove")
        self.plot_frame.pack(side="left", padx=10, pady=(10, 10), fill="both", expand=True)

        self.joystick_frame = tk.Frame(self.plot_frame, borderwidth=2, relief="groove")
        self.joystick_frame.pack(side="bottom", padx=25, pady=(10, 10))

        self.joystick_canvas = tk.Canvas(self.plot_frame, width=200, height=200,
                                         bg="white", borderwidth=2, relief="groove")
        self.joystick_canvas.pack(pady=(10, 0))

        # Buttons
        self.toggle_streaming_button = tk.Button(self.sensor_button_frame, text="Start Streaming",
                                                 command=self._toggle_streaming, height=2, width=20)
        self.toggle_streaming_button.pack(side=tk.TOP, padx=10, pady=4)

        self.calibrate_flex_button = tk.Button(self.sensor_button_frame, text="Calibrate Flex Sensors",
                                               command=self._begin_flex_calibration, height=2, width=20)
        self.calibrate_flex_button.pack(side=tk.TOP, padx=10, pady=4)

        self.calibrate_joystick_button = tk.Button(self.sensor_button_frame, text="Calibrate Joystick",
                                                   command=self._begin_joystick_calibration, height=2, width=20)
        self.calibrate_joystick_button.pack(side=tk.TOP, padx=10, pady=4)

        self.save_calibration_button = tk.Button(self.sensor_button_frame, text="Save Calibration",
                                                 command=self._save_calibration, height=2, width=20)
        self.save_calibration_button.pack(side=tk.TOP, padx=10, pady=4)

        self.load_calibration_button = tk.Button(self.sensor_button_frame, text="Load Calibration",
                                                 command=self._load_calibration, height=2, width=20)
        self.load_calibration_button.pack(side=tk.TOP, padx=10, pady=4)

        self.begin_prediction_button = tk.Button(self.ml_button_frame, text="Start Posture Prediction",
                                                 command=self._begin_prediction, height=2, width=20)
        self.begin_prediction_button.pack(side=tk.TOP, padx=10, pady=4)

        self.collect_sample_button = tk.Button(self.ml_button_frame, text="Collect Posture Samples",
                                               command=self._collect_posture_samples, height=2, width=20)
        self.collect_sample_button.pack(side=tk.TOP, padx=10, pady=4)

        self.load_samples_button = tk.Button(self.ml_button_frame, text="Load Posture Samples",
                                             command=self._load_posture_samples, height=2, width=20)
        self.load_samples_button.pack(side=tk.TOP, padx=10, pady=4)

        self.train_svm_model_button = tk.Button(self.ml_button_frame, text="Train SVM Model",
                                                command=self._train_svm_model, height=2, width=20)
        self.train_svm_model_button.pack(side=tk.TOP, padx=10, pady=4)

        self.save_svm_model_button = tk.Button(self.ml_button_frame, text="Save SVM Model",
                                               command=self._save_svm_model, height=2, width=20)
        self.save_svm_model_button.pack(side=tk.TOP, padx=10, pady=4)

        self.load_svm_model_button = tk.Button(self.ml_button_frame, text="Load SVM Model",
                                               command=self._load_svm_model, height=2, width=20)
        self.load_svm_model_button.pack(side=tk.TOP, padx=10, pady=4)

        # Acceleration
        self.acc_x = tk.StringVar()
        self.acc_y = tk.StringVar()
        self.acc_z = tk.StringVar()
        self.acc_cal = tk.IntVar()

        self.acceleration_label = ttk.Label(self.frame, text="Acceleration", font=("Segoe UI", 9))
        self.acceleration_label.grid(row=1, column=0)

        self.acceleration_x_label = ttk.Label(self.frame, text="x", font=("Segoe UI", 9))
        self.acceleration_x_label.grid(row=0, column=1)

        self.acceleration_y_label = ttk.Label(self.frame, text="y", font=("Segoe UI", 9))
        self.acceleration_y_label.grid(row=0, column=2)

        self.acceleration_z_label = ttk.Label(self.frame, text="z", font=("Segoe UI", 9))
        self.acceleration_z_label.grid(row=0, column=3)

        self.acceleration_cal_label = ttk.Label(self.frame, text="cal", font=("Segoe UI", 9))
        self.acceleration_cal_label.grid(row=0, column=6)

        self.acceleration_x_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                              justify="right", textvariable=self.acc_x)
        self.acceleration_x_value.grid(row=1, column=1, padx=5)

        self.acceleration_y_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                              justify="right", textvariable=self.acc_y)
        self.acceleration_y_value.grid(row=1, column=2, padx=5)

        self.acceleration_z_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                              justify="right", textvariable=self.acc_z)
        self.acceleration_z_value.grid(row=1, column=3, padx=5)

        self.acceleration_cal_value = ttk.Entry(self.frame, width=4, font=("Segoe UI", 9),
                                                justify="center", textvariable=self.acc_cal)
        self.acceleration_cal_value.grid(row=1, column=6, padx=5)

        # Rotation
        self.quat_i = tk.StringVar()
        self.quat_j = tk.StringVar()
        self.quat_k = tk.StringVar()
        self.quat_w = tk.StringVar()
        self.quat_rad_cal = tk.StringVar()
        self.quat_cal = tk.IntVar()

        self.rotation_label = ttk.Label(self.frame, text="Rotation", font=("Segoe UI", 9))
        self.rotation_label.grid(row=3, column=0)

        self.rotation_i_label = ttk.Label(self.frame, text="i", font=("Segoe UI", 9))
        self.rotation_i_label.grid(row=2, column=1, pady=(20, 0))

        self.rotation_j_label = ttk.Label(self.frame, text="j", font=("Segoe UI", 9))
        self.rotation_j_label.grid(row=2, column=2, pady=(20, 0))

        self.rotation_k_label = ttk.Label(self.frame, text="k", font=("Segoe UI", 9))
        self.rotation_k_label.grid(row=2, column=3, pady=(20, 0))

        self.rotation_w_label = ttk.Label(self.frame, text="w", font=("Segoe UI", 9))
        self.rotation_w_label.grid(row=2, column=4, pady=(20, 0))

        self.rotation_rad_cal_label = ttk.Label(self.frame, text="rad cal", font=("Segoe UI", 9))
        self.rotation_rad_cal_label.grid(row=2, column=5, pady=(20, 0))

        self.rotation_cal_label = ttk.Label(self.frame, text="cal", font=("Segoe UI", 9))
        self.rotation_cal_label.grid(row=2, column=6, pady=(20, 0))

        self.rotation_i_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                          justify="right", textvariable=self.quat_i)
        self.rotation_i_value.grid(row=3, column=1, padx=5)

        self.rotation_j_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                          justify="right", textvariable=self.quat_j)
        self.rotation_j_value.grid(row=3, column=2, padx=5)

        self.rotation_k_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                          justify="right", textvariable=self.quat_k)
        self.rotation_k_value.grid(row=3, column=3, padx=5)

        self.rotation_w_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                          justify="right", textvariable=self.quat_w)
        self.rotation_w_value.grid(row=3, column=4, padx=5)

        self.rotation_k_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                          justify="right", textvariable=self.quat_rad_cal)
        self.rotation_k_value.grid(row=3, column=5, padx=5)

        self.rotation_w_value = ttk.Entry(self.frame, width=4, font=("Segoe UI", 9),
                                          justify="center", textvariable=self.quat_cal)
        self.rotation_w_value.grid(row=3, column=6, padx=5)

        # Flex sensors
        self.flex_thumb = tk.StringVar()
        self.flex_point_base = tk.StringVar()
        self.flex_middle_base = tk.StringVar()
        self.flex_ring_base = tk.StringVar()
        self.flex_pinky = tk.StringVar()
        self.flex_point_tip = tk.StringVar()
        self.flex_middle_tip = tk.StringVar()
        self.flex_ring_tip = tk.StringVar()

        self.flex_label = ttk.Label(self.frame, text="Flex Sensors", font=("Segoe UI", 9))
        self.flex_label.grid(row=6, column=0)

        self.flex_thumb_label = ttk.Label(self.frame, text="thumb", font=("Segoe UI", 9))
        self.flex_thumb_label.grid(row=4, column=1, pady=(20, 0))

        self.flex_point_base_label = ttk.Label(self.frame, text="point", font=("Segoe UI", 9))
        self.flex_point_base_label.grid(row=4, column=2, pady=(20, 0))

        self.flex_middle_base_label = ttk.Label(self.frame, text="middle", font=("Segoe UI", 9))
        self.flex_middle_base_label.grid(row=4, column=3, pady=(20, 0))

        self.flex_ring_base_label = ttk.Label(self.frame, text="ring", font=("Segoe UI", 9))
        self.flex_ring_base_label.grid(row=4, column=4, pady=(20, 0))

        self.quat_pinky_label = ttk.Label(self.frame, text="pinky", font=("Segoe UI", 9))
        self.quat_pinky_label.grid(row=4, column=5, pady=(20, 0))

        self.flex_thumb_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                          justify="right", textvariable=self.flex_thumb)
        self.flex_thumb_value.grid(row=6, column=1, padx=5)

        self.flex_point_base_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                               justify="right", textvariable=self.flex_point_base)
        self.flex_point_base_value.grid(row=6, column=2, padx=5)

        self.flex_point_tip_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                              justify="right", textvariable=self.flex_point_tip)
        self.flex_point_tip_value.grid(row=5, column=2, padx=5, pady=(0, 5))

        self.flex_middle_base_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                                justify="right", textvariable=self.flex_middle_base)
        self.flex_middle_base_value.grid(row=6, column=3, padx=5)

        self.flex_middle_tip_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                               justify="right", textvariable=self.flex_middle_tip)
        self.flex_middle_tip_value.grid(row=5, column=3, padx=5, pady=(0, 5))

        self.flex_ring_base_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                              justify="right", textvariable=self.flex_ring_base)
        self.flex_ring_base_value.grid(row=6, column=4, padx=5)

        self.flex_ring_tip_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                             justify="right", textvariable=self.flex_ring_tip)
        self.flex_ring_tip_value.grid(row=5, column=4, padx=5, pady=(0, 5))

        self.flex_pinky_value = ttk.Entry(self.frame, width=8, font=("Segoe UI", 9),
                                          justify="right", textvariable=self.flex_pinky)
        self.flex_pinky_value.grid(row=6, column=5, padx=5)

        # Battery
        self.battery_voltage = tk.StringVar()
        self.battery_percentage = tk.StringVar()

        self.battery_label = ttk.Label(self.frame, text="Battery", font=("Segoe UI", 9))
        self.battery_label.grid(row=8, column=0)

        self.battery_voltage_label = ttk.Label(self.frame, text="Voltage", font=("Segoe UI", 9))
        self.battery_voltage_label.grid(row=7, column=1, pady=(20, 0))

        self.battery_percentage_label = ttk.Label(self.frame, text="%", font=("Segoe UI", 9))
        self.battery_percentage_label.grid(row=7, column=2, pady=(20, 0))

        self.battery_voltage_value = ttk.Entry(self.frame, width=6, font=("Segoe UI", 9),
                                               justify="center", textvariable=self.battery_voltage)
        self.battery_voltage_value.grid(row=8, column=1, padx=5)

        self.battery_percentage_value = ttk.Entry(self.frame, width=5, font=("Segoe UI", 9),
                                                  justify="center", textvariable=self.battery_percentage)
        self.battery_percentage_value.grid(row=8, column=2, padx=5)

        # Joystick
        self.joystick_x = tk.StringVar()
        self.joystick_y = tk.StringVar()

        self.joystick_label = ttk.Label(self.joystick_frame, text="Joystick", font=("Segoe UI", 9))
        self.joystick_label.grid(row=2, column=0, padx=(25, 5), pady=(0, 10))

        self.joystick_x_label = ttk.Label(self.joystick_frame, text="x", font=("Segoe UI", 9))
        self.joystick_x_label.grid(row=1, column=1)

        self.joystick_y_label = ttk.Label(self.joystick_frame, text="y", font=("Segoe UI", 9))
        self.joystick_y_label.grid(row=1, column=2, padx=(5, 25))

        self.joystick_x_value = ttk.Entry(self.joystick_frame, width=6, font=("Segoe UI", 9),
                                          justify="center", textvariable=self.joystick_x)
        self.joystick_x_value.grid(row=2, column=1, padx=5, pady=(0, 10))

        self.joystick_y_value = ttk.Entry(self.joystick_frame, width=6, font=("Segoe UI", 9),
                                          justify="center", textvariable=self.joystick_y)
        self.joystick_y_value.grid(row=2, column=2, padx=(5, 25), pady=(0, 10))

        self.get_updates = True

    def _load_posture_samples(self):
        self.glove.load_posture_data(self)

    def _begin_joystick_calibration(self):
        self.glove.calibrate_joystick(self)

    def _begin_flex_calibration(self):
        self.glove.calibrate_flex_sensors(self)

    def _collect_posture_samples(self):
        self.glove.collect_sample(self)

    def _draw_marker(self, joystick_x, joystick_y):
        self.joystick_canvas.delete("marker")
        if is_joystick_calibrated:
            joystick_x = map_range_clamped(joystick_x, -1, 1, 0, 200)
            joystick_y = 200 - map_range_clamped(joystick_y, -1, 1, 0, 200)
        else:
            joystick_x = map_range_clamped(joystick_x, 0, 4096, 0, 200)
            joystick_y = 200 - map_range_clamped(joystick_y, 0, 4096, 0, 200)
        self.joystick_canvas.create_oval(joystick_x - 5, joystick_y - 5, joystick_x + 5, joystick_y + 5,
                                         fill="red", tags="marker")

    def _destroy_window(self, event):
        event.widget.winfo_toplevel().destroy()

    def show_entry_box(self, title, message):
        answer = simpledialog.askstring(title, message, parent=self.root)

        if answer is not None:
            return answer
        else:
            return None

    def show_popup_message(self, title, message):
        popup = tk.Toplevel()
        popup.title(title)
        popup.focus_force()

        main_window_x = self.root.winfo_rootx() + self.root.winfo_width() / 2
        main_window_y = self.root.winfo_rooty() + self.root.winfo_height() / 2

        popup_width = 600
        popup_height = 100
        popup_x = main_window_x - popup_width / 2
        popup_y = main_window_y - popup_height / 2

        popup.geometry(f"{popup_width}x{popup_height}+{int(popup_x)}+{int(popup_y)}")

        label = tk.Label(popup, text=message)
        label.pack(padx=20, pady=20)

        button = tk.Button(popup, text="OK", command=popup.destroy)
        button.pack(pady=10)
        button.focus_set()
        button.bind("<Return>", self._destroy_window)

        popup.transient(self.root)
        popup.grab_set()
        self.root.wait_window(popup)

    def ask_yesno(self, title, message):
        popup = tk.Toplevel()
        popup.withdraw()

        answer = messagebox.askyesno(title, message)

        popup.destroy()
        return answer

    def _toggle_streaming(self):
        global is_streaming
        if is_streaming:
            with streaming_condition:
                is_streaming = False
                self.get_updates = True
            self.toggle_streaming_button.config(text="Start Streaming")
        else:
            with streaming_condition:
                is_streaming = True
                streaming_condition.notify()
                self.get_updates = False
            self.toggle_streaming_button.config(text="Stop Streaming")

    def _update_variables(self):
        if self.get_updates:
            acceleration = get_acceleration(get_udp_data())
            acc_x, acc_y, acc_z = ["{:.4f}".format(x) for x in acceleration[:3]]
            acc_cal = int(acceleration[3])
            self.acc_x.set(acc_x)
            self.acc_y.set(acc_y)
            self.acc_z.set(acc_z)
            self.acc_cal.set(acc_cal)

            rotation = get_rotation(get_udp_data())
            quat_i, quat_j, quat_k, quat_w, quat_rad_cal = ["{:.4f}".format(x) for x in rotation[:5]]
            quat_cal = int(rotation[5])
            self.quat_i.set(quat_i)
            self.quat_j.set(quat_j)
            self.quat_k.set(quat_k)
            self.quat_w.set(quat_w)
            self.quat_rad_cal.set(quat_rad_cal)
            self.quat_cal.set(quat_cal)

            flex = get_flex_sensors(get_udp_data())
            (flex_ring_tip, flex_middle_tip, flex_point_tip, flex_pinky, flex_ring_base,
             flex_middle_base, flex_point_base, flex_thumb) = flex
            if is_flex_calibrated:
                (flex_ring_tip, flex_middle_tip, flex_point_tip, flex_pinky, flex_ring_base,
                 flex_middle_base, flex_point_base, flex_thumb) = ["{:.2f}".format(x) for x in get_flex_mapped(flex)]
            self.flex_thumb.set(flex_thumb)
            self.flex_point_base.set(flex_point_base)
            self.flex_middle_base.set(flex_middle_base)
            self.flex_ring_base.set(flex_ring_base)
            self.flex_pinky.set(flex_pinky)
            self.flex_point_tip.set(flex_point_tip)
            self.flex_middle_tip.set(flex_middle_tip)
            self.flex_ring_tip.set(flex_ring_tip)

            joystick_x, joystick_y = get_joystick(get_udp_data())
            if is_joystick_calibrated:
                joystick_x, joystick_y = map_joystick(joystick_x, joystick_y)
                self.joystick_x.set("{:.2f}".format(joystick_x))
                self.joystick_y.set("{:.2f}".format(joystick_y))
            else:
                self.joystick_x.set(joystick_x)
                self.joystick_y.set(joystick_y)
            self._draw_marker(joystick_x, joystick_y)

            battery_voltage = (get_battery(get_udp_data()) / 4095) * 4.3349
            battery_percentage = map_range_clamped(battery_voltage, 3.5, 3.8, 0, 100)
            self.battery_voltage.set("{:.2f}".format(battery_voltage))
            self.battery_percentage.set("{:.1f}".format(battery_percentage))

            self.root.after(10, self._update_variables)

        else:
            self.root.after(100, self._update_variables)

    def _load_calibration(self):
        filename = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if filename:
            load_calibration_values(self, filename)

    def _save_calibration(self):
        filename = filedialog.asksaveasfilename(filetypes=[("Pickle Files", "*.pkl")], defaultextension=".pkl")
        if filename:
            save_calibration_values(self, filename)
            self.show_popup_message("Success!", f"Saved flex sensor and joystick calibrations to {filename}")

    def _begin_prediction(self):
        if self.is_model_ready:
            if not prediction_status.value:
                prediction_status.value = True
                self.begin_prediction_button.config(text="Stop Posture Prediction")
            else:
                prediction_status.value = False
                self.begin_prediction_button.config(text="Start Posture Prediction")
        else:
            self.show_popup_message("Warning", "Train the model or load a saved SVM model first.")

    def _train_svm_model(self):
        global df
        try:
            mm_scaler = preprocessing.MinMaxScaler()
            predict = "Posture Number"

            data = df.drop(columns="Posture Name")
            X = np.array(data.drop(columns=predict))
            y = np.array(data[predict])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            X_train = mm_scaler.fit_transform(X_train)
            X_test = mm_scaler.fit_transform(X_test)

            clf = svm.SVC(kernel="linear", decision_function_shape="ovr")
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")

            self.svm_model_tuple = (clf, mm_scaler, postures)
            queue.put((clf, mm_scaler))
            self.is_model_ready = True
        except ValueError:
            self.show_popup_message("Error", "Collect or load samples first.")

    def _save_svm_model(self):
        if self.is_model_ready:
            filename = filedialog.asksaveasfilename(filetypes=[("Pickle Files", "*.pkl")], defaultextension=".pkl")
            if filename:
                with open(filename, "wb") as file:
                    pickle.dump(self.svm_model_tuple, file)
                self.show_popup_message("Success!", f"SVM Model saved to {filename}")
        else:
            self.show_popup_message("Warning", "Train the model or load a saved SVM model first.")

    def _load_svm_model(self):
        global postures
        filename = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if filename:
            with open(filename, "rb") as file:
                self.svm_model_tuple = pickle.load(file)
            try:
                clf, mm_scaler, postures = self.svm_model_tuple
                queue.put((clf, mm_scaler))
                self.is_model_ready = True
                self.show_popup_message("Success!", f"SVM Model loaded.")
            except ValueError:
                self.show_popup_message("Error", "The selected file is not a saved SVM model.")

def create_variations(samples, num_variations, min_target_length, max_target_length):
    variations = []
    for _ in range(num_variations):
        new_sample = []
        for quaternion, acceleration in samples:
            # Randomly choose target length between min and max values
            target_length = np.random.randint(min_target_length, max_target_length + 1)

            # Interpolate quaternion and acceleration data for longer duration
            interp_quaternion_long = interp1d(np.linspace(0, 1, len(quaternion)), quaternion)
            interp_acceleration_long = interp1d(np.linspace(0, 1, len(acceleration)), acceleration)
            new_quaternion_long = interp_quaternion_long(np.linspace(0, 1, target_length))
            new_acceleration_long = interp_acceleration_long(np.linspace(0, 1, target_length))
            new_sample.append((new_quaternion_long, new_acceleration_long))

            # Interpolate quaternion and acceleration data for shorter duration
            interp_quaternion_short = interp1d(np.linspace(0, 1, len(quaternion)), quaternion)
            interp_acceleration_short = interp1d(np.linspace(0, 1, len(acceleration)), acceleration)
            new_quaternion_short = interp_quaternion_short(np.linspace(0, 1, int(target_length / 2)))
            new_acceleration_short = interp_acceleration_short(np.linspace(0, 1, int(target_length / 2)))
            new_sample.append((new_quaternion_short, new_acceleration_short))

        variations.append(new_sample)

    return variations


# Example usage
num_variations = 5  # Number of variations per sample
min_target_length = 40  # Minimum target length for interpolation
max_target_length = 60  # Maximum target length for interpolation
augmented_gestures = {}
for gesture_name, samples in gestures.items():
    augmented_samples = create_variations(samples, num_variations, min_target_length, max_target_length)
    augmented_gestures[gesture_name] = augmented_samples


# Save augmented samples to a CSV file
for gesture_name, augmented_samples_list in augmented_gestures.items():
    for i, augmented_samples in enumerate(augmented_samples_list):
        for j, (quaternion, acceleration) in enumerate(augmented_samples):
            row = {'gesture': f"{gesture_name}_variation_{i+1}_sample_{j+1}"}
            for k, q in enumerate(quaternion):
                row[f'q{k}'] = q
            for k, a in enumerate(acceleration):
                row[f'a{k}'] = a
            X.append(row)
            y.append(gesture_name)

# Create a DataFrame and save to CSV
df = pd.DataFrame(X)
df.to_csv('gesture_samples.csv', index=False)


def map_range(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return 0.0
    return float((((value - in_min) / (in_max - in_min)) * (out_max - out_min)) + out_min)


def map_range_clamped(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return 0.0
    return float(max(min(out_max, (((value - in_min) / (in_max - in_min)) * (out_max - out_min)) + out_min), out_min))


def main():
    glove = HandsOnGlove()

    osc_listen = Thread(target=osc_listen_thread, args=(glove,), daemon=True)
    osc_dispatch = Thread(target=osc_dispatch_thread, args=(), daemon=True)

    osc_listen.start()
    osc_dispatch.start()


if __name__ == "__main__":
    main()

#
#
#     angular_velocity_threshold = 0.5  # Example threshold
#     gesture_lockout_time = 0.2
#
#     # Collect gesture samples
#     while True:
#         glove.update_sensors()
#         gesture_name = input("Enter the name of the gesture (or 'exit' to finish): ")
#         if gesture_name.lower() == 'exit':
#             break
#
#         # Collect gesture data
#         print(f"Collecting samples for {gesture_name}. Press Enter to start recording, then Enter again to stop.")
#         input("Press Enter to start...")
#         samples = []
#         while True:
#             user_input = input("Press Enter to record sample, or type 'done' to finish: ")
#             if user_input.lower() == 'done':
#                 break
#             # Simulate collecting data from IMU (replace with actual data acquisition)
#             current_sensor_data = get_udp_data()
#             quaternion = get_rotation(current_sensor_data[:-2])
#             acceleration = get_acceleration(current_sensor_data[:-1])
#             samples.append((quaternion, acceleration))
#         gestures[gesture_name] = samples
#
#
# def calculate_angular_velocity(gyro_quaternion_current, gyro_quaternion_previous, dt):
#     # Calculate quaternion derivative
#     quaternion_derivative = (gyro_quaternion_current - gyro_quaternion_previous) / dt
#
#     # Calculate angular velocity
#     angular_velocity = 2 * quaternion_derivative * gyro_quaternion_current.conjugate
#
#     return angular_velocity.imaginary
#
#
# while True:
#     # Simulate real-time data acquisition (replace with actual data acquisition)
#     quaternion = get_rotation(current_sensor_data[:-2])
#     acceleration = get_acceleration(current_sensor_data[:-1])
#     angular_velocity = np.random.rand(3)
#
#     # Check if angular velocity exceeds the threshold
#     if np.linalg.norm(angular_velocity) > angular_velocity_threshold:
#         # Perform gesture recognition
#         print("Starting gesture recording and recognition...")
#         while True:
#             # Extract features for the current sample
#             sample_features = {'q0': [quaternion[0]], 'q1': [quaternion[1]], 'q2': [quaternion[2]],
#                                'q3': [quaternion[3]],
#                                'a0': [acceleration[0]], 'a1': [acceleration[1]], 'a2': [acceleration[2]]}
#
#             # Make prediction
#             gesture_prediction = knn.predict(sample_features)[0]
#             print(f"Predicted gesture: {gesture_prediction}")
#             time.sleep(1)  # Delay for demonstration purposes
#     else:
#         print("Waiting for angular velocity to exceed threshold...")
#         time.sleep(1)  # Check every second