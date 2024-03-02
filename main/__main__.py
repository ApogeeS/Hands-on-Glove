import ctypes
import os
import pickle
import socket
import sys
import time
import threading
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from collections import deque

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
from scipy.interpolate import interp1d
from scipy.fftpack import fft
from scipy.stats import entropy, kurtosis, skew
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler

from sklearnex import patch_sklearn
patch_sklearn()

# Must import ML algorithms after patching to use the Intel implementation
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# Start or stop OSC streaming to UE5 and Max
is_streaming = False
streaming_condition = threading.Condition()

child_processes = []

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
def osc_listen_thread(event, glove_ref, is_predicting_postures, is_predicting_gestures,
                      posture_result, gesture_result, stream_condition):
    global is_streaming
    while not event.is_set():
        while True:
            with stream_condition:
                while not is_streaming:
                    stream_condition.wait()

                client.send_message("/acceleration", glove_ref.acceleration)
                client.send_message("/rotation", glove_ref.rotation)

                # Send calibrated flex data if flex sensors are calibrated
                if glove_ref.is_flex_calibrated:
                    client.send_message("/flex", glove_ref.flex_sensors_mapped)
                else:
                    client.send_message("/flex", glove_ref.flex_sensors)

                # Send normalized joystick data if joystick is calibrated
                if glove_ref.is_joystick_calibrated:
                    client.send_message("/joystick", glove_ref.joystick_mapped)
                else:
                    client.send_message("/joystick", glove_ref.joystick)

                client.send_message("/button", glove_ref.buttons)
                client.send_message("/battery", glove_ref.battery)
                client.send_message("/temperature", glove_ref.temperature)
                if is_predicting_postures.value:
                    for name, number in glove_ref.postures.items():
                        if posture_result.value == number:
                            client.send_message("/posture/name", name)
                            client.send_message("/posture/number", number)

                if is_predicting_gestures.value:
                    for name, number in glove_ref.gestures.items():
                        if gesture_result.value == number + 1:
                            client.send_message("/gesture/name", name)
                            client.send_message("/gesture/number", number)
                time.sleep(0.005)


# Thread that listens to incoming OSC data
def osc_dispatch_thread(event):
    while not event.is_set():
        server = osc_server.ThreadingOSCUDPServer((LOCAL_IP, OSC_RECEIVE_PORT), dispatcher)
        server.serve_forever()


# Update sensor data and synchronized arrays
def update_sensors_thread(termination_event, glove_ref, flex, accel, rot):
    while not termination_event.is_set():
        glove_ref.update_sensors()

        if glove_ref.is_flex_calibrated:
            flex[:] = glove_ref.flex_sensors_mapped
        else:
            flex[:] = glove_ref.flex_sensors
        accel[:] = glove_ref.acceleration[:3]
        rot[:] = glove_ref.rotation[:4]


def core_process(process_termination_event, posture_queue, gesture_queue, flex, accel, rot,
                 posture_result, is_predicting_postures, gesture_result, is_predicting_gestures):
    thread_termination_event = threading.Event()
    core_threads = []

    glove = HandsOnGlove(posture_queue, gesture_queue)

    osc_listen = threading.Thread(target=osc_listen_thread,
                                  args=(thread_termination_event, glove, is_predicting_postures, is_predicting_gestures,
                                        posture_result, gesture_result, streaming_condition),
                                  daemon=True)
    core_threads.append(osc_listen)

    osc_dispatch = threading.Thread(target=osc_dispatch_thread,
                                    args=(thread_termination_event,),
                                    daemon=True)
    core_threads.append(osc_dispatch)

    glove_update = threading.Thread(target=update_sensors_thread,
                                    args=(thread_termination_event, glove, flex, accel, rot),
                                    daemon=True)
    core_threads.append(glove_update)

    for thread in core_threads:
        thread.start()

    root = tk.Tk()
    gui = HandsOnGloveGUI(glove, root, is_predicting_postures, is_predicting_gestures)
    glove.gui = gui
    root.protocol("WM_DELETE_WINDOW",
                  lambda: exit_handler(thread_termination_event, process_termination_event, core_threads))
    root.mainloop()


def predict_posture_process(posture_queue, prediction_status, posture_result, flex_sensor_readings):
    _svm_clf = None
    _svm_mm_scaler = None
    # TODO fine-tune the confidence
    _svm_confidence_threshold = 0.39
    last_postures = deque(maxlen=100)

    while True:
        start_time = time.time()

        while not posture_queue.empty():
            _svm_clf, _svm_mm_scaler = posture_queue.get()

        while prediction_status.value:
            features = extract_posture_features(flex_sensor_readings[:])

            _X_sample = np.array([[features[key] for key in features]])
            sensor_data = _svm_mm_scaler.transform(_X_sample)

            decision_scores = _svm_clf.decision_function(sensor_data)
            normalized_scores = normalize(decision_scores, norm="l1", axis=1)

            max_index = np.argmax(normalized_scores) + 1
            max_score = np.max(normalized_scores)

            if max_score < _svm_confidence_threshold:
                posture_result.value = -1  # -1 represents "not doing any trained posture"
            else:
                last_postures.append(max_index)
                if len(set(last_postures)) == 1 and len(last_postures) == 100:
                    posture_result.value = max_index

        # Wait for the remaining time to reach 0.01 seconds
        elapsed_time = time.time() - start_time
        remaining_time = max(0.0, 0.01 - elapsed_time)
        time.sleep(remaining_time)


def predict_gesture_process(gesture_queue, prediction_status, gesture_result, acceleration, rotation):
    _knn_clf = None
    _knn_scaler = None
    _knn_confidence_threshold = 0.5
    _gesture_activation_threshold = 10
    _gesture_termination_threshold = 1.5
    _gesture_lockout_period = 0.3
    _initial_rotation_quat = None
    _collected_samples = []
    _recording_start_time = None
    rotation_history = deque(maxlen=5)
    acceleration_history = deque(maxlen=5)
    is_currently_recording = False

    while True:
        while not gesture_queue.empty():
            _knn_clf, _knn_scaler = gesture_queue.get()

        while prediction_status.value:
            start_time = time.time()
            rotation_history.append(rotation[:])
            acceleration_history.append(acceleration[:])

            if len(rotation_history) == 5 and len(acceleration_history) == 5:
                _angular_velocity = calculate_angular_velocity(rotation_history[-1], rotation_history[-2], 0.01)
                if _angular_velocity > 0:
                    if _angular_velocity >= _gesture_activation_threshold and not is_currently_recording:
                        print(f"Started recording gesture.")
                        _recording_start_time = time.time()
                        _initial_rotation_quat = rotation_history[-1]
                        is_currently_recording = True

                    if is_currently_recording:
                        accel = acceleration_history[-1]
                        rot = calculate_relative_orientation(_initial_rotation_quat, rotation_history[-1])
                        _collected_samples.append([accel, rot])

                        if (_angular_velocity <= _gesture_termination_threshold and
                                _gesture_lockout_period <= time.time() - _recording_start_time):
                            print(f"Stopped recording gesture. Collected sample count: {len(_collected_samples)}")
                            is_currently_recording = False
                            break

            # Wait for the remaining time to reach 0.01 seconds
            elapsed_time = time.time() - start_time
            remaining_time = max(0.0, 0.01 - elapsed_time)
            time.sleep(remaining_time)

        if len(_collected_samples) >= 30:
            accel_data = np.array([sample[0] for sample in _collected_samples])
            quat_data = np.array([sample[1] for sample in _collected_samples])

            resampled_samples = resample_gesture_samples(accel_data, quat_data, 100)
            accel = []
            quat = []

            for i in range(len(resampled_samples)):
                accel.append(resampled_samples[i][0])
                quat.append(resampled_samples[i][1])

            features = extract_gesture_features(np.array(accel), np.array(quat))
            _collected_samples = []

            _X_sample = np.array([[features[key] for key in features]])
            sensor_data = _knn_scaler.transform(_X_sample)
            decision_scores = _knn_clf.predict_proba(sensor_data)
            normalized_scores = normalize(decision_scores, norm="l1", axis=1)
            max_index = np.argmax(normalized_scores) + 1
            max_score = np.max(normalized_scores)

            if max_score < _knn_confidence_threshold:
                gesture_result.value = -1  # -1 represents "not doing any trained posture"
            else:
                gesture_result.value = max_index
        elif not is_currently_recording:
            gesture_result.value = -1
            _collected_samples = []


# Attempt to gracefully quit the program
def exit_handler(thread_termination_event, process_termination_event, threads):
    global sock

    for active_thread in threads:
        terminate_thread(active_thread, thread_termination_event)

    sock.shutdown(socket.SHUT_RDWR)
    sock.close()
    process_termination_event.set()


# Signal threads to stop. Shut everything down forcefully if they don't comply
def terminate_thread(thread, thread_termination_event):
    thread_termination_event.set()
    time.sleep(0.01)
    if not thread.is_alive():
        return

    # Somewhat overkill, but: Forcefully terminate threads
    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("Non-existent thread")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class HandsOnGlove:
    def __init__(self, posture_queue, gesture_queue, gui=None):
        # Register handlers on dispatcher
        # Call corresponding functions whenever a message is sent to one of the addresses
        dispatcher.map("/rgb", self.rgb_handler)
        dispatcher.map("/haptic", self.haptic_handler)

        self._remote_IP = REMOTE_IP
        self._udp_send_port = UDP_SEND_PORT

        self.udp_data = None
        self.acceleration = None
        self.rotation = None
        self.flex_sensors = None
        self.flex_sensors_mapped = None
        self.joystick = None
        self.joystick_mapped = None
        self.buttons = None
        self.battery = None
        self.temperature = None

        self._joystick_min_x = self._joystick_min_y = 0
        self._joystick_max_x = self._joystick_max_y = 4095
        self._joystick_centre_x = self._joystick_centre_y = 2047

        self.is_flex_calibrated = False
        self._flex_calibration: dict = {x: {"open": None, "closed": None} for x in range(8)}
        self.is_joystick_calibrated = False
        self._deadzone_radius = 125
        self._edge_margin = 100

        self.gesture_activation_threshold = 15
        self.gesture_termination_threshold = 0.5
        self.gesture_lockout_period = 0.3
        self._initial_rotation_quat = None
        self._recording_start_time = None

        self.is_svm_ready = False
        self.postures = {}
        self._posture_file = None
        self.is_posture_data_loaded = False
        self._svm_clf = None
        self._svm_mm_scaler = None
        self.posture_queue = posture_queue

        self.is_knn_ready = False
        self.gestures = {}
        self._gesture_file = None
        self.is_gesture_data_loaded = False
        self._knn_clf = None
        self._knn_scaler = None
        self.gesture_queue = gesture_queue

        self.accel_axes = ["x", "y", "z"]
        self.quat_axes = ["i", "j", "k", "real"]
        self.num_gesture_samples = 100

        self.gui = gui

        # Initialize column structures for sample data collection
        self._posture_columns = ["Posture Name", "Posture Number",
                                 "Flex_1", "Flex_2", "Flex_3", "Flex_4", "Flex_5", "Flex_6", "Flex_7", "Flex_8"]
        self.df_posture = pd.DataFrame(columns=self._posture_columns)

        self._gesture_columns = ["Gesture Name", "Gesture Number", "Sample Number",
                                 "Acceleration_x", "Acceleration_y", "Acceleration_z",
                                 "Rotation_i", "Rotation_j", "Rotation_k", "Rotation_real"]
        self.df_gesture = pd.DataFrame(columns=self._gesture_columns)

        self.update_sensors()

    # Take OSC data regarding the onboard LED from UE5 or Max,
    # convert them into UDP messages in the format the glove can parse
    def rgb_handler(self, unused_addr, red, green, blue):
        sock.sendto(f"/rgb|{red}|{green}|{blue}|0|0|".encode("utf-8"), (self._remote_IP, self._udp_send_port))
        return

    # Take OSC data regarding the onboard vibration motor from UE5 or Max,
    # convert them into UDP messages in the format the glove can parse
    def haptic_handler(self, unused_addr, value):
        sock.sendto(f"/haptic|{value}|0|0|0|0|".encode("utf-8"), (self._remote_IP, self._udp_send_port))
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
        if self.is_flex_calibrated:
            self.get_flex_mapped()
        self.get_joystick()
        if self.is_joystick_calibrated:
            self.get_joystick_mapped()
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

    def calibrate_flex_sensors(self):
        _continue_calibration = self.gui.ask_yesno("Continue?", "Proceed to flex sensor calibration?")

        if _continue_calibration:
            print("Flex Sensor Calibration Started")

            self.gui.show_popup_message("Flex Sensor Calibration", "Open your hand completely and click OK...")
            open_values = self.flex_sensors

            self.gui.show_popup_message("Flex Sensor Calibration", "Close your fist and click OK...")
            closed_values = self.flex_sensors

            # Calculate calibration values for each sensor
            for sensor_num in range(8):
                open_value = open_values[sensor_num]
                closed_value = closed_values[sensor_num]

                # Store the calibration values
                self._flex_calibration[sensor_num]["open"] = open_value
                self._flex_calibration[sensor_num]["closed"] = closed_value

            self.is_flex_calibrated = True
            self.gui.show_popup_message("Success!", "Flex Sensor Calibration Completed.")

    # Take flex sensor readings, output mapped values between 0 and 1000
    def get_flex_mapped(self):
        flex_mapped = []
        flex_min = 0
        flex_max = 1000

        for sensor_num, value in enumerate(self.flex_sensors):
            open_value = self._flex_calibration[sensor_num]["open"]
            closed_value = self._flex_calibration[sensor_num]["closed"]
            flex_mapped.append(map_range_clamped(int(value), open_value, closed_value, flex_min, flex_max))

        self.flex_sensors_mapped = flex_mapped

    def calibrate_joystick(self):
        continue_calibration = self.gui.ask_yesno("Continue?", "Proceed to joystick calibration?")

        if continue_calibration:
            print("Joystick Calibration Started")
            self.gui.show_popup_message("Joystick Calibration", "Centre the joystick and click OK...")
            centre_x = self.joystick[0]
            centre_y = self.joystick[1]

            self.gui.show_popup_message("Joystick Calibration", "Move the joystick to the left edge and click OK...")
            min_x = self.joystick[0]

            self.gui.show_popup_message("Joystick Calibration", "Move the joystick to the right edge and click OK...")
            max_x = self.joystick[0]

            self.gui.show_popup_message("Joystick Calibration", "Move the joystick to the bottom edge and click OK...")
            min_y = self.joystick[1]

            self.gui.show_popup_message("Joystick Calibration", "Move the joystick to the top edge and click OK...")
            max_y = self.joystick[1]

            self._joystick_min_x = min_x
            self._joystick_max_x = max_x
            self._joystick_min_y = min_y
            self._joystick_max_y = max_y
            self._joystick_centre_x = centre_x
            self._joystick_centre_y = centre_y

            self.is_joystick_calibrated = True
            self.gui.show_popup_message("Success!", "Joystick Calibration Completed.")

    # Take joystick sensor readings, output normalized x and y
    def get_joystick_mapped(self):
        joystick_x, joystick_y = self.joystick

        if joystick_x < self._joystick_centre_x:
            normalized_x = map_range_clamped(joystick_x, self._joystick_min_x + self._edge_margin,
                                             self._joystick_centre_x - self._deadzone_radius, -1, 0)
        elif joystick_x > self._joystick_centre_x:
            normalized_x = map_range_clamped(joystick_x, self._joystick_centre_x + self._deadzone_radius,
                                             self._joystick_max_x - self._edge_margin, 0, 1)
        else:
            normalized_x = 0

        if joystick_y < self._joystick_centre_y:
            normalized_y = map_range_clamped(joystick_y, self._joystick_min_y + self._edge_margin,
                                             self._joystick_centre_y - self._deadzone_radius, -1, 0)
        elif joystick_y > self._joystick_centre_y:
            normalized_y = map_range_clamped(joystick_y, self._joystick_centre_y + self._deadzone_radius,
                                             self._joystick_max_y - self._edge_margin, 0, 1)
        else:
            normalized_y = 0

        self.joystick_mapped = normalized_x, normalized_y

    def save_calibration_values(self, filename):
        if self.is_flex_calibrated and self.is_joystick_calibrated:
            calibration_data = {"flex_calibration": self._flex_calibration,
                                "joystick_min_x": self._joystick_min_x,
                                "joystick_max_x": self._joystick_max_x,
                                "joystick_min_y": self._joystick_min_y,
                                "joystick_max_y": self._joystick_max_y,
                                "joystick_centre_x": self._joystick_centre_x,
                                "joystick_centre_y": self._joystick_centre_y}

            with open(filename, "wb") as file:
                pickle.dump(calibration_data, file)

            self.gui.show_popup_message("Calibration saved!", f"Calibration values saved to {filename}")

        else:
            self.gui.show_popup_message("Warning", "Run both calibration procedures before saving.")

    def load_calibration_values(self, filename):
        try:
            with open(filename, "rb") as file:
                calibration_data = pickle.load(file)
                if calibration_data:
                    self._flex_calibration = calibration_data["flex_calibration"]
                    self._joystick_min_x = calibration_data["joystick_min_x"]
                    self._joystick_max_x = calibration_data["joystick_max_x"]
                    self._joystick_min_y = calibration_data["joystick_min_y"]
                    self._joystick_max_y = calibration_data["joystick_max_y"]
                    self._joystick_centre_x = calibration_data["joystick_centre_x"]
                    self._joystick_centre_y = calibration_data["joystick_centre_y"]
                    if file:
                        self.is_flex_calibrated = True
                        self.is_joystick_calibrated = True
                        self.gui.show_popup_message("Success!", "Calibration values loaded.")
                else:
                    self.is_flex_calibrated = False
                    self.is_joystick_calibrated = False
                    self.gui.show_popup_message("Warning", "Empty calibration file. "
                                                           "You may need to recalibrate.")
            return None
        except TypeError:
            self.gui.show_popup_message("Warning", "Incompatible calibration file. You may need to recalibrate.")
        except FileNotFoundError:
            return None

    # Collect flex sensor readings for a specific posture
    def collect_posture_sample(self):
        _posture_name = ""
        _posture_number = 1

        if not self.is_flex_calibrated:
            self.gui.show_popup_message("Warning", "Calibrate flex sensors or load calibration file first.")
        else:
            self._posture_file = filedialog.asksaveasfilename(filetypes=[("Comma-Separated Values", "*.csv")],
                                                              defaultextension=".csv")
            if not self._posture_file:
                return

            new_sample = False
            if not new_sample:
                if not self.is_posture_data_loaded:
                    _posture_name = self.gui.show_entry_box("Posture name", "Type a posture name into the box.")
                else:
                    _posture_name = self.gui.show_entry_box("Posture name",
                                                             f"Type a posture name into the box.\n"
                                                             f"Below postures exists in the loaded file\n"
                                                             f"{[key for key in self.postures.keys()]}")
                if _posture_name is not None:
                    new_sample = True
                    if self.postures:
                        if _posture_name in self.postures:
                            _posture_number = self.postures[_posture_name]
                        else:
                            print(self.postures.values())
                            last_posture = max(self.postures.values())
                            _posture_number = last_posture + 1
                else:
                    self.gui.show_popup_message("Error", "Incorrect entry. Sample collection cancelled.")

            if self._posture_file:
                while new_sample:
                    self.gui.show_popup_message("Save",
                                                f"Click OK to save a new sample for "
                                                f"{_posture_number} - {_posture_name}.")
                    _posture_data = self.flex_sensors_mapped
                    self.posture_add_row(_posture_name, _posture_number, _posture_data)
                    new_sample = self.gui.ask_yesno("Continue?",
                                                    "Do you want to collect more data for the same posture?")

                    while not new_sample:
                        another_posture = self.gui.ask_yesno("New posture?",
                                                             "Do you want to collect data for a new posture?")

                        if another_posture:
                            if not self.is_posture_data_loaded:
                                _posture_name = self.gui.show_entry_box("Posture name",
                                                                         "Type a posture name into the box.")
                                _posture_number += 1
                            else:
                                _posture_name = self.gui.show_entry_box("Posture name",
                                                                         f"Type a posture name into the box.\n"
                                                                         f"Below postures exists in the loaded file\n"
                                                                         f"{[key for key in self.postures.keys()]}")
                            if _posture_name is not None:
                                new_sample = True
                                if self.postures:
                                    if _posture_name in self.postures:
                                        _posture_number = self.postures[_posture_name]
                                    else:
                                        last_posture = max(self.postures.values())
                                        _posture_number = last_posture + 1
                        else:
                            break
                if _posture_name is not None:
                    self.df_posture.to_csv(self._posture_file, index=False)
                    self.gui.show_popup_message("Success!", "Sample collection finished.")

    def collect_gesture_sample(self):
        collected_samples = []
        self._gesture_file = filedialog.asksaveasfilename(filetypes=[("Comma-Separated Values", "*.csv")],
                                                          defaultextension=".csv")

        if not self._gesture_file:
            return

        _gesture_name = ""
        _gesture_number = 0
        _sample_number = 1

        while True:
            if not _gesture_name:
                if not self.is_gesture_data_loaded:
                    _gesture_name = self.gui.show_entry_box("Gesture name", "Type a gesture name into the box.")
                else:
                    for label, number in zip(self.df_gesture["Gesture Name"], self.df_gesture["Gesture Number"]):
                        if label not in self.gestures:
                            self.gestures[label] = number
                    _gesture_name = self.gui.show_entry_box("Gesture name",
                                                            f"Type a gesture name into the box.\n"
                                                            f"Below gestures exists in the loaded file\n"
                                                            f"{[key for key in self.gestures.keys()]}")
                if _gesture_name is None:
                    self.gui.show_popup_message("Error", "Incorrect entry. Sample collection cancelled.")
                    break

                if self.gestures:
                    if _gesture_name in self.gestures:
                        _gesture_number = self.gestures[_gesture_name]
                        _df_filtered = self.df_gesture[self.df_gesture["Gesture Number"] == _gesture_number]
                        _sample_number = max(_df_filtered["Sample Number"]) + 1
                    else:
                        _gesture_number = max(self.gestures.values(), default=0) + 1
                        self.gestures[_gesture_name] = _gesture_number
                        _sample_number = 1

            rotation_history = deque(maxlen=5)
            acceleration_history = deque(maxlen=5)
            self.gui.show_popup_message("Save",
                                        f"Click OK and perform gesture to save a new sample for "
                                        f"{_gesture_number} - {_gesture_name}.")
            is_currently_recording = False

            while True:
                start_time = time.time()
                rotation_history.append(self.rotation[:4])
                acceleration_history.append(self.acceleration[:3])

                if len(rotation_history) == 5 and len(acceleration_history) == 5:
                    _angular_velocity = calculate_angular_velocity(rotation_history[-1], rotation_history[-2], 0.01)
                    if _angular_velocity > 0:
                        if _angular_velocity >= self.gesture_activation_threshold and not is_currently_recording:
                            print(f"Started recording gesture.")
                            self._recording_start_time = time.time()
                            self._initial_rotation_quat = rotation_history[-1]
                            is_currently_recording = True

                        if is_currently_recording:
                            acceleration = acceleration_history[-1]
                            rotation = calculate_relative_orientation(self._initial_rotation_quat,
                                                                      rotation_history[-1])
                            collected_samples.append([_gesture_name, _gesture_number, _sample_number,
                                                      acceleration, rotation])

                            if (_angular_velocity <= self.gesture_termination_threshold and
                                    self.gesture_lockout_period <= time.time() - self._recording_start_time):
                                print(f"Stopped recording gesture. Collected sample count: {len(collected_samples)}")
                                break

                # Wait for the remaining time to reach 0.01 seconds
                elapsed_time = time.time() - start_time
                remaining_time = max(0.0, 0.01 - elapsed_time)
                time.sleep(remaining_time)

            if len(collected_samples) < 50:
                self.gui.show_popup_message("Not enough samples", "Recorded gesture doesn't have enough samples")
            else:
                save_sample = self.gui.ask_yesno("Save sample?", "Do you want to save the collected sample?")
                if save_sample:
                    for sample in collected_samples:
                        gest_name, gest_no, samp_no, accel, rot = sample
                        self.gesture_add_row(gest_name, gest_no, samp_no, accel, rot)
                    collected_samples = []
                    _sample_number += 1

            new_sample = self.gui.ask_yesno("Continue?", "Do you want to collect more samples for the same gesture?")

            if not new_sample:
                another_gesture = self.gui.ask_yesno("New gesture?", "Do you want to collect data for a new gesture?")

                if another_gesture:
                    _gesture_name = ""
                    continue
                else:
                    break

        if _gesture_name:
            self.df_gesture.to_csv(self._gesture_file, index=False)
            self.gui.show_popup_message("Success!", "Sample collection finished.")

    # Load posture samples from a csv
    def load_posture_samples(self):
        self._posture_file = filedialog.askopenfilename(filetypes=[("Comma-separated Values", "*.csv")],
                                                        defaultextension=".csv")

        if self._posture_file:
            self.df_posture = pd.read_csv(self._posture_file, sep=",")
            for label, number in zip(self.df_posture["Posture Name"], self.df_posture["Posture Number"]):
                if label not in self.postures:
                    self.postures[label] = number

            if len(self.postures) > 0:
                self.is_posture_data_loaded = True
                self.gui.show_popup_message("Success!", "Data loaded successfully.")
            else:
                self.gui.show_popup_message("Error", "Empty or corrupted data. Try loading from another file.")

    def load_gesture_samples(self):
        self._gesture_file = filedialog.askopenfilename(filetypes=[("Comma-separated Values", "*.csv")],
                                                        defaultextension=".csv")

        if self._gesture_file:
            self.df_gesture = pd.read_csv(self._gesture_file, sep=",")
            for label, number in zip(self.df_gesture["Gesture Name"], self.df_gesture["Gesture Number"]):
                if label not in self.gestures:
                    self.gestures[label] = number

            if len(self.gestures) > 0:
                self.is_gesture_data_loaded = True
                self.gui.show_popup_message("Success!", "Data loaded successfully.")
            else:
                self.gui.show_popup_message("Error", "Empty or corrupted data. Try loading from another file.")

    def train_svm_model(self):
        try:
            self._svm_mm_scaler = MinMaxScaler()
            _predict = "Posture Number"

            data = self.df_posture.drop(columns="Posture Name")
            _X = np.array(data.drop(columns=_predict))
            _y = np.array(data[_predict])

            _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2)

            _X_train = self._svm_mm_scaler.fit_transform(_X_train)
            _X_test = self._svm_mm_scaler.fit_transform(_X_test)

            self._svm_clf = svm.SVC(kernel="linear", decision_function_shape="ovr")
            self._svm_clf.fit(_X_train, _y_train)

            _y_pred = self._svm_clf.predict(_X_test)

            accuracy = accuracy_score(_y_test, _y_pred)
            report = classification_report(_y_test, _y_pred)
            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")

            # Flush queue so that only the last trained model is sent through
            if not self.posture_queue.empty():
                while not self.posture_queue.empty():
                    self.posture_queue.get()

            self.posture_queue.put((self._svm_clf, self._svm_mm_scaler))
            self.is_svm_ready = True
        except ValueError:
            self.gui.show_popup_message("Error", "Collect or load samples first.")

    def train_knn_model(self):
        try:
            data = self.df_gesture.drop(columns=["Gesture Name", "Sample Number"])
            _predict = "Gesture Number"
            _X = np.array(data.drop(columns=_predict))
            _y = np.array(data[_predict])

            _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2)

            self._knn_scaler = StandardScaler()
            _X_train_scaled = self._knn_scaler.fit_transform(_X_train)
            _X_test_scaled = self._knn_scaler.transform(_X_test)

            self._knn_clf = KNeighborsClassifier(n_neighbors=3)
            self._knn_clf.fit(_X_train_scaled, _y_train)

            _y_pred = self._knn_clf.predict(_X_test_scaled)

            accuracy = accuracy_score(_y_test, _y_pred)
            precision = precision_score(_y_test, _y_pred, average='weighted')
            recall = recall_score(_y_test, _y_pred, average='weighted')
            f1 = f1_score(_y_test, _y_pred, average='weighted')

            # Display the evaluation metrics
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")

            # Flush queue so that only the last trained model is sent through
            if not self.gesture_queue.empty():
                while not self.gesture_queue.empty():
                    self.gesture_queue.get()

            self.gesture_queue.put((self._knn_clf, self._knn_scaler))
            self.is_knn_ready = True

            print(self.gestures)

        except ValueError:
            self.gui.show_popup_message("Error", "Collect or load samples first.")

    def save_svm_model(self, filename):
        if self.is_svm_ready:
            with open(filename, "wb") as file:
                pickle.dump((self._svm_clf, self._svm_mm_scaler, self.postures), file)
            self.gui.show_popup_message("Success!", f"SVM Model saved to {filename}")
        else:
            self.gui.show_popup_message("Warning", "Train the model or load a saved SVM model first.")

    def save_knn_model(self, filename):
        if self.is_knn_ready:
            with open(filename, "wb") as file:
                pickle.dump((self._knn_clf, self._knn_scaler, self.gestures), file)
            self.gui.show_popup_message("Success!", f"KNN Model saved to {filename}")
        else:
            self.gui.show_popup_message("Warning", "Train the model or load a saved KNN model first.")

    def load_svm_model(self, filename):
        with open(filename, "rb") as file:
            try:
                self._svm_clf, self._svm_mm_scaler, self.postures = pickle.load(file)
                # Flush queue so that only the last trained model is sent through
                if not self.posture_queue.empty():
                    while not self.posture_queue.empty():
                        self.posture_queue.get()
                self.posture_queue.put((self._svm_clf, self._svm_mm_scaler))
                self.is_svm_ready = True
                self.gui.show_popup_message("Success!", f"SVM Model loaded.")
            except ValueError:
                self.gui.show_popup_message("Error", "The selected file is not a saved SVM model.")

    def load_knn_model(self, filename):
        with open(filename, "rb") as file:
            try:
                self._knn_clf, self._knn_scaler, self.gestures = pickle.load(file)
                # Flush queue so that only the last trained model is sent through
                if not self.gesture_queue.empty():
                    while not self.gesture_queue.empty():
                        self.gesture_queue.get()
                self.gesture_queue.put((self._knn_clf, self._knn_scaler))
                self.is_knn_ready = True
                self.gui.show_popup_message("Success!", f"KNN Model loaded.")
            except ValueError:
                self.gui.show_popup_message("Error", "The selected file is not a saved KNN model.")

    # Add a row at the end of the posture dataframe
    def posture_add_row(self, posture_label, posture_no, sensor_values):
        new_row = {"Posture Name": posture_label, "Posture Number": posture_no,
                   **{f"Flex_{x}": float(val) for x, val in enumerate(sensor_values, start=1)}}
        self.df_posture.loc[len(self.df_posture)] = new_row

    # Add a row at the end of the gesture dataframe
    def gesture_add_row(self, gesture_label, gesture_no, sample_no, acceleration_values, rotation_values):
        new_row = {"Gesture Name": gesture_label, "Gesture Number": gesture_no, "Sample Number": sample_no,
                   **{f"Acceleration_{axis}": float(val) for axis, val in zip(["x", "y", "z"], acceleration_values)},
                   **{f"Rotation_{axis}": float(val) for axis, val in zip(["i", "j", "k", "real"], rotation_values)}}
        self.df_gesture.loc[len(self.df_gesture)] = new_row

    def create_posture_variations(self, num_variations):
        _posture_file = filedialog.asksaveasfilename(filetypes=[("Comma-Separated Values", "*.csv")],
                                                     defaultextension=".csv")
        _df_original = pd.read_csv(_posture_file)

        for posture_name in _df_original["Posture Name"].unique():
            _df_posture = _df_original[_df_original["Posture Name"] == posture_name]

            for index, row in _df_posture.iterrows():
                for i in range(num_variations):
                    flex_data = row.drop(["Posture Name", "Posture Number"])
                    noise = np.random.normal(loc=0, scale=25, size=len(flex_data))
                    variation = np.clip(flex_data + noise, 0, 1000)
                    self.posture_add_row(row["Posture Name"], row["Posture Number"], variation)
            print(f"Created variations for {posture_name} posture.")

        if _posture_file is not None:
            self.df_posture.to_csv(self._posture_file, index=False)
            self.gui.show_popup_message("Success!", "Variations created and saved to file.")

    def create_gesture_variations(self, min_target_length, max_target_length, num_variations):
        _gesture_name = filedialog.asksaveasfilename(filetypes=[("Comma-Separated Values", "*.csv")],
                                                     defaultextension=".csv")
        _df_original = pd.read_csv(_gesture_name)

        # For each unique gesture in dataframe
        for gesture_name in _df_original["Gesture Name"].unique():
            gesture = _df_original[_df_original["Gesture Name"] == gesture_name]

            for i in range(num_variations):
                last_sample_numbers = self.df_gesture.groupby("Gesture Name")["Sample Number"].max().to_dict()
                last_sample_no = last_sample_numbers[gesture_name]

                # Iterate through the samples of the specified gesture
                for sample_number, sample_data in gesture.groupby("Sample Number"):
                    accel_data = sample_data[["Acceleration_x", "Acceleration_y", "Acceleration_z"]].values
                    quat_data = sample_data[["Rotation_i", "Rotation_j", "Rotation_k", "Rotation_real"]].values
                    gesture_no = sample_data["Gesture Number"].iloc[0]

                    # Determine target length within min and max limits
                    target_length = np.random.randint(min_target_length, max_target_length + 1)

                    # Interpolate to extend or shrink timeline
                    new_quaternion = np.zeros((target_length, 4))
                    new_acceleration = np.zeros((target_length, 3))

                    # Fix the first and last values
                    new_quaternion[0] = quat_data[0]
                    new_acceleration[0] = accel_data[0]
                    new_quaternion[-1] = quat_data[-1]
                    new_acceleration[-1] = accel_data[-1]

                    # Calculate indices for interpolation
                    old_indices = np.linspace(0, len(quat_data) - 1, num=len(quat_data))
                    new_indices = np.linspace(0, len(quat_data) - 1, num=target_length)

                    # Perform interpolation for in-between values
                    for axis in range(4):
                        new_quaternion[1:-1, axis] = np.interp(new_indices[1:-1], old_indices, quat_data[:, axis])
                    for axis in range(3):
                        new_acceleration[1:-1, axis] = np.interp(new_indices[1:-1], old_indices, accel_data[:, axis])

                    # Add Gaussian noise to the new_acceleration and new_quaternion arrays
                    noise_acceleration = np.random.normal(loc=0, scale=0.01, size=new_acceleration.shape)
                    new_acceleration += noise_acceleration

                    noise_quaternion = np.random.normal(loc=0, scale=0.01, size=new_quaternion.shape)
                    new_quaternion += noise_quaternion

                    # Normalize the quaternion
                    new_quaternion /= np.linalg.norm(new_quaternion)

                    samp_no = sample_number + last_sample_no
                    # Add the new gesture data to the dataframe
                    for acc, quat in zip(new_acceleration, new_quaternion):
                        self.gesture_add_row(gesture_name, gesture_no, samp_no, acc, quat)
                    print(f"Created variations from #{sample_number} of {gesture_name} gesture. "
                          f"New sample number is: {samp_no}.")

        if _gesture_name is not None:
            self.df_gesture.to_csv(self._gesture_file, index=False)
            self.gui.show_popup_message("Success!", "Variations created and saved to file.")

    @staticmethod
    def extract_posture_features(filename):
        df_posture_data = pd.read_csv(filename)
        new_data = []

        for posture_name in df_posture_data["Posture Name"].unique():
            posture = df_posture_data[df_posture_data["Posture Name"] == posture_name]

            for index, row in posture.iterrows():
                features = extract_posture_features(row.drop(["Posture Name", "Posture Number"]).tolist())
                new_row = {"Posture Name": posture_name, "Posture Number": row["Posture Number"]}
                new_row.update(features)
                new_data.append(new_row)

                print(f"Extracted features for sample #{index} of {posture_name} posture.")

        df_features = pd.DataFrame(new_data)

        directory = os.path.dirname(filename)
        basename = os.path.basename(filename)
        name, ext = os.path.splitext(basename)
        features_filename = str(os.path.join(directory, name + "_features" + ext))
        df_features.to_csv(features_filename, index=False)

    @staticmethod
    def extract_gesture_features(filename):
        df_gesture_data = pd.read_csv(filename)
        new_data = []

        for gesture_name in df_gesture_data["Gesture Name"].unique():
            gesture = df_gesture_data[df_gesture_data["Gesture Name"] == gesture_name]

            for sample_number, sample_data in gesture.groupby("Sample Number"):
                accel_data, quat_data = parse_gesture_sample(sample_data)

                for i in range(len(accel_data)):
                    features = extract_gesture_features(accel_data[i], quat_data[i])

                    new_row = {"Gesture Name": gesture_name, "Gesture Number": sample_data["Gesture Number"].iloc[0],
                               "Sample Number": sample_number}
                    new_row.update(features)
                    new_data.append(new_row)

                print(f"Extracted features for sample #{sample_number} of {gesture_name} gesture.")

        df_features = pd.DataFrame(new_data)

        directory = os.path.dirname(filename)
        basename = os.path.basename(filename)
        name, ext = os.path.splitext(basename)
        if "resampled_" in name:
            name = name.replace("resampled_", "")
        features_filename = str(os.path.join(directory, name + "_features" + ext))
        df_features.to_csv(features_filename, index=False)

    @staticmethod
    def resample_gesture_samples(filename):
        df_gesture_data = pd.read_csv(filename)

        resampled_data = []

        for gesture_name in df_gesture_data["Gesture Name"].unique():
            gesture = df_gesture_data[df_gesture_data["Gesture Name"] == gesture_name]

            for sample_number, sample_data in gesture.groupby("Sample Number"):
                accel_data = sample_data[["Acceleration_x", "Acceleration_y", "Acceleration_z"]].values
                quat_data = sample_data[["Rotation_i", "Rotation_j", "Rotation_k", "Rotation_real"]].values
                gesture_no = sample_data["Gesture Number"].iloc[0]

                resampled_samples = resample_gesture_samples(accel_data, quat_data, 100)

                for i in range(len(resampled_samples)):
                    resampled_accel = resampled_samples[i][0]
                    resampled_rot = resampled_samples[i][1]

                    resampled_data.append((
                        gesture_name, gesture_no, sample_number,
                        resampled_accel[0], resampled_accel[1], resampled_accel[2],
                        resampled_rot[0], resampled_rot[1], resampled_rot[2], resampled_rot[3]))

                print(f"Resampled sample #{sample_number} of {gesture_name} gesture.")

        # Convert the resampled data to a DataFrame
        df_resampled_data = pd.DataFrame(resampled_data,
                                         columns=["Gesture Name", "Gesture Number", "Sample Number",
                                                  "Acceleration_x", "Acceleration_y", "Acceleration_z",
                                                  "Rotation_i", "Rotation_j", "Rotation_k", "Rotation_real"])

        # Save the resampled data to a new CSV file
        directory = os.path.dirname(filename)
        basename = os.path.basename(filename)
        resampled_filename = os.path.join(directory, "resampled_" + basename)
        df_resampled_data.to_csv(resampled_filename, index=False)


class HandsOnGloveGUI:
    def __init__(self, glove_ref, root, posture_prediction_status, gesture_prediction_status):
        self.glove = glove_ref
        self.posture_prediction_status = posture_prediction_status
        self.gesture_prediction_status = gesture_prediction_status

        self.root = root
        root.title("Hands-on Glove")

        # Call the update_variable function initially to start the updates
        self.get_updates = False
        self._update_variables()

        self.main_frame = tk.Frame(self.root, padx=10, pady=10, borderwidth=2, relief="groove")
        self.main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.frame = ttk.Frame(self.main_frame, padding=10, borderwidth=2, relief="groove")
        self.frame.pack(side="left", padx=10, pady=10, fill="x", expand=True)

        self.sensor_button_frame = tk.Frame(self.main_frame, padx=10, pady=10, borderwidth=2, relief="groove")
        self.sensor_button_frame.pack(side="right", padx=(10, 10), pady=10, fill="x", expand=True)

        self.posture_button_frame = tk.Frame(self.main_frame, padx=10, pady=10, borderwidth=2, relief="groove")
        self.posture_button_frame.pack(side="right", padx=(10, 10), pady=10, fill="x", expand=True)

        self.gesture_button_frame = tk.Frame(self.main_frame, padx=10, pady=10, borderwidth=2, relief="groove")
        self.gesture_button_frame.pack(side="right", padx=(10, 10), pady=10, fill="x", expand=True)

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

        self.posture_prediction_button = tk.Button(self.posture_button_frame, text="Start Posture Prediction",
                                                   command=self._begin_posture_prediction, height=2, width=20)
        self.posture_prediction_button.pack(side=tk.TOP, padx=10, pady=4)

        self.gesture_prediction_button = tk.Button(self.gesture_button_frame, text="Start Gesture Prediction",
                                                   command=self._begin_gesture_prediction, height=2, width=20)
        self.gesture_prediction_button.pack(side=tk.TOP, padx=10, pady=4)

        self.posture_collect_sample_button = tk.Button(self.posture_button_frame, text="Collect Posture Samples",
                                                       command=self._collect_posture_samples, height=2, width=20)
        self.posture_collect_sample_button.pack(side=tk.TOP, padx=10, pady=4)

        self.gesture_collect_sample_button = tk.Button(self.gesture_button_frame, text="Collect Gesture Samples",
                                                       command=self._collect_gesture_samples, height=2, width=20)
        self.gesture_collect_sample_button.pack(side=tk.TOP, padx=10, pady=4)

        self.load_posture_samples_button = tk.Button(self.posture_button_frame, text="Load Posture Samples",
                                                     command=self._load_posture_samples, height=2, width=20)
        self.load_posture_samples_button.pack(side=tk.TOP, padx=10, pady=4)

        self.load_gesture_samples_button = tk.Button(self.gesture_button_frame, text="Load Gesture Samples",
                                                     command=self._load_gesture_samples, height=2, width=20)
        self.load_gesture_samples_button.pack(side=tk.TOP, padx=10, pady=4)

        self.posture_variations_button = tk.Button(self.posture_button_frame, text="Create Posture Variations",
                                                   command=self._create_posture_variations, height=2, width=20)
        self.posture_variations_button.pack(side=tk.TOP, padx=10, pady=4)

        self.gesture_variations_button = tk.Button(self.gesture_button_frame, text="Create Gesture Variations",
                                                   command=self._create_gesture_variations, height=2, width=20)
        self.gesture_variations_button.pack(side=tk.TOP, padx=10, pady=4)

        self.posture_features_button = tk.Button(self.posture_button_frame, text="Extract Posture Features",
                                                 command=self._extract_posture_features, height=2, width=20)
        self.posture_features_button.pack(side=tk.TOP, padx=10, pady=4)

        self.gesture_features_button = tk.Button(self.gesture_button_frame, text="Extract Gesture Features",
                                                 command=self._extract_gesture_features, height=2, width=20)
        self.gesture_features_button.pack(side=tk.TOP, padx=10, pady=4)

        self.train_svm_model_button = tk.Button(self.posture_button_frame, text="Train SVM Model",
                                                command=self._train_svm_model, height=2, width=20)
        self.train_svm_model_button.pack(side=tk.TOP, padx=10, pady=4)

        self.train_knn_model_button = tk.Button(self.gesture_button_frame, text="Train KNN Model",
                                                command=self._train_knn_model, height=2, width=20)
        self.train_knn_model_button.pack(side=tk.TOP, padx=10, pady=4)

        self.save_svm_model_button = tk.Button(self.posture_button_frame, text="Save SVM Model",
                                               command=self._save_svm_model, height=2, width=20)
        self.save_svm_model_button.pack(side=tk.TOP, padx=10, pady=4)

        self.save_knn_model_button = tk.Button(self.gesture_button_frame, text="Save KNN Model",
                                               command=self._save_knn_model, height=2, width=20)
        self.save_knn_model_button.pack(side=tk.TOP, padx=10, pady=4)

        self.load_svm_model_button = tk.Button(self.posture_button_frame, text="Load SVM Model",
                                               command=self._load_svm_model, height=2, width=20)
        self.load_svm_model_button.pack(side=tk.TOP, padx=10, pady=4)

        self.load_knn_model_button = tk.Button(self.gesture_button_frame, text="Load KNN Model",
                                               command=self._load_knn_model, height=2, width=20)
        self.load_knn_model_button.pack(side=tk.TOP, padx=10, pady=4)

        self.resample_gestures_button = tk.Button(self.gesture_button_frame, text="Resample Gesture Samples",
                                                  command=self._resample_gesture_samples, height=2, width=20)
        self.resample_gestures_button.pack(side=tk.TOP, padx=10, pady=4)

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

    def _draw_marker(self, position):
        self.joystick_canvas.delete("marker")
        joystick_x, joystick_y = position

        if self.glove.is_joystick_calibrated:
            pos_x = map_range_clamped(joystick_x, -1, 1, 0, 200)
            pos_y = 200 - map_range_clamped(joystick_y, -1, 1, 0, 200)
        else:
            pos_x = map_range_clamped(joystick_x, 0, 4096, 0, 200)
            pos_y = 200 - map_range_clamped(joystick_y, 0, 4096, 0, 200)
        self.joystick_canvas.create_oval(pos_x - 5, pos_y - 5, pos_x + 5, pos_y + 5,
                                         fill="red", tags="marker")

    @staticmethod
    def _destroy_window(event):
        event.widget.winfo_toplevel().destroy()

    def show_entry_box(self, title, message):
        answer = simpledialog.askstring(title, message, parent=self.root)

        if answer is not None:
            return answer
        else:
            return None

    @staticmethod
    def ask_yesno(title, message):
        popup = tk.Toplevel()
        popup.withdraw()

        answer = messagebox.askyesno(title, message)

        popup.destroy()
        return answer

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

    def _update_variables(self):
        if self.get_updates:
            accel_data = self.glove.acceleration
            acc_x, acc_y, acc_z = ["{:.4f}".format(x) for x in accel_data[:3]]
            acc_cal = int(accel_data[3])
            self.acc_x.set(acc_x)
            self.acc_y.set(acc_y)
            self.acc_z.set(acc_z)
            self.acc_cal.set(acc_cal)

            rot_data = self.glove.rotation
            quat_i, quat_j, quat_k, quat_w, quat_rad_cal = ["{:.4f}".format(x) for x in rot_data[:5]]
            quat_cal = int(rot_data[5])
            self.quat_i.set(quat_i)
            self.quat_j.set(quat_j)
            self.quat_k.set(quat_k)
            self.quat_w.set(quat_w)
            self.quat_rad_cal.set(quat_rad_cal)
            self.quat_cal.set(quat_cal)

            if self.glove.is_flex_calibrated and self.glove.flex_sensors_mapped:
                (flex_ring_tip, flex_middle_tip, flex_point_tip, flex_pinky, flex_ring_base, flex_middle_base,
                 flex_point_base, flex_thumb) = ["{:.2f}".format(x) for x in self.glove.flex_sensors_mapped]
            else:
                (flex_ring_tip, flex_middle_tip, flex_point_tip, flex_pinky, flex_ring_base,
                 flex_middle_base, flex_point_base, flex_thumb) = self.glove.flex_sensors

            self.flex_thumb.set(flex_thumb)
            self.flex_point_base.set(flex_point_base)
            self.flex_middle_base.set(flex_middle_base)
            self.flex_ring_base.set(flex_ring_base)
            self.flex_pinky.set(flex_pinky)
            self.flex_point_tip.set(flex_point_tip)
            self.flex_middle_tip.set(flex_middle_tip)
            self.flex_ring_tip.set(flex_ring_tip)

            if self.glove.is_joystick_calibrated and self.glove.joystick_mapped:
                joystick = self.glove.joystick_mapped
                self.joystick_x.set("{:.2f}".format(joystick[0]))
                self.joystick_y.set("{:.2f}".format(joystick[1]))
            else:
                joystick = self.glove.joystick
                self.joystick_x.set(joystick[0])
                self.joystick_y.set(joystick[1])
            self._draw_marker(joystick)

            battery_voltage = (self.glove.battery / 4095) * 4.3349
            battery_percentage = map_range_clamped(battery_voltage, 3.5, 4.0, 0, 100)
            self.battery_voltage.set("{:.2f}".format(battery_voltage))
            self.battery_percentage.set("{:.1f}".format(battery_percentage))

            self.root.after(50, self._update_variables)

        else:
            self.root.after(200, self._update_variables)

    def _toggle_streaming(self):
        global is_streaming, streaming_condition
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

    def _begin_joystick_calibration(self):
        self.glove.calibrate_joystick()

    def _begin_flex_calibration(self):
        self.glove.calibrate_flex_sensors()

    def _load_calibration(self):
        filename = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if filename:
            self.glove.load_calibration_values(filename)

    def _save_calibration(self):
        filename = filedialog.asksaveasfilename(filetypes=[("Pickle Files", "*.pkl")], defaultextension=".pkl")
        if filename:
            self.glove.save_calibration_values(filename)

    def _begin_posture_prediction(self):
        if self.glove.is_svm_ready:
            if not self.posture_prediction_status.value:
                self.posture_prediction_status.value = True
                self.posture_prediction_button.config(text="Stop Posture Prediction")
            else:
                self.posture_prediction_status.value = False
                self.posture_prediction_button.config(text="Start Posture Prediction")
        else:
            self.show_popup_message("Warning", "Train the model or load a saved SVM model first.")

    def _begin_gesture_prediction(self):
        if self.glove.is_knn_ready:
            if not self.gesture_prediction_status.value:
                self.gesture_prediction_status.value = True
                self.gesture_prediction_button.config(text="Stop Gesture Prediction")
            else:
                self.gesture_prediction_status.value = False
                self.gesture_prediction_button.config(text="Start Gesture Prediction")
        else:
            self.show_popup_message("Warning", "Train the model or load a saved KNN model first.")

    def _collect_posture_samples(self):
        self.glove.collect_posture_sample()

    def _collect_gesture_samples(self):
        self.glove.collect_gesture_sample()

    def _load_posture_samples(self):
        self.glove.load_posture_samples()

    def _load_gesture_samples(self):
        self.glove.load_gesture_samples()

    def _train_svm_model(self):
        try:
            self.glove.train_svm_model()
        except ValueError:
            self.show_popup_message("Error", "Collect or load samples first.")

    def _train_knn_model(self):
        try:
            self.glove.train_knn_model()
        except ValueError:
            self.show_popup_message("Error", "Collect or load samples first.")

    def _save_svm_model(self):
        filename = filedialog.asksaveasfilename(filetypes=[("Pickle Files", "*.pkl")], defaultextension=".pkl")
        if filename:
            self.glove.save_svm_model(filename)

    def _save_knn_model(self):
        filename = filedialog.asksaveasfilename(filetypes=[("Pickle Files", "*.pkl")], defaultextension=".pkl")
        if filename:
            self.glove.save_knn_model(filename)

    def _load_svm_model(self):
        filename = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if filename:
            self.glove.load_svm_model(filename)

    def _load_knn_model(self):
        filename = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if filename:
            self.glove.load_knn_model(filename)

    def _create_posture_variations(self):
        try:
            num_variations = int(self.show_entry_box("Variations", "Enter number of variations to generate"))
            self.glove.create_posture_variations(num_variations)
        except TypeError:
            pass

    def _create_gesture_variations(self):
        try:
            min_target_len = int(self.show_entry_box("Minimum", "Enter minimum target length"))
            max_target_len = int(self.show_entry_box("Maximum", "Enter maximum target length"))
            num_variations = int(self.show_entry_box("Variations", "Enter number of variations to generate"))

            if min_target_len and max_target_len:
                self.glove.create_gesture_variations(min_target_len, max_target_len, num_variations)
        except TypeError:
            pass

    def _extract_posture_features(self):
        filename = filedialog.askopenfilename(filetypes=[("Comma-Separated Values", "*.csv")], defaultextension=".csv")

        if filename:
            self.glove.extract_posture_features(filename)

    def _extract_gesture_features(self):
        filename = filedialog.askopenfilename(filetypes=[("Comma-Separated Values", "*.csv")], defaultextension=".csv")

        if filename:
            self.glove.extract_gesture_features(filename)

    def _resample_gesture_samples(self):
        filename = filedialog.askopenfilename(filetypes=[("Comma-Separated Values", "*.csv")], defaultextension=".csv")
        if filename:
            self.glove.resample_gesture_samples(filename)


def map_range(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return 0.0
    return float((((value - in_min) / (in_max - in_min)) * (out_max - out_min)) + out_min)


def map_range_clamped(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return 0.0
    return float(max(min(out_max, (((value - in_min) / (in_max - in_min)) * (out_max - out_min)) + out_min), out_min))


def calculate_relative_orientation(starting_quaternion, current_quaternion):
    starting = Quaternion(starting_quaternion)
    current = Quaternion(current_quaternion)

    relative = current * starting.inverse
    relative = relative.normalised

    return relative.elements


def calculate_angular_velocity(gyro_quaternion_current, gyro_quaternion_previous, dt):
    gyro_quaternion_current = Quaternion(gyro_quaternion_current)
    gyro_quaternion_previous = Quaternion(gyro_quaternion_previous)
    quaternion_derivative = (gyro_quaternion_current - gyro_quaternion_previous) / dt

    angular_velocity = 2 * quaternion_derivative * gyro_quaternion_current.conjugate
    return abs(angular_velocity)


def extract_posture_features(flex_data):
    entropy_value = entropy(np.abs(flex_data))
    if np.isnan(entropy_value):
        entropy_value = 0.0

    features = {f"mean": np.mean(flex_data), f"std": np.std(flex_data),
                f"mad": np.median(np.abs(flex_data - np.median(flex_data))), f"min": np.min(flex_data),
                f"max": np.max(flex_data), f"entropy": entropy_value}

    # Prevent potential data loss errors
    if np.std(flex_data) > 1e-6:
        features[f"skew"] = skew(flex_data)
        features[f"kurtosis"] = kurtosis(flex_data)
    else:
        features[f"skew"] = 0.0
        features[f"kurtosis"] = 0.0

    fft_result = np.abs(fft(flex_data))
    fft_entropy_value = entropy(fft_result)
    if np.isnan(fft_entropy_value):
        fft_entropy_value = 0.0

    features[f"fft_mean"] = np.mean(fft_result)
    features[f"fft_std"] = np.std(fft_result)
    features[f"fft_mad"] = np.median(np.abs(fft_result - np.median(fft_result)))
    features[f"fft_entropy"] = fft_entropy_value

    return features


def extract_gesture_features(acceleration, quaternion):
    features = {}
    accel_axes = ["x", "y", "z"]
    quat_axes = ["i", "j", "k", "real"]

    for i, axis in enumerate(accel_axes):
        axis_data = acceleration[:, i]

        features[f"Acceleration_{axis}_mean"] = np.mean(axis_data)
        features[f"Acceleration_{axis}_std"] = np.std(axis_data)
        features[f"Acceleration_{axis}_mad"] = np.median(np.abs(axis_data - np.median(axis_data)))
        features[f"Acceleration_{axis}_min"] = np.min(axis_data)
        features[f"Acceleration_{axis}_max"] = np.max(axis_data)
        features[f"Acceleration_{axis}_entropy"] = entropy(np.abs(axis_data))
        features[f"Acceleration_{axis}_skew"] = skew(axis_data)
        features[f"Acceleration_{axis}_kurtosis"] = kurtosis(axis_data)

        fft_result = np.abs(fft(axis_data))
        features[f"Acceleration_{axis}_fft_mean"] = np.mean(fft_result)
        features[f"Acceleration_{axis}_fft_std"] = np.std(fft_result)
        features[f"Acceleration_{axis}_fft_mad"] = np.median(np.abs(fft_result - np.median(fft_result)))
        features[f"Acceleration_{axis}_fft_entropy"] = entropy(fft_result)

    for i, axis in enumerate(quat_axes):
        axis_data = quaternion[:, i]

        features[f"Quaternion_{axis}_mean"] = np.mean(axis_data)
        features[f"Quaternion_{axis}_std"] = np.std(axis_data)
        features[f"Quaternion_{axis}_mad"] = np.median(np.abs(axis_data - np.median(axis_data)))
        features[f"Quaternion_{axis}_min"] = np.min(axis_data)
        features[f"Quaternion_{axis}_max"] = np.max(axis_data)
        features[f"Quaternion_{axis}_entropy"] = entropy(np.abs(axis_data))
        features[f"Quaternion_{axis}_skew"] = skew(axis_data)
        features[f"Quaternion_{axis}_kurtosis"] = kurtosis(axis_data)

        fft_result = np.abs(fft(axis_data))
        features[f"Quaternion_{axis}_fft_mean"] = np.mean(fft_result)
        features[f"Quaternion_{axis}_fft_std"] = np.std(fft_result)
        features[f"Quaternion_{axis}_fft_mad"] = np.median(np.abs(fft_result - np.median(fft_result)))
        features[f"Quaternion_{axis}_fft_entropy"] = entropy(fft_result)

    return features


def resample_gesture_samples(accel, rot, num_gesture_samples=100):
    resampled_samples = []

    for i in range(num_gesture_samples):
        # Calculate the index in the original data for interpolation
        idx = i * (len(accel) - 1) / (num_gesture_samples - 1)

        # Interpolate acceleration data
        interp_accel_x = interp1d(np.arange(len(accel)), accel[:, 0])
        interp_accel_y = interp1d(np.arange(len(accel)), accel[:, 1])
        interp_accel_z = interp1d(np.arange(len(accel)), accel[:, 2])

        new_accel_x = interp_accel_x(idx)
        new_accel_y = interp_accel_y(idx)
        new_accel_z = interp_accel_z(idx)

        # Interpolate quaternion data
        interp_quat_i = interp1d(np.arange(len(rot)), rot[:, 0])
        interp_quat_j = interp1d(np.arange(len(rot)), rot[:, 1])
        interp_quat_k = interp1d(np.arange(len(rot)), rot[:, 2])
        interp_quat_real = interp1d(np.arange(len(rot)), rot[:, 3])

        new_quat_i = interp_quat_i(idx)
        new_quat_j = interp_quat_j(idx)
        new_quat_k = interp_quat_k(idx)
        new_quat_real = interp_quat_real(idx)

        resampled_samples.append(([new_accel_x, new_accel_y, new_accel_z],
                                  [new_quat_i, new_quat_j, new_quat_k, new_quat_real]))
    return resampled_samples


def parse_gesture_sample(sample_data):
    num_samples = len(sample_data)
    accel_data = np.zeros((num_samples, 3))
    quat_data = np.zeros((num_samples, 4))

    for i, (_, row) in enumerate(sample_data.iterrows()):
        accel_data[i] = row[["Acceleration_x", "Acceleration_y", "Acceleration_z"]]
        quat_data[i] = row[["Rotation_i", "Rotation_j", "Rotation_k", "Rotation_real"]]

    # Reshape data into 100-sample windows
    accel_data = accel_data.reshape(-1, 100, 3)
    quat_data = quat_data.reshape(-1, 100, 4)

    return accel_data, quat_data


def main():
    process_termination_event = mp.Event()
    posture_queue = mp.Queue()
    gesture_queue = mp.Queue()

    is_predicting_postures = mp.Value(ctypes.c_bool, False)
    is_predicting_gestures = mp.Value(ctypes.c_bool, False)
    posture_result = mp.Value(ctypes.c_int, 0)
    gesture_result = mp.Value(ctypes.c_int, 0)
    flex_sensor_readings = mp.Array(ctypes.c_float, [0] * 8)
    acceleration_readings = mp.Array(ctypes.c_float, [0] * 3)
    rotation_readings = mp.Array(ctypes.c_float, [0] * 4)

    predict_posture = mp.Process(target=predict_posture_process,
                                 args=(posture_queue, is_predicting_postures, posture_result, flex_sensor_readings),
                                 daemon=True)
    child_processes.append(predict_posture)
    predict_posture.start()

    predict_gesture = mp.Process(target=predict_gesture_process,
                                 args=(gesture_queue, is_predicting_gestures, gesture_result,
                                       acceleration_readings, rotation_readings),
                                 daemon=True)
    child_processes.append(predict_gesture)
    predict_gesture.start()

    core = mp.Process(target=core_process,
                      args=(process_termination_event, posture_queue, gesture_queue,
                            flex_sensor_readings, acceleration_readings, rotation_readings,
                            posture_result, is_predicting_postures, gesture_result, is_predicting_gestures),
                      daemon=True)
    child_processes.append(core)
    core.start()

    while True:
        if process_termination_event.is_set():
            for active_process in child_processes:
                active_process.terminate()
                active_process.join()
            sys.exit()
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        for process in child_processes:
            process.terminate()
            process.join()
        sys.exit()
