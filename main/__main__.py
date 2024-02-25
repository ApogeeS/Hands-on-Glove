import ctypes
import pickle
import socket
import sys
import time
import threading
import multiprocessing as mp
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler
# from scipy.interpolate import interp1d
# from collections import deque
from sklearnex import patch_sklearn
patch_sklearn()
# Must import algorithms after patching to use the Intel implementation
from sklearn import svm


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
                if is_predicting_postures:
                    for name, number in glove_ref.postures.items():
                        if posture_result.value == number:
                            client.send_message("/posture/name", name)
                            client.send_message("/posture/number", number)

                if is_predicting_gestures:
                    for name, number in glove_ref.gestures.items():
                        if gesture_result.value == number:
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
    gui = HandsOnGloveGUI(glove, root, is_predicting_postures)
    glove.gui = gui
    root.protocol("WM_DELETE_WINDOW",
                  lambda: exit_handler(thread_termination_event, process_termination_event, core_threads))
    root.mainloop()


def predict_posture_process(posture_queue, prediction_status, posture_result, flex_sensor_readings):
    __svm_clf = None
    __svm_mm_scaler = None
    # TODO fine tune after extracting features
    __svm_confidence_threshold = 0.5

    while True:
        if prediction_status.value:
            while not posture_queue.empty():
                __svm_clf, __svm_mm_scaler = posture_queue.get()

            while prediction_status.value:
                sensor_data = __svm_mm_scaler.transform([flex_sensor_readings[:]])

                # Make predictions using the SVM model
                decision_scores = __svm_clf.decision_function(sensor_data)
                normalized_scores = normalize(decision_scores, norm="l1", axis=1)

                max_index = np.argmax(normalized_scores) + 1
                max_score = np.max(normalized_scores)

                if max_score < __svm_confidence_threshold:
                    posture_result.value = -1  # -1 represents "not doing any trained posture"
                else:
                    posture_result.value = max_index
                time.sleep(0.01)


# TODO implementation
def predict_gesture_process(gesture_queue, prediction_status, gesture_result, acceleration, rotation):
    pass


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

        self._remote_IP = REMOTE_IP
        self._udp_send_port = UDP_SEND_PORT

        self.is_svm_ready = False
        self.postures = {}
        self.is_posture_data_loaded = False
        self.__svm_clf = None
        self.__svm_mm_scaler = None
        self.posture_queue = posture_queue

        self.is_knn_ready = False
        self.gestures = {}
        self.is_gesture_data_loaded = False
        self.__knn_clf = None
        self.__knn_mm_scaler = None
        self.gesture_queue = gesture_queue

        self.gui = gui

        # Initialize column structures for sample data collection
        self._posture_columns = ["Posture Name", "Posture Number",
                                 "Flex_1", "Flex_2", "Flex_3", "Flex_4", "Flex_5", "Flex_6", "Flex_7", "Flex_8"]
        self.df_posture = pd.DataFrame(columns=self._posture_columns)

        self._gesture_columns = ["Gesture Name", "Gesture Number",
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
            __open_values = self.flex_sensors

            self.gui.show_popup_message("Flex Sensor Calibration", "Close your fist and click OK...")
            __closed_values = self.flex_sensors

            # Calculate calibration values for each sensor
            for sensor_num in range(8):
                __open_value = __open_values[sensor_num]
                __closed_value = __closed_values[sensor_num]

                # Store the calibration values
                self._flex_calibration[sensor_num]["open"] = __open_value
                self._flex_calibration[sensor_num]["closed"] = __closed_value

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
        __posture_name = ""
        __posture_number = 0

        if not self.is_flex_calibrated:
            self.gui.show_popup_message("Warning", "Calibrate flex sensors or load calibration file first.")
        else:
            csv_filename = filedialog.asksaveasfilename(filetypes=[("Comma-Separated Values", "*.csv")],
                                                        defaultextension=".csv")
            new_sample = False
            if not new_sample:
                if not self.is_posture_data_loaded:
                    __posture_name = self.gui.show_entry_box("Posture name", "Type a posture name into the box.")
                else:
                    __posture_name = self.gui.show_entry_box("Posture name",
                                                             f"Type a posture name into the box.\n"
                                                             f"Below postures exists in the loaded file\n"
                                                             f"{[key for key in self.postures.keys()]}")
                if __posture_name is not None:
                    new_sample = True
                    if self.postures:
                        if __posture_name in self.postures:
                            __posture_number = self.postures[__posture_name]
                        else:
                            last_posture = max(self.postures.values())
                            __posture_number = last_posture + 1
                if __posture_name is None:
                    self.gui.show_popup_message("Error", "Incorrect entry. Sample collection cancelled.")

            if csv_filename:
                while new_sample:
                    self.gui.show_popup_message("Save",
                                                f"Click OK to save a new sample for "
                                                f"{__posture_number} - {__posture_name}.")
                    __posture_data = self.flex_sensors_mapped
                    self.posture_add_row(__posture_name, __posture_number, __posture_data)
                    new_sample = self.gui.ask_yesno("Continue?",
                                                    "Do you want to collect more data for the same posture?")

                    while not new_sample:
                        another_posture = self.gui.ask_yesno("New posture?",
                                                             "Do you want to collect data for a new posture?")

                        if another_posture:
                            if not self.is_posture_data_loaded:
                                __posture_name = self.gui.show_entry_box("Posture name",
                                                                         "Type a posture name into the box.")
                                __posture_number += 1
                            else:
                                __posture_name = self.gui.show_entry_box("Posture name",
                                                                         f"Type a posture name into the box.\n"
                                                                         f"Below postures exists in the loaded file\n"
                                                                         f"{[key for key in self.postures.keys()]}")
                            if __posture_name is not None:
                                new_sample = True
                                if self.postures:
                                    if __posture_name in self.postures:
                                        __posture_number = self.postures[__posture_name]
                                    else:
                                        last_posture = max(self.postures.values())
                                        __posture_number = last_posture + 1
                        else:
                            break
                if __posture_name is not None:
                    self.df_posture.to_csv(csv_filename, index=False)
                    self.gui.show_popup_message("Success!", "Sample collection finished.")

    # TODO don't forget to use relative orientation for gyro samples
    def collect_gesture_sample(self):
        pass

    # Load posture samples from a csv
    def load_posture_samples(self):
        csv_filename = filedialog.askopenfilename(filetypes=[("Comma-separated Values", "*.csv")],
                                                  defaultextension=".csv")

        if csv_filename:
            self.df_posture = pd.read_csv(csv_filename, sep=",")
            for label, number in zip(self.df_posture["Posture Name"], self.df_posture["Posture Number"]):
                if label not in self.postures:
                    self.postures[label] = number

            if len(self.postures) > 0:
                self.is_posture_data_loaded = True
                self.gui.show_popup_message("Success!", "Data loaded successfully.")
            else:
                self.gui.show_popup_message("Error", "Empty or corrupted data. Try loading from another file.")

    def train_svm_model(self):
        try:
            self.__svm_mm_scaler = MinMaxScaler()
            _predict = "Posture Number"

            data = self.df_posture.drop(columns="Posture Name")
            _X = np.array(data.drop(columns=_predict))
            _y = np.array(data[_predict])

            _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2)

            _X_train = self.__svm_mm_scaler.fit_transform(_X_train)
            _X_test = self.__svm_mm_scaler.fit_transform(_X_test)

            self.__svm_clf = svm.SVC(kernel="linear", decision_function_shape="ovr")
            self.__svm_clf.fit(_X_train, _y_train)

            _y_pred = self.__svm_clf.predict(_X_test)

            accuracy = accuracy_score(_y_test, _y_pred)
            report = classification_report(_y_test, _y_pred)
            print(f"Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")

            # Flush queue so that only the last trained model is sent through
            if not self.posture_queue.empty():
                while not self.posture_queue.empty():
                    self.posture_queue.get()

            self.posture_queue.put((self.__svm_clf, self.__svm_mm_scaler))
            self.is_svm_ready = True
        except ValueError:
            self.gui.show_popup_message("Error", "Collect or load samples first.")

    def save_svm_model(self, filename):
        if self.is_svm_ready:
            with open(filename, "wb") as file:
                pickle.dump((self.__svm_clf, self.__svm_mm_scaler, self.postures), file)
            self.gui.show_popup_message("Success!", f"SVM Model saved to {filename}")
        else:
            self.gui.show_popup_message("Warning", "Train the model or load a saved SVM model first.")

    def load_svm_model(self, filename):
        with open(filename, "rb") as file:
            try:
                self.__svm_clf, self.__svm_mm_scaler, self.postures = pickle.load(file)
                # Flush queue so that only the last trained model is sent through
                if not self.posture_queue.empty():
                    while not self.posture_queue.empty():
                        self.posture_queue.get()
                self.posture_queue.put((self.__svm_clf, self.__svm_mm_scaler))
                self.is_svm_ready = True
                self.gui.show_popup_message("Success!", f"SVM Model loaded.")
            except ValueError:
                self.gui.show_popup_message("Error", "The selected file is not a saved SVM model.")

    # TODO implement KNN
    def train_knn_model(self):
        try:
            pass
        except ValueError:
            self.gui.show_popup_message("Error", "Collect or load samples first.")

    # TODO implement save
    def save_knn_model(self, filename):
        if self.is_knn_ready:
            with open(filename, "wb") as file:
                pickle.dump((self.__knn_clf, self.__knn_mm_scaler, self.gestures), file)
            self.gui.show_popup_message("Success!", f"KNN Model saved to {filename}")
        else:
            self.gui.show_popup_message("Warning", "Train the model or load a saved KNN model first.")

    # TODO implement load
    def load_knn_model(self, filename):
        with open(filename, "rb") as file:
            try:
                self.__knn_clf, self.__knn_mm_scaler, self.gestures = pickle.load(file)
                # Flush queue so that only the last trained model is sent through
                if not self.gesture_queue.empty():
                    while not self.gesture_queue.empty():
                        self.gesture_queue.get()
                self.gesture_queue.put((self.__knn_clf, self.__knn_mm_scaler))
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
    def gesture_add_row(self, gesture_label, gesture_no, acceleration_values, rotation_values):
        new_row = {"Gesture Name": gesture_label, "Gesture Number": gesture_no,
                   **{f"Acceleration_{axis}": float(val) for axis, val in ["x", "y", "z"]},
                   **{f"Rotation_{axis}": float(val) for axis, val in ["i", "j", "k", "real"]}}
        self.df_gesture.loc[len(self.df_gesture)] = new_row

    # TODO implement variations for both posture and gesture data
    # def create_variations(samples, num_variations, min_target_length, max_target_length):
    #     variations = []
    #     for _ in range(num_variations):
    #         new_sample = []
    #         for quaternion, acceleration in samples:
    #             # Randomly choose target length between min and max values
    #             target_length = np.random.randint(min_target_length, max_target_length + 1)
    #
    #             # Interpolate quaternion and acceleration data for longer duration
    #             interp_quaternion_long = interp1d(np.linspace(0, 1, len(quaternion)), quaternion)
    #             interp_acceleration_long = interp1d(np.linspace(0, 1, len(acceleration)), acceleration)
    #             new_quaternion_long = interp_quaternion_long(np.linspace(0, 1, target_length))
    #             new_acceleration_long = interp_acceleration_long(np.linspace(0, 1, target_length))
    #             new_sample.append((new_quaternion_long, new_acceleration_long))
    #
    #             # Interpolate quaternion and acceleration data for shorter duration
    #             interp_quaternion_short = interp1d(np.linspace(0, 1, len(quaternion)), quaternion)
    #             interp_acceleration_short = interp1d(np.linspace(0, 1, len(acceleration)), acceleration)
    #             new_quaternion_short = interp_quaternion_short(np.linspace(0, 1, int(target_length / 2)))
    #             new_acceleration_short = interp_acceleration_short(np.linspace(0, 1, int(target_length / 2)))
    #             new_sample.append((new_quaternion_short, new_acceleration_short))
    #
    #         variations.append(new_sample)
    #
    #     return variations
    #
    #
    # # Example usage
    # num_variations = 5  # Number of variations per sample
    # min_target_length = 40  # Minimum target length for interpolation
    # max_target_length = 60  # Maximum target length for interpolation
    # augmented_gestures = {}
    # for gesture_name, samples in gestures.items():
    #     augmented_samples = create_variations(samples, num_variations, min_target_length, max_target_length)
    #     augmented_gestures[gesture_name] = augmented_samples
    #
    #
    # # Save augmented samples to a CSV file
    # for gesture_name, augmented_samples_list in augmented_gestures.items():
    #     for i, augmented_samples in enumerate(augmented_samples_list):
    #         for j, (quaternion, acceleration) in enumerate(augmented_samples):
    #             row = {'gesture': f"{gesture_name}_variation_{i+1}_sample_{j+1}"}
    #             for k, q in enumerate(quaternion):
    #                 row[f'q{k}'] = q
    #             for k, a in enumerate(acceleration):
    #                 row[f'a{k}'] = a
    #             X.append(row)
    #             y.append(gesture_name)
    #
    # # Create a DataFrame and save to CSV
    # df = pd.DataFrame(X)
    # df.to_csv('gesture_samples.csv', index=False)


class HandsOnGloveGUI:
    def __init__(self, glove_ref, root, prediction_status):
        self.glove = glove_ref
        self.prediction_status = prediction_status

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

        self.posture_prediction_button = tk.Button(self.ml_button_frame, text="Start Posture Prediction",
                                                   command=self._begin_posture_prediction, height=2, width=20)
        self.posture_prediction_button.pack(side=tk.TOP, padx=10, pady=4)

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

    def _load_posture_samples(self):
        self.glove.load_posture_samples()

    def _begin_joystick_calibration(self):
        self.glove.calibrate_joystick()

    def _begin_flex_calibration(self):
        self.glove.calibrate_flex_sensors()

    def _collect_posture_samples(self):
        self.glove.collect_posture_sample()

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
            if not self.prediction_status.value:
                self.prediction_status.value = True
                self.posture_prediction_button.config(text="Stop Posture Prediction")
            else:
                self.prediction_status.value = False
                self.posture_prediction_button.config(text="Start Posture Prediction")
        else:
            self.show_popup_message("Warning", "Train the model or load a saved SVM model first.")

    def _train_svm_model(self):
        try:
            self.glove.train_svm_model()
        except ValueError:
            self.show_popup_message("Error", "Collect or load samples first.")

    def _save_svm_model(self):
        filename = filedialog.asksaveasfilename(filetypes=[("Pickle Files", "*.pkl")], defaultextension=".pkl")
        if filename:
            self.glove.save_svm_model(filename)

    def _load_svm_model(self):
        filename = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if filename:
            self.glove.load_svm_model(filename)


def map_range(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return 0.0
    return float((((value - in_min) / (in_max - in_min)) * (out_max - out_min)) + out_min)


def map_range_clamped(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return 0.0
    return float(max(min(out_max, (((value - in_min) / (in_max - in_min)) * (out_max - out_min)) + out_min), out_min))


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
        # Terminate main process
        sys.exit()

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
