# Bare-bones version for receiving glove sensor data over UDP and
# sending them out to other devices on the network over OSC and vice versa

import socket
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server
from threading import Thread
import pandas as pd
import numpy as np
from pyquaternion import Quaternion
from scipy.interpolate import interp1d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


# Initialize variables
gestures = {}
X = []
y = []

# Broadcast OSC data to each device on the same network.
# Change the remote_IP to send it to a specific device.
local_IP = "127.0.0.1"
remote_IP = "192.168.0.255"

udp_receive_port = 48001
udp_send_port = 48002
osc_receive_port = 48003
osc_send_port = 48004

# Bind to a socket and allow for communication even if it is being used
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("", udp_receive_port))
except Exception as e:
    print(f"Socket error: {e}")

# Create OSC client and dispatcher
client = udp_client.SimpleUDPClient(local_IP, osc_send_port)
dispatcher = dispatcher.Dispatcher()


# Thread that sends OSC data to other devices on the network
def osc_listen_thread():
    while True:
        udp_data = get_udp_data()

        acceleration = get_acceleration(udp_data)
        client.send_message("/acceleration", acceleration)
        client.send_message("/rotation", get_rotation(udp_data))

        flex = [float(i) for i in get_flex_sensors(udp_data)]
        client.send_message("/flex", flex)

        joystick = [float(i) for i in get_joystick(udp_data)]
        client.send_message("/joystick", joystick)

        client.send_message("/button", get_buttons(udp_data))
        client.send_message("/battery", get_battery(udp_data))
        client.send_message("/temperature", get_temperature(udp_data))


# Thread that listens to incoming OSC data
def osc_dispatch_thread():
    server = osc_server.ThreadingOSCUDPServer((local_IP, osc_receive_port), dispatcher)
    server.serve_forever()


# Take OSC data regarding the onboard LED from UE5 or Max,
# convert them into UDP messages in the format the glove can parse
def rgb_handler(unused_addr, red, green, blue):
    sock.sendto(f"/rgb|{red}|{green}|{blue}|0|0|".encode("utf-8"), (remote_IP, udp_send_port))
    return


# Take OSC data regarding the onboard vibration motor from UE5 or Max,
# convert them into UDP messages in the format the glove can parse
def haptic_handler(unused_addr, value):
    sock.sendto(f"/haptic|{value}|0|0|0|0|".encode("utf-8"), (remote_IP, udp_send_port))
    return


# Register handlers on dispatcher
# Call corresponding functions whenever a message is sent to one of the addresses
dispatcher.map("/rgb", rgb_handler)
dispatcher.map("/haptic", haptic_handler)


# Receive UDP data
def get_udp_data():
    udp_data = sock.recv(255).decode("utf-8")
    udp_data = udp_data.split()
    return udp_data


# Parse acceleration readings
# |------------------------------------------------------------|
# |   x   |   y   |   z   |  calibration status (3 is the max) |
# | float | float | float |               int                  |
# |------------------------------------------------------------|
def get_acceleration(udp_data):
    return [float(x) for x in udp_data[1:4]] + [int(udp_data[4])]


# Parse rotation readings
# |------------------------------------------------------------------------------------|
# |   i   |   j   |   k   |   w   | error (radian) | calibration status (3 is the max) |
# | float | float | float | float |     float      |              int                  |
# |------------------------------------------------------------------------------------|
def get_rotation(udp_data):
    return [float(x) for x in udp_data[6:11]] + [int(udp_data[11])]


# Parse flex sensors
# ADC on the other end is 12-bit
def get_flex_sensors(udp_data):
    return [int(x) for x in udp_data[13:21]]


# Parse joystick readings
# ADC on the other end is 12-bit
def get_joystick(udp_data):
    return [int(x) for x in udp_data[22:24]]


# Parse buttons
# 0 is off, 1 is on
def get_buttons(udp_data):
    return [int(x) for x in udp_data[25:27]]


# Parse battery readings
# ADC on the other end is 12-bit
def get_battery(udp_data):
    return int(udp_data[28])


# Parse temperature readings
# Data from the onboard temperature sensor
def get_temperature(udp_data):
    return float(udp_data[30][:7])


if __name__ == "__main__":
    osc_listen = Thread(target=osc_listen_thread, args=(), daemon=True)
    osc_dispatch = Thread(target=osc_dispatch_thread, args=(), daemon=True)

    osc_listen.start()
    osc_dispatch.start()

    while True:
        current_sensor_data = get_udp_data()
        print(get_acceleration(current_sensor_data))
        print(get_rotation(current_sensor_data))
        print(get_flex_sensors(current_sensor_data))
        print(get_joystick(current_sensor_data))
        print(get_buttons(current_sensor_data))
        print(get_battery(current_sensor_data))
        print(get_temperature(current_sensor_data))
