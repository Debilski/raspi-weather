#!/usr/bin/env python3

import functools
import itertools
import math
import subprocess
import time

import zmq

import opc

import Adafruit_DHT as DHT

numLEDs = 512
client = opc.Client('localhost:7890')

PIXELS_HUMIDITY = range(29)
PIXELS_TEMPERATURE = range(30, 59)


def set_pixels(pixels, value, min_val, max_val):
    l = len(pixels)
    val_norm = value / (max_val - min_val)
    if not val_norm > 0:
        val_norm = 0
    if val_norm > 1:
        val_norm = 1
    return pixels[: int(val_norm * l)]


def cache_decorator(f):
    f.__last_call = 0
    f.__cache_time = 60
    f.__cache_val = None

    @functools.wraps(f)
    def wrapper():
        if time.monotonic() - f.__last_call > f.__cache_time:
            f.__last_call = time.monotonic()
            f.__cache_val = f()
        return f.__cache_val
    return wrapper


@cache_decorator
def check_usb_port_mensa():
    if not subprocess.call('grep 1d50 /sys/bus/usb/devices/1-1.2/idVendor', shell=True):
        return True
    else:
        return False


@cache_decorator
def check_usb_port_silent():
    if not subprocess.call('grep 1d50 /sys/bus/usb/devices/1-1.5/idVendor', shell=True):
        return True
    else:
        return False


def init():
    for i in range(10):
        client.put_pixels([(20, 200, 20)] * numLEDs)
        time.sleep(0.05)
        client.put_pixels([(0, 0, 0)] * numLEDs)
        time.sleep(0.05)


init()

t = 0
while True:
    t += 0.4

    humidity, temperature = DHT.read_retry(DHT.DHT11, 4)

    hum_pixels = list(set_pixels(PIXELS_HUMIDITY, humidity, 0, 100))
    temp_pixels = list(set_pixels(PIXELS_TEMPERATURE, temperature, -10, 35))
    frame = [(0, 0, 0)] * numLEDs

    if not check_usb_port_silent():
        for p in (hum_pixels + temp_pixels):
            frame[p] = (255, 255, 255)

    DO_MENSA = check_usb_port_mensa()
    if DO_MENSA:
        mensa_pixels = list(set_pixels(range(60), people_in_mensa(), 0, 400))
        for p in mensa_pixels:
            oldframe = frame[p]
            frame[p] = (200, 20, oldframe[2])

    orange = (255, 255, 0)

    h_scaled = humidity / 100
    orange_val = (orange[0] * h_scaled, orange[1]
                  * h_scaled, orange[2] * h_scaled)

    frame[29] = orange_val

    t_scaled = temperature / (35 + 10)
    orange_val = (orange[0] * t_scaled, orange[1]
                  * t_scaled, orange[2] * t_scaled)

    frame[59] = orange_val

    client.put_pixels(frame)
    time.sleep(0.05)
