#!/usr/bin/env python3

import functools
import itertools
import json
import math
from collections import namedtuple
import subprocess
import time

import zmq

import opc

numLEDs = 512
client = opc.Client('localhost:7890')

PIXELS_TOTAL = set(range(2 * 64))

PIXELS_HUMIDITY = range(29)
PIXELS_TEMPERATURE = range(64 + 30, 64 + 59)

PIXELS_BASE = {2, 3, 4, 8}

LIGHTS_CONFIG_FILE = 'lights-config.json'

try:
    with open(LIGHTS_CONFIG_FILE, 'r') as cfg:
        CONFIG = json.load(cfg)
except (FileNotFoundError, ValueError) as e:
    print(e)
    print("Using default values for config")

    CONFIG = {
        "COLOR": (255, 47, 1), # opc.hex_to_rgb('#ffcc33'),
        "NUM": 10
    }


def rgb_to_hsv(rgb):
    r, g, b = rgb
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    MAX = max(r, g, b)
    MIN = min(r, g, b)
    if MAX == MIN:
        h = 0
    elif MAX == r:
        h = 60 * (0 + (g - b) / (MAX -  MIN))
    elif MAX == g:
        h = 60 * (2 + (b - r) / (MAX -  MIN))
    elif MAX == b:
        h = 60 * (4 + (r - g) / (MAX -  MIN))
    if h < 0:
        h += 360
    if MAX == 0:
        s = 0
    else:
        s = (MAX - MIN) / MAX
    v = MAX
    return (h, s, v)

def hsv_to_rgb(hsv):
    h, s, v = hsv
    v = max(0, min(255, v))

    h_i = h // 60
    f = h / 60 - h_i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    if h_i == 0 or h_i == 6:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return r, g, b 

def dim_pixel(rgb, delt_v):
    h, s, v = rgb_to_hsv(rgb)
    return hsv_to_rgb((h, s, v + delt_v))


def map_pixels(pixels):
    fill = [(0, 0, 0)] * 64 * 4
    return pixels[0:64] + fill + pixels[64:64*2]

DHDataRaw = namedtuple('DHDataRaw', ['temp', 'hum', 'wind', 'rain'])
DHData = namedtuple('DHData', ['temp', 'hum', 'wind', 'rain_deriv'])

def get_closest_data(conn, date):
    for row in conn.execute('''
        SELECT * FROM weather
        ORDER BY ABS( strftime( "%s", date ) - strftime( "%s", ? ) ) ASC
        ''', (date, )):
        return row

def get_data_in_timerange(conn, start, end):
    return list(conn.execute('''
        SELECT * FROM weather
        WHERE date BETWEEN ? and ?
        ''', (start, end)))


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

def parse_msg(msg):
    if "CHANGE_COLOR" in msg:
        r, g, b = msg["CHANGE_COLOR"]
        old_r, old_g, old_b = CONFIG["COLOR"]
        CONFIG["COLOR"] = (old_r + r, old_g + g, old_b + b)
        return CONFIG["COLOR"]

    if "CHANGE_NUM_LEDS" in msg:
        diff = msg["CHANGE_NUM_LEDS"]
        old_num = CONFIG["NUM"]
        CONFIG["NUM"] = old_num + diff
        return CONFIG["NUM"]

    if "DIM_LEDS" in msg:
        mult = msg["DIM_LEDS"]
        rgb = CONFIG["COLOR"]
        CONFIG["COLOR"] = dim_pixel(rgb, mult)
        return CONFIG["COLOR"]


    if "SAVE_CONFIG" in msg:
        with open(LIGHTS_CONFIG_FILE, 'w') as outfile:
            json.dump(jsonData, outfile, sort_keys = True, indent = 4, ensure_ascii = False)



def init():
    for i in range(10):
        client.put_pixels(map_pixels([(20, 200, 20)] * numLEDs))
        time.sleep(0.05)
        client.put_pixels(map_pixels([(0, 0, 0)] * numLEDs))
        time.sleep(0.05)

def main(socket):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.bind(socket)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    init()

    t = 0
    while True:
        socks = dict(poller.poll(timeout=0))
        if socks and sock in socks and socks[sock] == zmq.POLLIN:
            try:
                msg = sock.recv_json()
                res = parse_msg(msg)
                sock.send_json(res)
            except json.decoder.JSONDecodeError:
                print("Bad json msg. Ignoring.")

        t += 0.4

        humidity, temperature = 29, 20 # DHT.read_retry(DHT.DHT11, 4)

        #hum_pixels = list(set_pixels(PIXELS_HUMIDITY, humidity, 0, 100))
        #temp_pixels = list(set_pixels(PIXELS_TEMPERATURE, temperature, -10, 35))
        hum_pixels = list(set_pixels(PIXELS_HUMIDITY, CONFIG["NUM"], 0, 100))
        temp_pixels = list(set_pixels(PIXELS_TEMPERATURE, CONFIG["NUM"], 0, 100))
        frame = [(0, 0, 0)] * numLEDs

        for p in (hum_pixels + temp_pixels):
            frame[p] = CONFIG["COLOR"]

        DO_MENSA = False
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
#        print(frame)

        client.put_pixels(map_pixels(frame))
        time.sleep(0.05)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Control the weather')
    parser.add_argument('socket', metavar='SOCKET', type=str, help='Bind socket')

    args = parser.parse_args()
    main(args.socket)
