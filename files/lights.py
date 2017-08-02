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

PIXELS_HUMIDITY = range(29)
PIXELS_TEMPERATURE = range(30, 59)

CONFIG = {

}


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


def init():
    for i in range(10):
        client.put_pixels([(20, 200, 20)] * numLEDs)
        time.sleep(0.05)
        client.put_pixels([(0, 0, 0)] * numLEDs)
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
                parse_msg(msg)
            except json.decoder.JSONDecodeError:
                print("Bad json msg. Ignoring.")

        t += 0.4

        humidity, temperature = 29, 20 # DHT.read_retry(DHT.DHT11, 4)

        hum_pixels = list(set_pixels(PIXELS_HUMIDITY, humidity, 0, 100))
        temp_pixels = list(set_pixels(PIXELS_TEMPERATURE, temperature, -10, 35))
        frame = [(0, 0, 0)] * numLEDs

        for p in (hum_pixels + temp_pixels):
            frame[p] = opc.hex_to_rgb('#ffcc33')

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

        client.put_pixels(frame)
        time.sleep(0.05)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Control the weather')
    parser.add_argument('socket', metavar='SOCKET', type=str, help='Bind socket')

    args = parser.parse_args()
    main(args.socket)
