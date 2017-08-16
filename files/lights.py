#!/usr/bin/env python3

import datetime
from datetime import timezone
import functools
import itertools
import json
import math
from collections import namedtuple
import random
import subprocess
import time

import numpy as np
import pysolar

import sqlite3
import zmq

import opc

numLEDs = 512
client = opc.Client('localhost:7890')

PIXELS_TOTAL = set(range(2 * 64))

def BASE_PIXELS_A():
    num = CONFIG["NUM"]
    return range(6, 6 + num)
def BASE_PIXELS_B():
    num = CONFIG["NUM"]
    return range(64 + 10, 64 + 10 + num)

def SUN_PIXELS_A():
    num = CONFIG["NUM"]
    return range(6 + num, 6 + num + 29)
def SUN_PIXELS_B():
    return range(64 + 30, 64 + 59)


PIXELS_BASE = {2, 3, 4, 8}

LIGHTS_CONFIG_FILE = 'lights-config.json'

try:
    with open(LIGHTS_CONFIG_FILE, 'r') as cfg:
        CONFIG = json.load(cfg)
except (FileNotFoundError, ValueError) as e:
    print(e)
    print("Using default values for config")

    CONFIG = {
        "LATITUDE_DEG": 52.0934,
        "LONGITUDE_DEG": 7.2360,

        "ADAPTED_TIME_START": None,
        "ADAPTED_TIME_END": None,

        "COLOR": (255, 47, 1),
        "SUN_COLOR": opc.hex_to_rgb("#ffcc00"),
        "NUM": 4,
        "WIND_FACTOR": 0.1,
        "RAIN_FACTOR": 0.05,
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
    if max(rgb) + delt_v <= 0:
        return rgb
    return hsv_to_rgb((h, s, v + delt_v))

def dim_percentage(rgb, factor):
    h, s, v = rgb_to_hsv(rgb)
    return hsv_to_rgb((h, s, v * factor))

def map_pixels(pixels):
    fill = [(0, 0, 0)] * 64 * 3
    return pixels[0:64].tolist() + fill + pixels[64:64*2].tolist()

DHDataRaw = namedtuple('DHDataRaw', ['date', 'temp', 'hum', 'wind', 'rain'])
DHData = namedtuple('DHData', ['date', 'temp', 'hum', 'wind', 'rain_deriv'])

def weather_row_factory(cursor, row):
    return DHDataRaw(*row)

conn = sqlite3.connect('weather.db', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)

def get_closest_data(conn, date):
    conn.row_factory = weather_row_factory
    for row in conn.execute('''
        SELECT * FROM weather
        ORDER BY ABS( strftime( "%s", date ) - strftime( "%s", ? ) ) ASC
        ''', (date, )):
        return row

def get_closest_data_deriv(conn, date, max_diff):
    conn.row_factory = weather_row_factory
    rows = list(conn.execute('''
        SELECT date as "[timestamp]", temp, hum, wind, rain FROM weather
        ORDER BY ABS( strftime( "%s", date ) - strftime( "%s", ? ) ) ASC
        LIMIT 2
        ''', (date, )))
    dt = (rows[0].date - rows[1].date)
    if dt.seconds > max_diff:
        r = rows[0]
        dhdata = DHData(date=r.date, temp=r.temp, hum=r.hum, wind=r.wind, rain_deriv=None)
    else:
        r = rows[0]
        rain = (rows[0].rain - rows[1].rain) / dt.seconds
        if rain < 0:
            rain = 0
        dhdata = DHData(date=r.date, temp=r.temp, hum=r.hum, wind=r.wind, rain_deriv=rain)
    return dhdata

def get_data_in_timerange(conn, start, end):
    conn.row_factory = weather_row_factory
    return list(conn.execute('''
        SELECT * FROM weather
        WHERE date BETWEEN ? and ?
        ''', (start, end)))

def get_most_recent_date(conn):
    return get_closest_data(conn, datetime.datetime.now(timezone.utc))

def set_density(pixels, value, min_val, max_val):
    l = len(pixels) - 1
    if l < 0:
        l = 0
    val_norm = value / (max_val - min_val)
    if not val_norm > 0:
        val_norm = 0
    if val_norm > 1:
        val_norm = 1
    full_pixels = int(val_norm * l)
    rest = (val_norm * l) - full_pixels
    scaling_frame = np.zeros((numLEDs, 1))
    scaling_frame[pixels[:full_pixels]] = 1
    scaling_frame[pixels[full_pixels]] = rest
    return scaling_frame


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

    if "INFO" in msg:
        return CONFIG

    if "SAVE_CONFIG" in msg:
        with open(LIGHTS_CONFIG_FILE, 'w') as outfile:
            json.dump(jsonData, outfile, sort_keys=True, indent=4, ensure_ascii=False)

    if "START_ADAPT" in msg:
        delta = msg["START_ADAPT"]
        CONFIG["ADAPTED_TIME_START"] = datetime.datetime.now(timezone.utc)
        CONFIG["ADAPTED_TIME_END"] = datetime.datetime.now(timezone.utc) + datetime.timedelta(seconds=delta)

    if "STOP_ADAPT" in msg:
        delta = msg["STOP_ADAPT"]
        CONFIG["ADAPTED_TIME_START"] = None
        CONFIG["ADAPTED_TIME_END"] = None

    if "WIND_FACTOR" in msg:
        wind_delta = msg["WIND_FACTOR"]
        CONFIG["WIND_FACTOR"] = CONFIG["WIND_FACTOR"] + wind_delta
        if CONFIG["WIND_FACTOR"] < 0:
            CONFIG["WIND_FACTOR"] = 0
        return CONFIG["WIND_FACTOR"]

    if "RAIN_FACTOR" in msg:
        rain_delta = msg["RAIN_FACTOR"]
        CONFIG["RAIN_FACTOR"] = CONFIG["RAIN_FACTOR"] + rain_delta
        if CONFIG["RAIN_FACTOR"] < 0:
            CONFIG["RAIN_FACTOR"] = 0
        if CONFIG["RAIN_FACTOR"] > 1:
            CONFIG["RAIN_FACTOR"] = 1
        return CONFIG["RAIN_FACTOR"]


def adapt_date(range_start, range_end):
    #if CONFIG["MODE"] == "TODAY": 
    #if CONFIG["MODE"] == "YESTERDAY":

#    if CONFIG["MODE"] == "LIVE":
#        return datetime.datetime.now(timezone.utc)

    if not CONFIG["ADAPTED_TIME_END"]:
        return datetime.datetime.now(timezone.utc)
    if not CONFIG["ADAPTED_TIME_START"]:
        return datetime.datetime.now(timezone.utc)

    if datetime.datetime.now(timezone.utc) > CONFIG["ADAPTED_TIME_END"]:
        CONFIG["ADAPTED_TIME_END"] = None
        CONFIG["ADAPTED_TIME_START"] = None
        return datetime.datetime.now(timezone.utc)

    delta_orig = range_end - range_start
    delta_adapt = CONFIG["ADAPTED_TIME_END"] - CONFIG["ADAPTED_TIME_START"]
    percent_in = (datetime.datetime.now(timezone.utc) - CONFIG["ADAPTED_TIME_START"]) / delta_adapt
    return range_start + percent_in * delta_orig


def init():
    for i in range(10):
        frame = np.ones((numLEDs, 3)) * (20, 200, 20)
        client.put_pixels(map_pixels(frame))
        time.sleep(0.05)
        frame = np.zeros((numLEDs, 3))
        client.put_pixels(map_pixels(frame))
        time.sleep(0.05)


def find_min_max_range(conn):
    # we search the last 10000 entries for wind and rain and figure out the max values
    # max rain is trivial
    conn.row_factory = None
    query = """
        SELECT MAX(wind) FROM weather
        ORDER BY date DESC
        LIMIT 10000
    """
    try:
        max_wind = list(conn.execute(query))[0][0]
    except IndexError:
        max_wind = 10 # a guess

    query = """
        SELECT date as "[timestamp]", rain FROM weather
        ORDER BY date DESC
        LIMIT 10000
    """
    rain_deriv = 0
    last_date = None
    last_rain = None
    for date, rain in conn.execute(query):
        if last_date is None:
            last_date = date
            last_rain = rain
            continue
        drv = (rain - last_rain) / (date - last_date).seconds
        rain_deriv = max(rain_deriv, abs(drv))

    if rain_deriv == 0:
        rain_deriv = 3 / 60.0 # a guess, 3 per minute

    return max_wind, rain_deriv



def main(socket):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(socket)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    init()

    min_temp = - 10
    min_hum = 20
    min_wind = 0
    min_rain_deriv = 0

    max_temp = 30
    max_hum = 80
    max_wind, max_rain_deriv = find_min_max_range(conn)

    print("Assuming min vals: temp={}, hum={}, wind={}, rain_deriv={}".format(min_temp, min_hum, min_wind, min_rain_deriv))
    print("Assuming max vals: temp={}, hum={}, wind={}, rain_deriv={}".format(max_temp, max_hum, max_wind, max_rain_deriv))

    adapted_date = None
    last_adapted_date = None

    while True:
        socks = dict(poller.poll(timeout=100))
        if socks and sock in socks and socks[sock] == zmq.POLLIN:
            try:
                id_, msg = sock.recv_multipart()
                print("Received msg", msg)
                res = parse_msg(json.loads(msg.decode()))
                sock.send_multipart([id_, json.dumps(res).encode()])
            except json.decoder.JSONDecodeError:
                print("Bad json msg. Ignoring.")

        # yesterday is the day that is more than 28 hours ago …
        yesterday = datetime.datetime.now(timezone.utc) - datetime.timedelta(hours=28)
        yesterday_sunrise, yesterday_sunset = pysolar.util.get_sunrise_sunset(
            latitude_deg=CONFIG["LATITUDE_DEG"],
            longitude_deg=CONFIG["LONGITUDE_DEG"],
            when=yesterday
        )
        # today is the day that was more than 4 hours ago …
        today = datetime.datetime.now(timezone.utc) - datetime.timedelta(hours=4)
        today_sunrise, today_sunset = pysolar.util.get_sunrise_sunset(
            latitude_deg=CONFIG["LATITUDE_DEG"],
            longitude_deg=CONFIG["LONGITUDE_DEG"],
            when=today
        )

        adapted_date = adapt_date(yesterday_sunrise, yesterday_sunset)
        if not last_adapted_date or (adapted_date - last_adapted_date).seconds > 60:
            data = get_closest_data_deriv(conn, adapted_date, 60 * 30)
            print("Fetching new data:", data)

        last_adapted_date = adapted_date


        sun_altitude = pysolar.solar.get_altitude(
            latitude_deg=CONFIG["LATITUDE_DEG"],
            longitude_deg=CONFIG["LONGITUDE_DEG"],
            when=adapted_date
        )
        print(repr(adapted_date), sun_altitude)

        sun_pixels_a_dens = set_density(SUN_PIXELS_A(), sun_altitude, 0, 90)
        sun_pixels_b_dens = set_density(SUN_PIXELS_B(), sun_altitude, 0, 90)

        frame = [(0, 0, 0)] * numLEDs

        frame = np.zeros((numLEDs, 3))

        sun_a_frame = np.ones((numLEDs, 3)) * CONFIG["SUN_COLOR"]# * sun_pixels_a_dens
        sun_b_frame = np.ones((numLEDs, 3)) * CONFIG["SUN_COLOR"]# * sun_pixels_b_dens

        combined_frame = sun_pixels_a_dens + sun_pixels_b_dens

        for p in (list(BASE_PIXELS_A()) + list(BASE_PIXELS_B())):
            col = CONFIG["COLOR"]
            #col = dim_pixel(col, random.randint(-40, 10))
            wind_factor = data.wind / max_wind
            wind_dim = random.uniform(0.9 - CONFIG["WIND_FACTOR"] * wind_factor, 1.1)
            col = dim_percentage(col, wind_dim)
            frame[p] = col

        for p in (list(SUN_PIXELS_A()) + list(SUN_PIXELS_B())):
            col = CONFIG["SUN_COLOR"]
            #col = dim_pixel(col, random.randint(-40, 10))
            wind_factor = data.wind / max_wind
            wind_dim = random.uniform(0.9 - CONFIG["WIND_FACTOR"] * wind_factor, 1.1)
            col = dim_percentage(col, wind_dim * combined_frame[p])
            frame[p] = col

        all_pixels = list(range(3, 60)) + list(range(64 + 3, 64 + 60))
        for p in (list(BASE_PIXELS_A()) + list(BASE_PIXELS_B()) + list(SUN_PIXELS_A()) + list(SUN_PIXELS_B())):
            # iterate and swap with a chance of 0.1 with another random pixel:
            if data.rain_deriv is not None:
                rain_factor = data.rain_deriv / max_rain_deriv * CONFIG["RAIN_FACTOR"] + 0.001
            else:
                rain_factor = 0 / max_rain_deriv * CONFIG["RAIN_FACTOR"] + 0.001
            if random.random() <= rain_factor:
                rand_pix = random.choice(all_pixels)
                c1 = frame[p].copy()
                c2 = frame[rand_pix].copy()
                frame[p] = c2
                frame[rand_pix] = c1

#        print(np.count_nonzero(frame), sum((frame != [0., 0., 0.]).all(1)))
        client.put_pixels(map_pixels(frame))
        time.sleep(0.05)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Control the weather')
    parser.add_argument('socket', metavar='SOCKET', type=str, help='Bind socket')

    args = parser.parse_args()
    main(args.socket)
