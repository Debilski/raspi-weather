#!/usr/bin/python3 -u

import datetime
from datetime import timezone
import serial
import sys
import os

import sqlite3
import pandas as pd

WDE_DATA_FORMAT = [
    'INV1', 'INV2', 'INV3',
    'TEMP1', 'TEMP2', 'TEMP3', 'TEMP4', 'TEMP5', 'TEMP6', 'TEMP7', 'TEMP8',
    'HUM1', 'HUM2', 'HUM3', 'HUM4', 'HUM5', 'HUM6', 'HUM7', 'HUM8',
    'TEMP', 'HUM', 'WIND', 'RAIN', 'RAIN_SENSOR',
    'END'
]

# serial port of USB-WDE1
port = '/dev/ttyUSB0'

def create_db():
    conn = sqlite3.connect('weather.db')

    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS weather
        (date TEXT, temp REAL, hum INT, wind REAL, rain INT);
        ''')
    c.execute('''
        CREATE INDEX IF NOT EXISTS date_index ON weather(date);
        ''')

    conn.commit()
    conn.close()


def main():
    create_db()

    # open serial line
    ser = serial.Serial(port, 9600)
    if not ser.isOpen():
        print("Unable to open serial port %s" % port)
        sys.exit(1)


    conn = sqlite3.connect('weather.db')

    c = conn.cursor()


    while True:
        print("Trying to read line from {}".format(port))
        # read line from WDE1
        line = ser.readline()
        line = line.strip()
        data = line.split(b';')

        frame = dict(zip(WDE_DATA_FORMAT, data))
        for k, v in frame.items():
            if k.startswith('TEMP') or k.startswith('WIND'):
                try:
                    frame[k] = float(str(v, 'utf-8').replace(',', '.'))
                except (ValueError, UnicodeDecodeError):
                    frame[k] = None
            if k.startswith('HUM') or k == 'RAIN':
                try:
                    frame[k] = int(str(v, 'utf-8').replace(',', '.'))
                except (ValueError, UnicodeDecodeError):
                    frame[k] = None

        frame["TIME"] = pd.to_datetime('now')


        temp = frame['TEMP']
        hum = frame['HUM']
        wind = frame['WIND']
        rain = frame['RAIN']
        c.execute('''
            INSERT INTO weather
            VALUES
            (?, ?, ?, ?, ?)
        ''', (datetime.datetime.now(timezone.utc), temp, hum, wind, rain))
        conn.commit()

        df = pd.DataFrame([frame], columns=["TIME"] + WDE_DATA_FORMAT)
        print(df)

    conn.close()

if __name__ == '__main__':
    main()
