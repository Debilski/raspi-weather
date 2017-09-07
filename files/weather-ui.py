#!/usr/bin/env python2

import functools
import os
import random
import time

import zmq

import cherrypy
from piui import PiUi

current_dir = os.path.dirname(os.path.abspath(__file__))


class DemoPiUi(object):

    def __init__(self, socket):
        self.title = None
        self.txt = None
        self.img = None
        self.ui = PiUi(img_dir=os.path.join(current_dir, 'imgs'))
        self.src = "sunset.png"

        self._socket = socket

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.connect(self._socket)

        self.pollin = zmq.Poller()
        self.pollin.register(self.sock, zmq.POLLIN)
        self.pollout = zmq.Poller()
        self.pollout.register(self.sock, zmq.POLLOUT)

    def page_static(self):
        self.page = self.ui.new_ui_page(title="Static Content", prev_text="Back", onprevclick=self.main_menu)
        self.page.add_textbox("Add a mobile UI to your Raspberry Pi project", "h1")
        self.page.add_element("hr")
        self.page.add_textbox("You can use any static HTML element " +
                              "in your UI and <b>regular</b> <i>HTML</i> <u>formatting</u>.", "p")
        self.page.add_element("hr")
        self.page.add_textbox("Your python code can update page contents at any time.", "p")
        update = self.page.add_textbox("Like this...", "h2")
        time.sleep(2)
        for a in range(1, 5):
            update.set_text(str(a))
            time.sleep(1)

    def page_buttons(self):
        self.page = self.ui.new_ui_page(title="Buttons", prev_text="Back", onprevclick=self.main_menu)
        self.title = self.page.add_textbox("Buttons!", "h1")
        plus = self.page.add_button("Up Button &uarr;", self.onupclick)
        minus = self.page.add_button("Down Button &darr;", self.ondownclick)

    def page_input(self):
        self.page = self.ui.new_ui_page(title="Input", prev_text="Back", onprevclick=self.main_menu)
        self.title = self.page.add_textbox("Input", "h1")
        self.txt = self.page.add_input("text", "Name")
        button = self.page.add_button("Say Hello", self.onhelloclick)

    def page_images(self):
        self.page = self.ui.new_ui_page(title="Images", prev_text="Back", onprevclick=self.main_menu)
        self.img = self.page.add_image("sunset.png")
        self.page.add_element('br')
        button = self.page.add_button("Change The Picture", self.onpicclick)

    def page_toggles(self):
        self.page = self.ui.new_ui_page(title="Toggles", prev_text="Back", onprevclick=self.main_menu)
        self.list = self.page.add_list()
        self.list.add_item("Lights", chevron=False, toggle=True,
                           ontoggle=functools.partial(self.ontoggle, "lights"))
        self.list.add_item("TV", chevron=False, toggle=True,
                           ontoggle=functools.partial(self.ontoggle, "tv"))
        self.list.add_item("Microwave", chevron=False, toggle=True,
                           ontoggle=functools.partial(self.ontoggle, "microwave"))
        self.page.add_element("hr")
        self.title = self.page.add_textbox("Home Appliance Control", "h1")

    def page_console(self):
        con = self.ui.console(title="Console", prev_text="Back", onprevclick=self.main_menu)
        con.print_line("Hello Console!")

    def main_menu(self):
        self.page = self.ui.new_ui_page(title="PiUi")
        self.list = self.page.add_list()
        self.list.add_item("Reload UI", onclick=lambda: os.system("sudo systemctl reload piui.service"))
        
        self.list.add_item("Info", onclick=self.show_info)

        self.list.add_item("Change Color", onclick=self.page_change_color)

        self.list.add_item("Reboot", chevron=True, onclick=self.page_reboot)
        self.list.add_item("Poweroff", chevron=True, onclick=self.page_poweroff)

#        self.list.add_item("Static Content", chevron=True, onclick=self.page_static)
#        self.list.add_item("Buttons", chevron=True, onclick=self.page_buttons)
#        self.list.add_item("Input", chevron=True, onclick=self.page_input)
#        self.list.add_item("Images", chevron=True, onclick=self.page_images)
#        self.list.add_item("Toggles", chevron=True, onclick=self.page_toggles)
        self.list.add_item("Console!", chevron=True, onclick=self.page_console)
        self.ui.done()

    def page_change_color(self):
        self.page = self.ui.new_ui_page(title="Change Color", prev_text="Back", onprevclick=self.main_menu)
        self.title = self.page.add_textbox("Buttons!", "h1")
        self.page.add_element('br')
        plus = self.page.add_button("R &uarr;", lambda: self.do_change_color(+1, 0, 0))
        minus = self.page.add_button("R &darr;", lambda: self.do_change_color(-1, 0, 0))
        self.page.add_element('br')
        plus = self.page.add_button("G &uarr;", lambda: self.do_change_color(0, +1, 0))
        minus = self.page.add_button("G &darr;", lambda: self.do_change_color(0, -1, 0))
        self.page.add_element('br')
        plus = self.page.add_button("B &uarr;", lambda: self.do_change_color(0, 0, +1))
        minus = self.page.add_button("B &darr;", lambda: self.do_change_color(0, 0, -1))

        self.page.add_element('br')
        self.page.add_element('br')
        self.page.add_element('br')

        plus = self.page.add_button("Num &uarr;", lambda: self.do_change_num_leds(+1))
        minus = self.page.add_button("Num &darr;", lambda: self.do_change_num_leds(-1))

        self.page.add_element('br')

        plus = self.page.add_button("Dim red &uarr;", lambda: self.do_change_dim_leds(8))
        minus = self.page.add_button("Dim red &darr;", lambda: self.do_change_dim_leds(-8))

        self.page.add_element('br')

        plus = self.page.add_button("Dim yellow &uarr;", lambda: self.do_change_dim_sun(8))
        minus = self.page.add_button("Dim yellow &darr;", lambda: self.do_change_dim_sun(-8))

        self.page.add_element('br')

        plus = self.page.add_button("Wind &uarr;", lambda: self.do_change_wind(+0.1))
        minus = self.page.add_button("Wind &darr;", lambda: self.do_change_wind(-0.1))

        self.page.add_element('br')

        plus = self.page.add_button("Rain &uarr;", lambda: self.do_change_rain(+0.05))
        minus = self.page.add_button("Rain &darr;", lambda: self.do_change_rain(-0.05))

        self.page.add_element('br')
        fast = self.page.add_button("Fast", lambda: self.do_show_yesterday(30))
        slow = self.page.add_button("Slow", lambda: self.do_show_yesterday(60 * 30))
        slow = self.page.add_button("Stop", self.do_stop_adapt)

    def show_info(self):
        self.page = self.ui.new_ui_page(title="Change Color", prev_text="Back", onprevclick=self.main_menu)
        self.title = self.page.add_textbox("Info", "h1")
        self.page.add_element('br')

        self.list = self.page.add_list()
        info = self.query_info()
        if info:
            for k, v in info.items():
                self.list.add_item("{}: {}".format(k, v))

    def query(self, msg, args):
        msg = {msg: args}
        evts = self.pollout.poll(1000)
        if not evts:
            return
        self.sock.send_json(msg)
        evts = self.pollin.poll(1000)
        if not evts:
            return
        new_col = self.sock.recv_json()
        self.title.set_text(str(new_col))

    def query_info(self):
        msg = {"INFO": []}
        evts = self.pollout.poll(1000)
        if not evts:
            return
        self.sock.send_json(msg)
        evts = self.pollin.poll(3000)
        if not evts:
            return
        info = self.sock.recv_json()
        return info

    def do_change_wind(self, delt):
        return self.query("WIND_FACTOR", delt)

    def do_change_rain(self, delt):
        return self.query("RAIN_FACTOR", delt)

    def do_show_yesterday(self, delt):
        return self.query("START_ADAPT", delt)

    def do_stop_adapt(self):
        return self.query("STOP_ADAPT", [])

    def do_change_color(self, r, g, b):
        return self.query("CHANGE_COLOR", [r, g, b])

    def do_change_num_leds(self, diff):
        return self.query("CHANGE_NUM_LEDS", diff)

    def do_change_dim_leds(self, mult):
        return self.query("DIM_LEDS", mult)

    def do_change_dim_sun(self, mult):
        return self.query("DIM_SUN", mult)

    def page_reboot(self):
        self.page = self.ui.new_ui_page(title="Reboot", prev_text="Back", onprevclick=self.main_menu)
        self.page.add_element('br')
        button = self.page.add_button("Reboot", self.do_reboot)

    def page_poweroff(self):
        self.page = self.ui.new_ui_page(title="Poweroff", prev_text="Back", onprevclick=self.main_menu)
        self.page.add_element('br')
        button = self.page.add_button("Poweroff", self.do_poweroff)

    def do_reboot(self):
        os.system("sudo systemctl reboot")

    def do_poweroff(self):
        os.system("sudo systemctl poweroff")

    def main(self):
        self.main_menu()

        self.ui.done()

    def onupclick(self):
        self.title.set_text("Up ")

    def ondownclick(self):
        self.title.set_text("Down")

    def onhelloclick(self):
        self.title.set_text("Hello " + self.txt.get_text())

    def onpicclick(self):
        if self.src == "sunset.png":
          self.img.set_src("sunset2.png")
          self.src = "sunset2.png"
        else:
          self.img.set_src("sunset.png")
          self.src = "sunset.png"

    def ontoggle(self, what, value):
        self.title.set_text("Toggled " + what + " " + str(value))


def main(socket):
    piui = DemoPiUi(socket)
    piui.main()


if __name__ == '__main__':

    cherrypy.config.update({'engine.autoreload.on': False})

    import argparse

    parser = argparse.ArgumentParser(description='Control the weather')
    parser.add_argument('socket', metavar='SOCKET', type=str, help='Connection socket')

    args = parser.parse_args()
    main(args.socket)
