---
layout: post
title: "Getting started with ESP8266 using MicroPython"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction
[image]
Microcontrollers sit between the abstract world of bytes and the real world objects you can see and touch. Programming
microcontrollers is a way to connect these two worlds. The satisfaction of getting a sensor reading flashing on a cheap
LCD is known only to those who have spent the hours fighting the electronics. Many low-cost (< $10)
microcontrollers and development boards have significantly reduced the barrier to get started. One such example is
the ESP8266 microcontroller chip and the NodeMCU development board. I recently got started with this system and the 
present article is a compilation of step-by-step notes I made while setting up my environment. 

## 1: USB-to-serial converter driver
The first step is to install the USB-to-serial driver appropriate for the board we're using. For the computer to
recognize the microcontroller when plugged in, the USB-to-serial driver is necessary. Without it, your computer won't
recognize the microcontroller and programming it would be impossible. The name of the USB-to-serial converter is usually
written on the back side of the development board [image]. The board I'm using has the CP2102 USB-to-serial converter
whose driver can be downloaded from the
[vendor website](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers).

Once you have downloaded and installed the USB-to-serial driver, you can check if the ESP is recognized by your PC 
by plugging it to the USB port. On a MAC, the ESP, when recognized properly, shows up as a file item in the `/dev`.
[image]. At this point, we're ready to take the next step.

## Firmware
The firmware can be informally thought of as the operating system (OS) of a microcontroller. The firmware provides 
APIs for accessing the resources of the chip. For the ESP8266, there are multiple choices for the firmware depending 
on the language. 
[Micropython](https://docs.micropython.org/en/latest/esp8266/tutorial/intro.html) 
(or its close cousin CircuitPython) is the Python-based firmware and the one we will be using in this article. 
Micropython is a compiled binary with a `.bin` extension. It can be downloaded from the Micropython 
[downloads section](https://micropython.org/download/#esp8266). 

So this small 600 kB `.bin` file is the operating system for the ESP8266. Next we have to figure out a way to load 
this OS into the the ESP8266.  

## Step 2: Bootloader
When a chip is freshly powered on, it does not yet have the OS in its memory; some process must load software 
into memory before it can be executed. This initial process is called the bootloader. For ESP8266 the most common 
bootloader is the `esptool` which is fortunately `pip` installable:

```shell
$ pip install esptool
```

After installing `esptool` we first need to erase the flash memory and then load the Micropython firmware we downloaded
in the previous step. Flash can be erased using the command:

```shell
$ esptool.py --port /dev/cu.SLAB_USBtoUART erase_flash
```

The firmware can be loaded using:

```shell
$ esptool.py --port /dev/tty.SLAB_USBtoUART \
           --baud 115200 write_flash \ 
           --verify \
           --flash_size=detect -fm dio 0 \ 
           esp8266-20210902-v1.17.bin
```

The command for loading the firmware is a bit long. We can copy-paste it on the commandline as is, or editing it 
into a single line by removing the backslashes. The crucial input here is the baud rate. The recommended baud rate is 
115200. The Higher the baud rate, the faster the upload/download of the data from the chip. Note that the recommended 
baud rate differs from 9600 written on the bottom of the chip. The `--verify` and `-fm dio` options can be skipped, 
but I found that skipping them leads to corrupted loads. 

If everything went correctly, then the firmware should be loaded and the chip is ready for programming. The next step 
shows how to verify the correct loading of the firmware. 

## Launching the Python REPL
This part describes how to obtain a Python shell running on the ESP8266. This is exciting since different chip level 
operations (making a pin 'hi' or 'low', sensor voltage, list of WiFi access points etc) can be tested in an 
interactive manner without having to recompile and reload the code. 

Once the firmware has been loaded, multiple application can be used to connect to the ESP8266. On a Mac, the native 
terminal multiplexer program [`screen`](https://ss64.com/osx/screen.html) can be used to open a console terminal on 
the ESP8266. While many people have had success using `screen` to communicate with the chip, I could not get it to 
work. This is when I switched to Minicom.

## Minicom

[Minicom](https://macappstore.org/minicom/) is a serial port communication application. That is, it enables 
communication with devices connected to the computer's serial port. Minicom can be installed via Homebrew:

```shell
$ brew install minicom
```

Minicom is a menu-driven application. One-time initial setup is required, which can be performed by bringing up the 
setup menu:

```shell
$ minicom -s
```

The main settings are the baud rate and the default serial port location under "Serial port settings". To avoid having
to repeatedly type the name of the serial port, the default value can be provided under "Serial Device". Replace the
empty or existing value by the full path of the serial port. Also replace the empty or the default baud rate with 
the recommended value 115200. At this point you can press `ESC` to exit out of minicom. 

Once the setup is complete you can simply type `minicom` in the shell; it should open up the Python terminal on 
ESP8266 ! Remember this is a greatly stripped down version of the regular Python. Poke around to 
learn more about the available modules, system information, files etc. If you have some LEDs and resistors, you can 
do some cool things right in the Python shell at this point including:

1. learn more about the chip by querying hardware specs
2. pulsing LEDs ON/OFF by setting the output pins high/low
3. reading a list of WiFi access points near you
4. use pulse width modulation (PWM) to dim the LEDs (an analog effect using digital operations)

All of these can be done by following the brief 
[tutorial](https://docs.micropython.org/en/latest/esp8266/quickref.html) 
at Micropython.org.

To exit out of Minicom, bring up the menu by pressing the exit key combination (`ESC-x` on my Mac) and selecting 
`'yes' on the confirmation menu. 

## Ampy

It is cool to interactively execute code and watch it make changes in the real world like turning on lights. But 
your code will vanish from the chip memory the moment you exit out of minicom. The cool thing about microcontrollers 
like the ESP8266 is that they can be powered externally (i.e. without the USB connection) and allowed to run freely. 
To allow that, the code has to be burned into its non-volatile memory. Once the code is burned in the non-volatile 
memory, it says there even after the power is turned off. Also, if the code is named `main.py` it will be run 
automatically after the chip is powered up again. To know how this happens read more about the 
[ESP8266 boot procedure](https://docs.micropython.org/en/latest/esp8266/general.html).

To burn our locally developed code, we use the Adafruit Micropython Tool, aka 
[`ampy`](https://learn.adafruit.com/micropython-basics-load-files-and-run-code?view=all). 

You can install `ampy` using `pip`:

```shell
$ pip install adafruit-ampy
```
`ampy` has simple get and put commands for two-way communication between the computer and the ESP. To transfer a 
Python file on your local machine to the ESP's non-volatile memory, you can use:

```shell
$ ampy --port /dev/cu.SLAB_USBtoUART put path/to/localfile.py path/to/remotefile.py
```

Similarly to transfer a file on the ESP to current directory on your local machine you can use:

```shell
$ ampy --port /dev/cu.SLAB_USBtoUART get path/to/remotefile.py
```

`ampy` supports a whole range of operations in addition to the `get` and `put`, which can be read from the 
[documentation](https://learn.adafruit.com/micropython-basics-load-files-and-run-code?view=all).

## Using Minicom to break infinite loop of `main.py`

## Powering the ESP8266 via external power supply

Once you've flashed the code onto ESP's ROM, it is ready to be executed in a standalone mode. We can detach the ESP from
the computer and power it with an external supply. For that, you can apply a voltage of between 2.5V to 3.3V to pin(s)
labeled `3v3` or a voltage of 7-10V to the pin labeled `Vin`. The negative terminal of the battery should go to `Gnd`.


