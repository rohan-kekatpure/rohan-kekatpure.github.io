---
layout: post
title: "DRAFT: Getting started with ESP8266"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction
[image]
Microcontrollers sit between the abstract world of bytes and the real world objects you can sense. Programming
microcontrollers is a great compliment to 'regular' programming since it connects these two worlds. The satisfaction of
getting an analog sensor reading to flash on a cheap LCD display is known only to those who have spent countless hours
fighting the electronics. Many low-cost (< $10) microcontrollers and development boards have dramatically reduced the
barrier to experimentation for newcomers like me. One such example is the NodeMCU board for the ESP8266 microcontroller
chip. The present article is a compilation of notes I made while setting up my ESP8266 programming environment.

## Step 1: USB-to-serial converter driver
The first step is to install the USB-to-serial (also known as the USB-to-UART bridge) driver appropriate for the board
we're using. For the computer to recognize the microcontroller when plugged in, the USB-to-serial driver is necessary.
Without it, your computer won't recognize the microcontroller and programming it would be impossible. The name of the
USB-to-serial converter is usually written on the back side of the development board [image]. The board I'm using has
the CP2102 USB-to-serial converter whose driver can be downloaded from the 
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
pip install esptool
```

After installing `esptool` we first need to erase the flash memory and then load the Micropython firmware we downloaded
in the previous step. Flash can be erased using the command:

```shell
esptool.py --port /dev/cu.SLAB_USBtoUART erase_flash
```

The firmware can be loaded using:

```shell
esptool.py --port /dev/tty.SLAB_USBtoUART \
           --baud 115200 write_flash \ 
           --verify \
           --flash_size=detect -fm dio 0 \ 
           esp8266-20210902-v1.17.bin
```

The command for loading the firmware is a bit long. We can copy paste it on the commandline as is, or editing it 
into a single line by removing the back slashes.

At this point, the firmware is loaded and the chip is ready for programming.

## Launching the Python REPL

## Minicom

## Ampy

## Using Minicom to break infinite loop of `main.py`

## Powering the ESP8266 via external power supply


