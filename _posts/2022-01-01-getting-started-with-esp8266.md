---
layout: post
title: "Getting started with ESP8266"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction
Programming microcontrollers is a great compliment to 'regular' programming. Many low-cost (< $10) microcontrollers 
and development boards have dramatically reduced the barrier to experimentation for newcomers like me. One such 
example is the NodeMCU board for the ESP8266 microcontroller chip. The present article is a set of notes I compiled 
while setting up my ESP8266 programming environment.

## USB-to-serial converter driver
The first step is to install the USB-to-serial (also known as the USB-to-UART bridge) driver appropriate for the board
we're using. For the computer to recognize the microcontroller when plugged in, the USB-to_Serial driver is necessary.
Without it, your computer won't recognize the microcontroller and programming it would be impossible. The name of the
USB-to-serial converter is usually written on the back side of the development board. The board I'm using has the CP2102
USB-to-serial converter whose driver can be 
[downloaded from the vendor website](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers).

## Firmware

## ESPTool

## Launching the Python REPL

## Minicom

## Ampy

## Using Minicom to break infinite loop of `main.py`

## Powering the ESP8266 via external power supply


