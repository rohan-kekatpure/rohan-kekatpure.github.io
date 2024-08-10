---
layout: post
title: "Getting started with ESP8266 using the Arduino IDE"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction

In an earlier [earlier post]({% post_url 2022-01-01-getting-started-with-esp8266 %}) we saw how to get started with
ESP8266 on a NodeMCU board using Micropython. The setup required many steps including the installation of the driver,
the bootloader, and separate applications for serial communications and flashing the code on the ROM. Following these
steps can help us appreciate the various parts of the process. But setting up the process on new computers or for
different variations of the microcontrollers is tedious. Once we understand the setup, the actual steps can distract us
from what is the main attraction for the most of us: exploring different hardware (Infrared, RF, Bluetooth, motor
control) or creating cool projects. It'd be great to have a tool that takes care of drivers, bootloaders, and various
libraries.  









