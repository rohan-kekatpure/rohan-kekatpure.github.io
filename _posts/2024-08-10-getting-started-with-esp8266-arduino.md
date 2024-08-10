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
the bootloader. Following that, the setup required installation of separate applications for serial communications and
code upload. Going through these steps can help us appreciate the various parts of the process. But setting up the
process on new computers or for different variations of the microcontrollers is tedious. Once we understand the setup,
the actual steps can distract us from what is the main attraction for the most of us: exploring different hardware (
Infrared, RF, Bluetooth, motor control) or creating cool projects. It'd be great to have a tool that takes care of the
whole set up process and lets us focus on making things. This tool is
the [Arduino IDE](https://www.arduino.cc/en/software). 

## The Arduino IDE

IDE stands for [Integrated Development Environment](https://en.wikipedia.org/wiki/Integrated_development_environment).
An IDE is a software for enabling developments of a certain kinds of applications. For example Jetbrains IntelliJ is an
IDE for Java development. Android Studio is an IDE for developing Android application. There are IDEs for nearly all major
application types including Apple products (Xcode), C/C++ (CLion, Qt), and Python (Pycharm). Each IDE makes it easier to
perform environment setup, library search and installation, auto-complete, code browsing and other repetitive tasks.

The Arduino IDE makes it easy to develop robotics and embedded applications on the Arduino microcontroller board. Though
primarily developed for Arduino, the IDE nevertheless has been extended for use with other microcontrollers, including
the ones from the ESP family (ESP8266 and ESP32).












