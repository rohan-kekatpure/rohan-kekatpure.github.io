---
layout: post
title: "Hello World Example of Web-Controlled LED"
author: "Rohan"
categories: journal
tags: [documentation,sample]
---

<style>
.boxed { border: 2px solid green;}
</style>

## Introduction

In this post we will demonstrate a minimal example of interfacing ESP8266 with an external webserver. ESP <> Webserver
interfacing will allow us to send commands to the ESP using our computer or cellphone, with the webserver as a
mediator. This in turn allows us to build autonomous systems (using our computer as the main device and the ESP as the
edge device). It also lets us send messages to the ESP from any place with internet access. 

## System structure

Architecture is a perhaps a big word for what we're doing so we will go with "structure". The system diagram for the
web-controlled LED is simple. 

<figure>
    <img src="{{site.url}}/assets/img/webserver_led.png" alt='hierarchy' style='margin: 10px;'>
    <figcaption></figcaption>
</figure>

We host an external webserver which stores the current brightness level of the LED. We
then create a simple responsive UI which can be accessed from desktops as well as mobile devices such as tablets and
cellphones. The UI will have simple a control for LED brightness. The UI will communicate with the webserver and update
the led state. The ESP will be polling the web server every few seconds to inquire the LED brightness. The ESP will
update the LED brightness according to the value it gets from the webserver.  








