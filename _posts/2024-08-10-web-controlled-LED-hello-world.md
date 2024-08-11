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

In this post we will demonstrate a minimal example of interfacing ESP8266 with an external webserver. The
ESP-to-Webserver interfacing will allow us to send commands to the ESP using our computer or cellphone, with the
webserver as a mediator. This in turn allows us to build autonomous systems (using our computer as the main device and
the ESP as the edge device). It also lets us send messages to the ESP from any place with internet access.

## System Structure

The system diagram for the web-controlled LED is simple. 

<figure>
    <img src="{{site.url}}/assets/img/webserver_led.png" alt='hierarchy' style='margin: 10px;'>
    <figcaption></figcaption>
</figure>

We host an external webserver which stores the current brightness level of the LED. A simple responsive UI, accessible
from desktops, tablets and cellphones, will control the LED brightness. The UI will communicate with the webserver and
update the led state. The ESP will poll the web server every few seconds to inquire the LED brightness. The ESP will
then update the LED brightness according to the value it gets from the webserver.

## Webserver Code

First we create a webserver using the [Flask](https://flask.palletsprojects.com/en/3.0.x/) framework. Other frameworks
such as Java/Spring, Django, Rails can be used as well if you're more familiar with them. This part is separate from the
ESP and is just plain vanilla creation of a webserver. If needed, please refer to separate tutorials to refresh your
knowledge of webservers. We reproduce the code for it below. 

```python
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

LED_STATE = 0.5

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/LED', methods=['GET', 'POST'])
def led_state():
    global LED_STATE
    if request.method == 'POST':
        dct = request.json
        val = dct['led_state']
        if 0.0 <= val <= 1.0:
            LED_STATE = val
        return jsonify({'status': 'ok', 'led_state': LED_STATE}), 200
    elif request.method == 'GET':
        return jsonify({'status': 'ok', 'led_state': LED_STATE}), 200
    else:
        return jsonify({'status': 'error', 'msg': 'invalid request'}), 400


if __name__ == '__main__':
    app.run(host='192.168.1.67', port=5000, debug=True)
```










