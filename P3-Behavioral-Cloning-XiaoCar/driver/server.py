#!/usr/bin/python

import os
import pickle
import threading
import atexit

from flask import Flask, redirect
from flask import render_template
from flask import Response
from apscheduler.schedulers.background import BackgroundScheduler

from drive import Driver
from drive import Camera

app = Flask(__name__)
app.debug = True
setattr(app, 'driver', Driver(app))
setattr(app, 'left', 0)
setattr(app, 'right', 0)
setattr(app, 'data', {})
setattr(app, 'sched', BackgroundScheduler())

@atexit.register
def exit():
    app.driver.drive(0,0)


@app.route("/stoprecord")
def stoprecord():
    if hasattr(app, 'job'):
        getattr(app, 'job').remove()
    return redirect("/")

@app.route("/record")
def record():
    camera = Camera(app)
    sched = getattr(app, 'sched')
    sched.start()
    job = sched.add_job(camera.get_frame, 'interval', seconds=1.5)
    setattr(app, 'job', job)
    return redirect("/")

@app.route("/stop")
def stop():
    app.driver.drive(0, 0)
    return redirect("/")


@app.route("/save")
def save_data():
    DATA = app.data
    if os.path.exists('robot-daylight-S-track.p'):
        with open('robot-daylight-S-track.p', 'rb') as _file:
            data = pickle.load(_file)
    else:
        data = {}
    DATA.update(data)
    with open('robot-daylight-S-track.p', 'wb') as _file:
        pickle.dump(DATA, _file)   
    return ""


@app.route("/drive/<left>/<right>")
def drive(left, right):
    setattr(app, 'left', int(left))
    setattr(app, 'right', int(right))
    app.driver.drive(app.left, app.right)
    return ""


@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host = "0.0.0.0")

    
