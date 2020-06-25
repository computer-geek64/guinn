#!/usr/bin/python3
# api.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FLASK_IP, FLASK_PORT
from flask import Flask, jsonify, redirect, request, render_template


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))


pass


if __name__ == "__main__":
    app.run(app.config["IP"], app.config["PORT"])
