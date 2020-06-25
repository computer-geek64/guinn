#!/usr/bin/python3
# api.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FLASK_IP, FLASK_PORT
from flask import Flask, jsonify, redirect, request, render_template


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))


@app.route('/', methods=['GET'])
def get_home():
    return 'Hello world!'


# Error handlers
@app.errorhandler(404)
def error_404(e):
    return render_template('errors/404.html'), 404


@app.errorhandler(400)
def error_400(e):
    return 'HTTP 400 - Bad Request', 400


@app.errorhandler(500)
def error_500(e):
    return 'HTTP 500 - Internal Server Error', 500


@app.errorhandler(403)
def error_403(e):
    return render_template('errors/403.html'), 403


if __name__ == '__main__':
    app.run(FLASK_IP, FLASK_PORT)

