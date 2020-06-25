#!/usr/bin/python3
# api.py

import os
from flask import Flask, jsonify, redirect, request, render_template


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

