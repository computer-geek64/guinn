#!/usr/bin/python3
# api.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import git
import json
import shutil
from uuid import uuid4
from config import FLASK_IP, FLASK_PORT, FLASK_SECRET_KEY, DEBUG_TOKEN
from flask import Flask, jsonify, redirect, session, request, render_template


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = FLASK_SECRET_KEY


# Additional functions
def delete_orphaned_networks():
    networks = next(os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'networks')))[1]
    for network in networks:
        if network not in session:
            shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'networks', network))


# Endpoints
@app.route('/', methods=['GET'])
def get_home():
    return 'Hello world!', 200


@app.route('/git/pull/', methods=['GET'])
def get_git_pull():
    if request.args.get('token') == DEBUG_TOKEN:
        git.cmd.Git(os.path.dirname(os.path.abspath(__file__))).pull()
        return 'Pulling', 200
    return error_403(403)


@app.route('/networks/create/', methods=['POST'])
def get_networks_create():
    delete_orphaned_networks()

    if 'layers' not in request.form:
        return error_400(400)

    if 'nn' not in session:
        uuid = str(uuid4())
        while uuid in session.values():
            uuid = str(uuid4())
        session['nn'] = uuid

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'networks', session['nn']))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'networks', session['nn'], 'network.json'), 'w') as file:
        file.write(request.form.get('layers'))
    return 'Success', 200


@app.route('/networks/delete/', methods=['GET'])
def get_networks_delete():
    if 'nn' in session:
        session.pop('nn')
    delete_orphaned_networks()
    return 'Success', 200


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
