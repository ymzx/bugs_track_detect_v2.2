# -*- coding: utf-8 -*-
# @Time    : 2021/2/21
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : bug_http_server.py
# @Software: PyCharm
from flask import Flask
from bug_api import api

app = Flask(__name__) #  Create a Flask WSGI application
api.init_app(app)

app.run(host="0.0.0.0", port=8002, debug=False)