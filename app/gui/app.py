from flask import Flask
import dash

import urllib.parse as urlparse
from urllib.parse import parse_qs


def get_param_from_url(url, parameter):
    try:
        return next((x for x in parse_qs(urlparse.urlparse(url).query)[parameter]), '')
    except KeyError:
        return ''


server = Flask(__name__)

external_stylesheets = [
    {
        'href': 'https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2',
        'crossorigin': 'anonymous'
    }
]

app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)
