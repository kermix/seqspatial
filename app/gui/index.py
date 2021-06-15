import re

import logging
from logging import Formatter

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from app import app, get_param_from_url
from apps import choose_dataset, coordinates, optics_pcoa, optics_mds, plot_mds, plot_pcoa, error_404


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')],
              [State('url', 'href')])
def display_page(pathname, href):
    if pathname == "/":
        return choose_dataset.get_layout()

    input_name = get_param_from_url(href, 'dataset')

    if pathname == '/coordinates':
        return coordinates.get_layout(input_name)

    if pathname == '/pcoa':
        return optics_pcoa.get_layout(input_name)

    if pathname in ['/optics', "/plot"]:
        procedure = get_param_from_url(href, 'procedure')
        n_components = get_param_from_url(href, 'n_components')

        if procedure == "mds" or procedure == "mds_v2":
            max_iter = get_param_from_url(href, 'max_iter')
            eps = get_param_from_url(href, 'eps')

            if pathname == "/optics":
                if n_components and max_iter and eps:
                    return optics_mds.get_layout(input_name, procedure, n_components, max_iter, eps)

            if pathname == "/plot":
                min_sample = get_param_from_url(href, 'min_sample')
                xi = get_param_from_url(href, 'xi')

                if n_components and max_iter and eps and min_sample and xi:
                    return plot_mds.get_layout(input_name, procedure, n_components, max_iter, eps, min_sample, xi)

        elif procedure == "pcoa":
            if pathname == "/optics":
                if n_components:
                    return optics_pcoa.get_layout(input_name, n_components)

            if pathname == "/plot":
                min_sample = get_param_from_url(href, 'min_sample')
                xi = get_param_from_url(href, 'xi')

                if n_components and min_sample and xi:
                    return plot_pcoa.get_layout(input_name, n_components, min_sample, xi)

    return error_404.get_layout()


if __name__ == '__main__':
    #remove coruppted default logger
    for handler in app.logger.handlers[:]:
        app.logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    handler.setFormatter(Formatter(
        '%(asctime)s %(levelname)s: %(message)s '
        '[in %(pathname)s:%(lineno)d]', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    app.logger.addHandler(handler)
    app.run_server(host='0.0.0.0', port=8050, debug=True)
