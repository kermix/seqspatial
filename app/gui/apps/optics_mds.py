import re
import pandas as pd

import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from app import app


def get_available_optics_dataset(input_name, procedure, n_components, max_iter, eps):
    options = []

    if procedure == "mds":
        r = re.compile(fr"""
                    ^/optics
                    /c{n_components}
                    /m{max_iter}
                    /e{eps}
                    /s(?P<min_sample>\d+)
                    /x(?P<xi>(\d+(_\d*)?|_\d+)([eE][-+]?\d+)?)
                    /output$""", re.VERBOSE)
    elif procedure == "mds_v2":
        r = re.compile(fr"""
                    ^/clustering_mds
                    /c{n_components}
                    /m{max_iter}
                    /e{eps}
                    /s(?P<min_sample>\d+)
                    /x(?P<xi>(\d+(_\d*)?|_\d+)([eE][-+]?\d+)?)
                    /output$""", re.VERBOSE)

    hdf_filename = f'/output/{input_name}'

    with pd.HDFStore(hdf_filename) as store:
        for key in store.keys():

            m = r.match(key)
            if m:
                result = m.groupdict()

                options.append({
                    "n_components": n_components,
                    "max_iter": max_iter,
                    "eps": eps,
                    "min_sample": result["min_sample"],
                    "xi": result["xi"],
                })
    return sorted(options, key=lambda o: (o["min_sample"], o["xi"]))


def get_layout(input_name, procedure, n_components, max_iter=None, eps=None):
    # TODO: add analysis statictics
    avaliable_options = get_available_optics_dataset(input_name, procedure, n_components, max_iter, eps)

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Choose MDS dataset"),
                html.P("Choose MDS-transformed dataset")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Select(
                    id='optics-dropdown',
                    options=[{
                        'label': f"Components: {o['n_components']}; Max. iterations: {o['max_iter']}; Eps: 1e-{o['eps']}; "
                                 f"Min. sample: {o['min_sample']}; Xi: {o['xi'].replace('_', '.')}",
                        'value': f"f{input_name}_p{procedure}_c{o['n_components']}_"
                                 f"m{o['max_iter']}_e{o['eps']}_s{o['min_sample']}_x{o['xi']}"
                    } for o in avaliable_options
                    ]
                ),
                dbc.Button("Next", id="optics-next-button", size="lg", outline=True, color="primary",
                           className="float-right mt-1",
                           n_clicks=0, disabled=True)
            ]),
        ]),
    ])


@app.callback([Output('optics-next-button', 'href'),
               Output('optics-next-button', 'disabled')],
              [Input('optics-dropdown', 'value')])
def update_next_button_redirect(optics_data):
    if optics_data is None:
        return '', True

    r = re.compile(r"""^f(?P<file>.+.h5)
    _p(?P<procedure>.+)
    _c(?P<n_components>\d+)
    _m(?P<max_iter>\d+)
    _e(?P<eps>\d+)
    _s(?P<min_sample>\d+)
    _x(?P<xi>(\d+(_\d*)?|_\d+)([eE][-+]?\d+)?)$""", re.VERBOSE)
    m = r.match(optics_data)

    if m is None:
        return '', True

    result = m.groupdict()
    file = result['file']
    n_components = result['n_components']
    max_iter = result['max_iter']
    eps = result['eps']
    min_sample = result['min_sample']
    xi = result['xi']
    procedure = result['procedure']

    return f'/plot?dataset={file}&procedure={procedure}&n_components={n_components}' \
           f'&max_iter={max_iter}&eps={eps}&min_sample={min_sample}&xi={xi}', False
