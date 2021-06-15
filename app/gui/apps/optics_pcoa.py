import re

import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import pandas as pd

from app import app

def get_layout(input_name):
    def get_proportion_explained_figure(input_name):
        hdf_filename = f'/output/{input_name}'

        with pd.HDFStore(hdf_filename) as store:
            proportion_explained = store['components_pcoa/proportion_explained']

        individual_trace = dict(
            type='scatter',
            x=proportion_explained.index,
            y=proportion_explained.values,
            name='Individual'
        )

        cummulatice_trace = dict(
            type='scatter',
            x=proportion_explained.index,
            y=proportion_explained.cumsum().values,
            name='Cumulative',
            fill='tozeroy'
        )

        return {
            'data': [cummulatice_trace, individual_trace],
            'layout': {
                'yaxis': {
                    'title': 'Explained variance in percent',
                    'tickformat': ',.2%',
                    'range': [0, 1]
                },
                'title': 'Explained variance by different principal components',
            }
        }

    def get_available_optics_dataset(input_name):
        options = []

        r = re.compile(fr"""
                            ^/clustering_pcoa
                            /c(?P<n_components>\d+)
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
                        "n_components": result["n_components"],
                        "min_sample": result["min_sample"],
                        "xi": result["xi"],
                    })
        return sorted(options, key=lambda o: (o["n_components"], o["min_sample"], o["xi"]))


    avaliable_options = get_available_optics_dataset(input_name)

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Choose number of the components/coordinates"),
                html.P("Choose the number of components based on explained proportion")
            ])
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Form(
                    [
                        dbc.FormGroup(
                            [
                                dbc.Select(
                                    id='pcoa-dropdown',
                                    options=[{
                                        'label': f"Components: {o['n_components']}; Min. sample: {o['min_sample']}; "
                                                 f"Xi: {o['xi'].replace('_', '.')}",
                                        'value': f"f{input_name}_c{o['n_components']}_s{o['min_sample']}_"
                                                 f"x{o['xi']}"
                                    } for o in avaliable_options
                                    ]
                                )
                            ],
                            className="mr-3",
                        ),
                        dbc.Button("Next", id="pcoa-next-button", size="lg", outline=True, color="primary",
                                   className="next-btn mt-1",
                                   n_clicks=0, disabled=True),
                    ],
                    inline=True,
                )
            ),
        ]),
        dbc.Row([
            dbc.Col([
                html.P("Elbow graph:"),
                dcc.Graph(id='proportion-graph',
                          figure=get_proportion_explained_figure(input_name)),
            ])
        ]),
    ])


@app.callback([Output('pcoa-next-button', 'href'),
               Output('pcoa-next-button', 'disabled')],
              [Input('pcoa-dropdown', 'value')])
def update_next_button_redirect(optics_data):
    if optics_data is None:
        return "", True

    r = re.compile(r"""^f(?P<file>.+.h5)
    _c(?P<n_components>\d+)
    _s(?P<min_sample>\d+)
    _x(?P<xi>(\d+(_\d*)?|_\d+)([eE][-+]?\d+)?)$""", re.VERBOSE)
    m = r.match(optics_data)

    if m is None:
        return '', True

    result = m.groupdict()
    file = result['file']
    n_components = result['n_components']
    min_sample = result['min_sample']
    xi = result['xi']

    return f'/plot?dataset={file}&' \
           f'procedure=pcoa&' \
           f'n_components={n_components}&' \
           f'min_sample={min_sample}&' \
           f'xi={xi}', False


