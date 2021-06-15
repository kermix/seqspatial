import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from sklearn.cluster import cluster_optics_dbscan

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd

from app import app


def get_layout(input_name, n_components, min_sample, xi):
    # TODO: add analysis statictics
    def get_avaliable_principal_components(input_name, n_components):
        pcoa_hdf_key = f"components_pcoa/coordinates"
        hdf_filename = f'/output/{input_name}'

        with pd.HDFStore(hdf_filename, 'r') as store:
            n_components = int(n_components)
            return store[pcoa_hdf_key].columns[:n_components]

    avaliable_components = get_avaliable_principal_components(input_name, n_components)

    return dbc.Container(
        [
            dbc.Row([
                dbc.Col(html.H2("OPTICS Clustering")),
                dbc.Col(html.H2("DBSCAN Clustering")),
            ]),
            dbc.Row(
                dbc.Col(
                    dbc.Form([
                        dbc.FormGroup([
                            dbc.Label("Components", className="mr-2"),
                            dbc.Select(
                                id='first-pc-dropdown',
                                options=[{"label": pc, "value": pc} for pc in avaliable_components]
                            ),
                        ], className="mr-3"),
                        dbc.FormGroup([
                            dbc.Label("vs.", className="mr-2"),
                            dbc.Select(
                                id='second-pc-dropdown',
                                options=[{"label": pc, "value": pc} for pc in avaliable_components]
                            ),
                        ], className="mr-3"),
                        dbc.FormGroup([
                            dbc.Label("DBScan cutoff", className="mr-2"),
                            dbc.Input(
                                type="number",
                                id="pcoa-dbscan-cutoff",
                                min=.001,
                                step=.001,
                            ),
                        ], className="mr-3"),
                        dbc.Button("Plot", id="pcoa-plot-button", color="primary"),
                    ], inline=True)
                )
            ),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='pcoa-optics-graph')
                    , width={"size": 6}),
                dbc.Col(
                    dcc.Graph(id='pcoa-dbscan-graph')
                    , width={"size": 6})
            ]),
            dbc.Row(
                dbc.Col(
                    html.H2("Reachability plot"),
                ),
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(id='pcoa-reachability-graph')
                )
            ),
            dcc.Store(id='pcoa-dataset-params', data={
                "input_name": input_name,
                "n_components": n_components,
                "min_sample": min_sample,
                "xi": xi,
                "first_pc": None,
                "second_pc": None,
                "dbscan_cutoff": None,
            })
        ], fluid=True,
    )


@app.callback(Output("pcoa-dataset-params", "data"),
              Input('pcoa-plot-button', 'n_clicks'),
              [
                  State("first-pc-dropdown", 'value'),
                  State("second-pc-dropdown", 'value'),
                  State("pcoa-dbscan-cutoff", 'value'),
                  State("pcoa-dataset-params", 'data'),
              ])
def set_ploting_params(n_clicks, first_pc, second_pc, cutoff, data):
    if not n_clicks:
        raise PreventUpdate

    return {
        "input_name": data["input_name"],
        "n_components": data["n_components"],
        "min_sample": data["min_sample"],
        "xi": data["xi"],
        "first_pc": first_pc,
        "second_pc": second_pc,
        "dbscan_cutoff": cutoff,
    }


@app.callback([Output("pcoa-optics-graph", "figure"),
               Output('pcoa-dbscan-graph', 'figure'),
               Output('pcoa-reachability-graph', 'figure')],
              [Input("pcoa-dataset-params", "data")])
def update_graphs(data):
    input_name = data["input_name"]
    n_components = data["n_components"]
    min_sample = data["min_sample"]
    xi = data["xi"]

    first_pc = data["first_pc"]
    second_pc = data["second_pc"]
    cutoff = data["dbscan_cutoff"]

    if input_name is None or n_components is None or min_sample is None or xi is None:
        raise PreventUpdate

    if first_pc is None or second_pc is None:
        raise PreventUpdate

    pcoa_hdf_key = f"components_pcoa/coordinates"
    optics_hdf_key = f"clustering_pcoa/c{n_components}/s{min_sample}/x{xi}/output"
    hdf_filename = f'/output/{input_name}'

    optics_fig = get_optics_fig(
        hdf_filename,
        pcoa_hdf_key,
        optics_hdf_key,
        first_pc,
        second_pc
    )

    db_scan_fig, reachability_fig = get_dbscan_and_reachability_figs(
        hdf_filename,
        pcoa_hdf_key,
        optics_hdf_key,
        first_pc,
        second_pc,
        cutoff
    )

    return optics_fig, db_scan_fig, reachability_fig


def get_optics_fig(hdf_filename, pcoa_hdf_key, optics_hdf_key, first_pc, second_pc):
    with pd.HDFStore(hdf_filename, 'r') as store:
        pcoa = store[pcoa_hdf_key][[first_pc, second_pc]]
        optics = store[optics_hdf_key]

        df = pd.merge(pcoa, optics, left_index=True, right_index=True)
        names = df.index

    return px.scatter(df, x=first_pc, y=second_pc, color="labels", hover_name=names)


def get_dbscan_and_reachability_figs(hdf_filename, pcoa_hdf_key, optics_hdf_key, first_pc, second_pc, cutoff):
    with pd.HDFStore(hdf_filename, 'r') as store:
        pcoa = store[pcoa_hdf_key][[first_pc, second_pc]]
        optics = store[optics_hdf_key]

    df = pd.merge(pcoa, optics, left_index=True, right_index=True)
    labels = df.labels[optics.ordering]
    names = df.index[optics.ordering]
    space = np.arange(len(df.index))
    reachability = df.reachability[optics.ordering]

    reach_fig = px.scatter(x=space, y=reachability, color=labels, hover_name=names,
                           range_x=[min(space), max(space) + 1])
    dbscan_fig = go.Figure()

    if cutoff is not None:
        reach_fig.add_annotation(
            x=1,
            y=cutoff + .05,
            text="DBSCAN cutoff",
            xref="paper",
            showarrow=False,
            font_size=12
        )
        reach_fig.add_shape(
            type="line",
            xref='paper',
            x0=0,
            y0=cutoff,
            x1=1,
            y1=cutoff,
            line=dict(color="RoyalBlue", width=3)
        )

        x = df[first_pc]
        y = df[second_pc]
        labels_db = cluster_optics_dbscan(reachability=optics.reachability,
                                          core_distances=optics.core_distances,
                                          ordering=optics.ordering, eps=cutoff)
        labels_db_text = [f"cluster {x}" for x in labels_db]

        dbscan_fig = px.scatter(x=x, y=y, color=labels_db_text, hover_name=df.index)

    return dbscan_fig, reach_fig
