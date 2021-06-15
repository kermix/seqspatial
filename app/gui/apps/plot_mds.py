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


def get_layout(input_name, procedure, n_components, max_iter, eps, min_sample, xi):
    # TODO: add analysis statictics
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col([
                        html.H2("OPTICS Clustering"),
                        dcc.Graph(id='optics-graph',
                                  figure=get_optics_fig(input_name, procedure, n_components, max_iter, eps, min_sample, xi)),
                    ], width={"size": 6}),
                    dbc.Col([
                        html.H2("DBSCAN Clustering"),
                        dcc.Graph(id='dbscan-graph')
                    ], width={"size": 6}),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([
                        html.H2("Reachability plot"),
                    ], width={"size": 6}),
                    dbc.Col([
                        dbc.FormGroup(
                            [
                                dbc.Input(
                                    type="number",
                                    id="dbscan-cutoff",
                                    placeholder="Set DBSCAN cutoff and press ENTER or change focus",
                                    min=.001,
                                    step=.001,
                                    debounce=True,
                                ),
                            ])
                    ], width={"size": 3, "offset": 2}),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id='reachability-graph')
                    )
                ]
            ),
            dcc.Store(id='dataset-params', data={
                "input_name": input_name,
                "n_components": n_components,
                "max_iter": max_iter,
                "eps": eps,
                "min_sample": min_sample,
                "xi": xi,
                "procedure": procedure
            })
        ], fluid=True,
    )


def get_layout(input_name, procedure, n_components, max_iter, eps, min_sample, xi):
    # TODO: add analysis statictics
    def get_avaliable_dimmensions(input_name, procedure, n_components, max_iter, eps):
        if procedure == "mds":
            mds_hdf_key = f"mds/c{n_components}/m{max_iter}/e{eps}"
        elif procedure == "mds_v2":
            mds_hdf_key = f"components_mds/c{n_components}/m{max_iter}/e{eps}/coordinates"

        hdf_filename = f'/output/{input_name}'

        with pd.HDFStore(hdf_filename, 'r') as store:
            return store[mds_hdf_key].columns

    avaliable_dimmensions = get_avaliable_dimmensions(input_name, procedure, n_components, max_iter, eps)

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
                                id='first-dim-dropdown',
                                options=[{"label": d, "value": d} for d in avaliable_dimmensions]
                            ),
                        ], className="mr-3"),
                        dbc.FormGroup([
                            dbc.Label("vs.", className="mr-2"),
                            dbc.Select(
                                id='second-dim-dropdown',
                                options=[{"label": d, "value": d} for d in avaliable_dimmensions]
                            ),
                        ], className="mr-3"),
                        dbc.FormGroup([
                            dbc.Label("DBScan cutoff", className="mr-2"),
                            dbc.Input(
                                type="number",
                                id="mds-dbscan-cutoff",
                                min=.001,
                                step=.001,
                            ),
                        ], className="mr-3"),
                        dbc.Button("Plot", id="mds-plot-button", color="primary"),
                    ], inline=True)
                )
            ),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='mds-optics-graph')
                    , width={"size": 6}),
                dbc.Col(
                    dcc.Graph(id='mds-dbscan-graph')
                    , width={"size": 6})
            ]),
            dbc.Row(
                dbc.Col(
                    html.H2("Reachability plot"),
                ),
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(id='mds-reachability-graph')
                )
            ),
            dcc.Store(id='mds-dataset-params', data={
                "input_name": input_name,
                "n_components": n_components,
                "max_iter": max_iter,
                "eps": eps,
                "min_sample": min_sample,
                "xi": xi,
                "procedure": procedure,
                "first_dim": None,
                "second_dim": None,
                "dbscan_cutoff": None,
            })
        ], fluid=True,
    )


@app.callback(Output("mds-dataset-params", "data"),
              Input('mds-plot-button', 'n_clicks'),
              [
                  State("first-dim-dropdown", 'value'),
                  State("second-dim-dropdown", 'value'),
                  State("mds-dbscan-cutoff", 'value'),
                  State("mds-dataset-params", 'data'),
              ])
def set_ploting_params(n_clicks, first_dim, second_dim, cutoff, data):
    if not n_clicks:
        raise PreventUpdate

    return {
                "input_name": data["input_name"],
                "n_components": data["n_components"],
                "max_iter": data["max_iter"],
                "eps": data["eps"],
                "min_sample": data["min_sample"],
                "xi": data["xi"],
                "procedure": data["procedure"],
                "first_dim": first_dim,
                "second_dim": second_dim,
                "dbscan_cutoff": cutoff,
            }

@app.callback([Output("mds-optics-graph", "figure"),
               Output('mds-dbscan-graph', 'figure'),
               Output('mds-reachability-graph', 'figure')],
              [Input("mds-dataset-params", "data")])
def update_graphs(data):
    input_name = data["input_name"]
    n_components = data["n_components"]
    min_sample = data["min_sample"]
    max_iter = data["max_iter"]
    eps = data["eps"]
    procedure = data["procedure"]
    xi = data["xi"]

    first_dim = data["first_dim"]
    second_dim = data["second_dim"]
    cutoff = data["dbscan_cutoff"]

    if input_name is None \
            or procedure is None \
            or n_components is None \
            or max_iter is None \
            or eps is None \
            or min_sample is None \
            or xi is None:
        raise PreventUpdate

    if first_dim is None or second_dim is None:
        raise PreventUpdate

    if procedure == "mds":
        mds_hdf_key = f"mds/c{n_components}/m{max_iter}/e{eps}"
        optics_hdf_key = f"optics/c{n_components}/m{max_iter}/e{eps}/s{min_sample}/x{xi}/output"
    elif procedure == "mds_v2":
        mds_hdf_key = f"components_mds/c{n_components}/m{max_iter}/e{eps}/coordinates"
        optics_hdf_key = f"clustering_mds/c{n_components}/m{max_iter}/e{eps}/s{min_sample}/x{xi}/output"

    hdf_filename = f'/output/{input_name}'

    optics_fig = get_optics_fig(
        hdf_filename,
        mds_hdf_key,
        optics_hdf_key,
        first_dim,
        second_dim
    )

    db_scan_fig, reachability_fig = get_dbscan_and_reachability_figs(
        hdf_filename,
        mds_hdf_key,
        optics_hdf_key,
        first_dim,
        second_dim,
        cutoff
    )

    return optics_fig, db_scan_fig, reachability_fig


def get_optics_fig(hdf_filename, mds_hdf_key, optics_hdf_key, first_dim, second_dim):
    with pd.HDFStore(hdf_filename, 'r') as store:
        mds = store[mds_hdf_key]
        try:
            mds = mds[[first_dim, second_dim]]
        except KeyError:
            mds = mds.iloc[:, [int(first_dim), int(second_dim)]]
        optics = store[optics_hdf_key]

        df = pd.merge(mds, optics, left_index=True, right_index=True)
        names = df.index

    return px.scatter(df, x=df.columns[0], y=df.columns[1], color="labels", hover_name=names,
                      labels={
                          f"{df.columns[0]}": f"Component {int(first_dim)}",
                          f"{df.columns[1]}": f"Component {int(second_dim)}",
                          "labels": "Clusters"
                      },
                      title="Automatic Optics Clustering"
                      )


def get_dbscan_and_reachability_figs(hdf_filename, mds_hdf_key, optics_hdf_key, first_dim, second_dim, cutoff):
    with pd.HDFStore(hdf_filename, 'r') as store:
        mds = store[mds_hdf_key]
        try:
            mds = mds[[first_dim, second_dim]]
        except KeyError:
            mds = mds.iloc[:, [int(first_dim), int(second_dim)]]
        optics = store[optics_hdf_key]

    # df = pd.concat([mds, optics], axis=1, sort=False)
    df = pd.merge(mds, optics, left_index=True, right_index=True)
    labels = df.labels[optics.ordering]
    names = df.index[optics.ordering]
    space = np.arange(len(df.index))
    reachability = df.reachability[optics.ordering]

    reach_fig = px.scatter(x=space, y=reachability, color=labels, hover_name=names, range_x=[min(space), max(space)+1])
    dbscan_fig = go.Figure()

    if cutoff is not None:
        reach_fig.add_annotation(
            x=1,
            y=cutoff+.05,
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

        x = df.iloc[:, int(first_dim)]
        y = df.iloc[:, int(second_dim)]
        labels_db = cluster_optics_dbscan(reachability=optics.reachability,
                                           core_distances=optics.core_distances,
                                           ordering=optics.ordering, eps=cutoff)
        labels_db_text = [f"cluster {x}" for x in labels_db]

        dbscan_fig = px.scatter(
            x=x,
            y=y,
            color=labels_db_text,
            hover_name=df.index,
            labels={
                "x": f"Component {int(first_dim)}",
                "y": f"Component {int(second_dim)}",
                "color": "Clusters"
            },
            title=f"DBSCAN clustering for epsilon {cutoff}"
        )

    return dbscan_fig, reach_fig
