import dash_table
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import re
import pandas as pd

from app import app


def get_available_mds_dataset(input_name):
    options = []

    hdf_filename = f'/output/{input_name}'

    with pd.HDFStore(hdf_filename) as store:
        for key in store.keys():
            m = re.match(r'^/(?:mds|components_mds)/'
                         r'c(?P<n_components>\d+)/'
                         r'm(?P<max_iter>\d+)/'
                         r'e(?P<eps>\d+)'
                         r'(?P<new_version>/coordinates)?$', key)
            if m:
                result = m.groupdict()

                options.append({
                    "procedure": "MDS",
                    "n_components": result["n_components"],
                    "max_iter": result["max_iter"],
                    "eps": result["eps"],
                    "new_version": result["new_version"] == "/coordinates"
                })

                continue

            n = re.match(r'^/components_pcoa/coordinates$', key)
            if n:
                options.append({
                    "procedure": "PCoA",
                })

    return sorted(options, key=lambda o: (o["procedure"],
                                          o["n_components"] if "n_components" in o.keys() else None,
                                          o["max_iter"] if "max_iter" in o.keys() else None,
                                          o["eps"] if "eps" in o.keys() else None))


def get_layout(input_name):
    def gen_options_list(available_options):
        options = []
        for o in available_options:
            if o["procedure"] == "MDS":
                options.append({
                    'label': f"Procedure: MDS{'_v2' if o['new_version'] else ''}; "
                             f"Components: {o['n_components']}; "
                             f"Max. iterations: {o['max_iter']}; "
                             f"Eps: 1e-{o['eps']}",
                    'value': f"f{input_name};"
                             f"mds{'_v2' if o['new_version'] else ''};"
                             f"c{o['n_components']};"
                             f"m{o['max_iter']};"
                             f"e{o['eps']}"
                })
            elif o["procedure"] == "PCoA":
                options.append({
                    'label': f"Procedure: PCoA",
                    'value': f"f{input_name};pcoa"
                })
        return options

    available_options = get_available_mds_dataset(input_name)

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Choose coordinates dataset"),
                html.P("Choose dataset transformed by MDS or PCoA")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Select(
                    id='coordinates-dropdown',
                    options=gen_options_list(available_options)
                ),
                dbc.Button("Next", id="coordinates-next-button", size="lg", outline=True, color="primary",
                           className="float-right mt-1",
                           n_clicks=0, disabled=True)
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(id="mds-statistics")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.P("Data preview:"),
                dash_table.DataTable(
                    id='coordinates-datatable',
                    # page_current=0,
                    # page_size=20,
                    # page_action='custom',
                    # changed to preview
                    style_table={"overflowX": "auto"}

                )
            ],
            )
        ]),
    ])


@app.callback([Output('coordinates-next-button', 'href'),
               Output('coordinates-next-button', 'disabled')],
              [Input('coordinates-dropdown', 'value')])
def update_next_button_redirect(components_path):
    if components_path is None:
        return "", True

    m = re.match(r'^f(?P<file>.+);'
                 r'(?P<procedure>mds(?:_v2)?|pcoa)'
                 r'(?:;c(?P<n_components>\d+);m(?P<max_iter>\d+);e(?P<eps>\d+))?$', components_path)

    if m is None:
        return "", True

    result = m.groupdict()
    file = result['file']
    procedure = result['procedure']

    if procedure == "pcoa":
        return f'/pcoa?dataset={file}', False
    elif "mds" == procedure[:3]:
        n_components = result['n_components']
        max_iter = result['max_iter']
        eps = result['eps']
        return f'/optics?dataset={file}&' \
               f'procedure={procedure}&' \
               f'n_components={n_components}&' \
               f'max_iter={max_iter}&' \
               f'eps={eps}', False


@app.callback(
    [Output('coordinates-datatable', 'columns'),
     Output('coordinates-datatable', 'data'),
     Output('mds-statistics', 'children')],
    [
        # Input('mds-datatable', "page_current"),
        # Input('mds-datatable', "page_size"),
        # changed to preview
        Input('coordinates-dropdown', 'value')
    ])
def update_table(
        # page_current,
        # page_size,
        # changed to preview
        components_path
):
    if components_path is None:
        raise PreventUpdate

    m = re.match(r'^f(?P<file>.+);'
                 r'(?P<procedure>mds(?:_v2)?|pcoa)'
                 r'(?:;c(?P<n_components>\d+);m(?P<max_iter>\d+);e(?P<eps>\d+))?$', components_path)

    if m is None:
        raise PreventUpdate

    result = m.groupdict()
    file = result['file']
    procedure = result['procedure']

    if procedure == "pcoa":
        hdf_key = "/components_pcoa/coordinates"
    elif "mds" == procedure[:3]:
        n_components = result['n_components']
        max_iter = result['max_iter']
        eps = result['eps']
        hdf_key = f"mds/c{n_components}/m{max_iter}/e{eps}"

        if "v2" == procedure[-2:]:
            hdf_key = f"components_{hdf_key}/coordinates"
            stats_hdf_key = hdf_key.replace("coordinates", "statistics")

    hdf_filename = f'/output/{file}'

    with pd.HDFStore(hdf_filename, 'r') as store:
        mds_stats = None
        df = store[hdf_key]
        if procedure == "mds_v2":
            stats = store[stats_hdf_key]
            mds_stats = [
                html.P(f"Stress: {stats['stress']:.2f}"),
                html.P(f"Iterations needed: {int(stats['n_iter'])}")
            ]

    # columns = [{"name": str(i), "id": str(i)} for i in sorted(df.columns)]
    # changed to preview
    columns = [{"name": str(i), "id": str(i)} for i in df.columns][:10]
    columns.insert(0, {"name": "Index", "id": "Index"})
    # selected = df.iloc[
    #            page_current * page_size:(page_current + 1) * page_size
    #            ]
    # changed to preview
    selected = df.iloc[:20]
    selected.index = selected.index.rename("Index")
    return columns, selected.reset_index().to_dict('records'), mds_stats
