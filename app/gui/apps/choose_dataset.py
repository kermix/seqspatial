import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from pathlib import Path
import pandas as pd

from app import app

PAGE_SIZE = 15


def get_datasets(directory=None):
    if directory is None:
        directory = Path.cwd()
    files = [{
        'name': x.name,
        'path': str(x)
    } for x in Path(directory).iterdir() if x.is_file()]
    return files


def get_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Choose dataset file"),
                html.P("Choose dataset file from input directory.")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Select(
                    id='dataset-dropdown',
                    options=[
                        {'label': p['name'], 'value': p['name']} for p in get_datasets('/output')
                    ]
                ),
                dbc.Button("Next", id="next-button", size="lg", outline=True, color="primary",
                           className="float-right mt-1",
                           n_clicks=0, disabled=True)
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.P("Data preview:"),
                dash_table.DataTable(
                    id='dataset-datatable',
                    page_current=0,
                    page_size=PAGE_SIZE,
                    page_action='custom',
                    style_table={"overflowX": "auto"}

                )
            ],
            )
        ]),

    ])


@app.callback([Output('next-button', 'href'),
               Output('next-button', 'disabled')],
              [Input('dataset-dropdown', 'value')])
def update_next_button_redirect(input_file):
    if not input_file:
        return '', True
    return f'/coordinates?dataset={input_file}', False


@app.callback(
    [Output('dataset-datatable', 'columns'),
     Output('dataset-datatable', 'data')],
    [Input('dataset-datatable', "page_current"),
     Input('dataset-datatable', "page_size"),
     Input('dataset-dropdown', 'value')])
def update_table(page_current, page_size, input_file):
    if input_file is None:
        raise PreventUpdate
    
    hdf_key = 'input_data'
    hdf_filename = f'/output/{input_file}'

    with pd.HDFStore(hdf_filename, 'r') as store:
        df = store[hdf_key]

    columns = [{"name": str(i[0]), "id": str(i[1])} for i in zip(df.index, sorted(df.columns))]
    columns.insert(0, {"name": "Index", "id": "0"})
    selected = df.iloc[
               page_current * page_size:(page_current + 1) * page_size
               ]
    return columns, selected.reset_index().to_dict('records')
