import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


def get_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.P([
                    "Error 404. Analysis not found.",
                    html.Br(),
                    html.A("Go to homepage >", href="/", style={"text-align": "center"}),
                    ],
                    style={"text-align": "center", "font-size": "3rem"}),


            ], align="center")
        ], style={"height": "100vh"}),
    ], fluid=True, style={"height": "100vh"})
