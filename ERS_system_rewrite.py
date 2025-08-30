from utiliti_functions.compare_images import *
from utiliti_functions.compare_tables import *
from utiliti_functions.compare_texts import *
from utiliti_functions.database_management import *
from utiliti_functions.extract_normalize_tables import *
from utiliti_functions.pdf_management import *
from utiliti_functions.processing_lines import *
from dotenv import load_dotenv
import tempfile, zipfile, base64, io
import pandas as pd
from datetime import datetime
import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dash_table, html, dcc
from flask import Flask, send_from_directory
from psycopg2 import sql
from psycopg2 import *

def run_web():

    load_dotenv(dotenv_path=r"C:\Users\phungdoan\.vscode\project\ERS\ERS system source code\.env")

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname_ERS = os.getenv("DB_NAME_ERS")
    dbname_LWS = os.getenv("DB_NAME_LWS")
    dbname_PTC = os.getenv("DB_NAME_PTC")
    dbname_PDF = os.getenv("DB_NAME_PDF")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    temp_folder = tempfile.mkdtemp()

    # Initialize Flask and Dash with a professional theme
    app = Flask(__name__)
    dash_app = Dash(__name__, server=app, url_base_pathname="/", external_stylesheets=[dbc.themes.ZEPHYR])

    # Fetch schema names
    schema_names = get_schema_names(host=host, port=port, database=dbname_ERS, username=username, password=password)
    filtered_schema_names = [schema_name for schema_name in schema_names if schema_name not in ['pg_toast', 'pg_catalog', 'information_schema', 'public']]
    schema_names_with_labels = [{"label": html.Span([schema_name + ".pdf"], style={'color': 'Red'}), "value": schema_name + ".pdf"} for schema_name in filtered_schema_names]

    # Fetch lws name
    # lws_names = get_table_names('public', host=host, port=port, database=dbname_LWS, username=username, password=password)
    # lws_names_with_label = [{"label": html.Span([lws_name + ".csv"], style={'color': 'Green'}), "value": lws_name + ".csv"} for lws_name in lws_names]
 
    files_name = schema_names_with_labels# + lws_names_with_label

    files_name = sorted(files_name, key=lambda x: x['value'])

    # Fetch image schema names
    schema_image_names = get_schema_names(host=host, port=port, database=dbname_PTC, username=username, password=password)
    filtered_schema_image_names = [schema_name for schema_name in schema_image_names if schema_name not in ['pg_toast', 'pg_catalog', 'information_schema', 'public']]
    schema_image_names_with_labels = [{"label": html.Span([schema_name + ".pdf"], style={'color': 'Red'}), "value": schema_name + ".pdf"} for schema_name in filtered_schema_image_names]

    credential = {"admin":'ERS9999'}
    # Define a consistent color palette
    COLORS = {
        'primary': '#1e5185',      # Dark blue
        'secondary': '#3d8bd4',    # Medium blue
        'accent': '#e9f2fd',       # Light blue
        'light': '#f6f9fc',        # Nearly white
        'dark': '#263238',         # Nearly black
        'success': '#28a745',      # Green
        'warning': '#ffc107',      # Yellow
        'danger': '#dc3545'        # Red
    }
    
    dash_app.title = "ERS Comparison System"
    
    login_page = html.Div([
                html.Div([
                    html.Div([
                        html.P("Username:", style={"margin-right": "10px"}),  # Add some spacing between text and input
                        dbc.Input(
                            id="username", 
                            placeholder="Enter your Username", 
                            type="text", 
                            style={"flex": "1"}  # Allows input to stretch
                        )],
                        style={
                            "display": "flex",
                            "align-items": "center",  # Vertically aligns content
                            "margin-bottom": "20px"  # Space between username and password sections
                        },
                    ),
                    html.Div([
                        html.P("Password:", style={"margin-right": "10px"}),  # Add some spacing between text and input
                        dbc.Input(
                            id="password", 
                            placeholder="Enter your Password", 
                            type="password", 
                            style={"flex": "1"}  # Allows input to stretch
                        )],
                        style={
                            "display": "flex",
                            "align-items": "center",  # Vertically aligns content
                            "margin-bottom": "20px"  # Space below the password section
                        },
                    ),
                    dbc.Button("Login", id="login-button", color="primary", style={"width": "100%"}),
                    html.Div(id="login-status")
                    ],
                    style={
                        "border": "2px solid black",  
                        "border-radius": "10px",  
                        "padding": "30px",  
                        "width": "600px",
                        "background-color": "#f9f9f9"  
                    }
                )],
                id="login-page",
                style={
                    "display": "flex",
                    "justify-content": "center",  # Horizontally center the login form
                    "align-items": "center",  # Vertically center the login form
                    "height": "100vh",  # Full height of the viewport
                    "background-color": "#e0f7fa"  # Optional background color for the page
                }
            )

    main_page = html.Div([
                dcc.Store(id='docs-store', data=[]),  # Store for uploaded documents
                # Navigation sidebar
                html.Div(
                    dbc.Nav([
                        dbc.NavLink("Import", href="/import", active="exact", 
                                style={"fontSize": "16px", "fontWeight": "500", "padding": "12px 15px"}),
                        dbc.NavLink("Compare", href="/compare", active="exact", 
                                style={"fontSize": "16px", "fontWeight": "500", "padding": "12px 15px"}),
                        dbc.NavLink("Export Tables", href="/export-tables", active="exact", 
                                style={"fontSize": "16px", "fontWeight": "500", "padding": "12px 15px"}),
                        dbc.NavLink("Compare Images", href="/compare-images", active="exact", 
                                style={"fontSize": "16px", "fontWeight": "500", "padding": "12px 15px"}),
                        dbc.NavLink("Compare Text", href="/compare-texts", active="exact", 
                                style={"fontSize": "16px", "fontWeight": "500", "padding": "12px 15px"}),
                        dbc.Button(
                            "Logout", 
                            id='logout-button',
                            color="primary",
                            style={
                                'color': 'white', 
                                "borderRadius": "8px", 
                                "marginTop": "auto"})
                    ],
                    vertical=True,
                    pills=True,
                    style={
                        "borderRadius": "8px",
                        "backgroundColor": COLORS['light'],
                        "height": "100%",
                        "width": "180px",
                        "boxShadow": "0 2px 5px rgba(0,0,0,0.05)",
                        "padding": "15px 0",
                        "marginRight" : "10px"
                    }
                    )
                ),
                
                # Main content container
                html.Div( 
                    children=[
                        # Import tab
                        html.Div([
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Upload(
                                            id="Upload-data",
                                            children=dbc.Button(
                                                [html.I(className="fas fa-file-upload me-2"), "Import Files"],
                                                id="upload-button", 
                                                color="primary", 
                                                size="lg", 
                                                className="mb-3",
                                                style={"width": "200px"}
                                            ),
                                            multiple=True
                                        ),
                                        width=3,
                                        style={"textAlign": "left"}
                                    )
                                ],
                                justify="start",
                                className="m-3"
                            ),
                            html.Div(id="message-board", style={"position": "absolute", "top": "30%", "left": "50%"}),
                            dcc.Loading(
                                id="loading-upload",
                                type="default",
                                children=[
                                dash_table.DataTable(
                                    id="datatable-upload-container",
                                    style_table={
                                        "width": "97.5%", 
                                        "margin": "auto",
                                        "overflowX": "auto",
                                        "borderRadius": "8px",
                                        "boxShadow": "0 2px 5px rgba(0,0,0,0.05)"
                                    },
                                    style_header={
                                        "backgroundColor": COLORS['primary'],
                                        "color": "white",
                                        "textAlign": "center", 
                                        "fontWeight": "bold",
                                        "fontSize": "16px",
                                        "padding": "12px"
                                    },
                                    style_cell={
                                        "textAlign": "left", 
                                        "padding": "12px 15px",
                                        "fontFamily": "sans-serif"
                                    },
                                    style_data_conditional=[
                                        {
                                            "if": {"row_index": "odd"},
                                            "backgroundColor": COLORS['accent'],  
                                        },
                                        {
                                            "if": {"state": "selected"},
                                            "backgroundColor": COLORS['secondary'],
                                            "color": "white"
                                        }
                                    ])]
                                )],
                            id="import-tab",
                            style={"display": "block"}
                            ),
                        
                        # Compare tab
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Select Files To Compare Tables", 
                                            style={"backgroundColor": COLORS['primary'], "color": "white", "fontSize": "18px"}),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id='file-compare-dropdown-list',
                                        options=files_name,
                                        placeholder="Select files to compare (minimum 2)",
                                        multi=True,
                                        className="mb-4",
                                        style={
                                            "width": "100%",
                                            "borderRadius": "4px",
                                        }
                                    ),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Button(
                                                [html.I(className="fas fa-sync-alt me-2"), "Reflect Doc"], 
                                                id="Reflect-button", 
                                                color="primary", 
                                                n_clicks=0, 
                                                className="me-2",
                                                style={"width": "160px"}
                                            ), 
                                            width="auto"
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [html.I(className="fas fa-file-archive me-2"), "Export Excel"], 
                                                id="Output-Zip-button", 
                                                color="success", 
                                                n_clicks=0, 
                                                disabled=True,
                                                style={"width": "160px"}
                                            ), 
                                            width="auto"
                                        ),
                                    ],
                                    justify="center", 
                                    className="mb-4"),
                                ])
                            ], className="mb-4"),
                            
                            dcc.Loading(
                                id="loading-compare",
                                type="default",
                                children=[                          
                                    dbc.Card([
                                        dbc.CardHeader("Comparison Results", 
                                                    style={"backgroundColor": COLORS['primary'], "color": "white", "fontSize": "18px"}),
                                        dbc.CardBody([
                                            html.Iframe(
                                                id='iframe-url', 
                                                src='', 
                                                style={
                                                    "width": "100%", 
                                                    "height": "400px", 
                                                    "border": "none", 
                                                    "borderRadius": "4px",
                                                    "backgroundColor": "white"
                                                }
                                            ),
                                        ]),
                                    ])
                                ]),                            
                            dcc.Download(id="download"),
                            html.Div(id='Output-message', className="mt-3")
                            ],
                            id="compare-tab",
                            style={"display": "none"},
                            ),
                        
                        # Export tables tab
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Export Tables", 
                                            style={"backgroundColor": COLORS['primary'], "color": "white", "fontSize": "18px"}),
                                dbc.CardBody([
                                    html.P("Select files to export their tables:", className="text-muted mb-3"),
                                    dcc.Dropdown(
                                        id='file-export-dropdown-list',
                                        options=schema_names_with_labels,
                                        placeholder="Select files",
                                        multi=True,
                                        className="mb-4",
                                        style={"width": "100%"}
                                    ),
                                    dbc.Button(
                                        [html.I(className="fas fa-file-export me-2"), "Export Tables"], 
                                        id="Output-Tables-button", 
                                        color="primary", 
                                        n_clicks=0,
                                        style={"margin": "0 auto", "display": "block", "width": "200px"}
                                    ),                                    
                                    dcc.Download(id="download-tables"),
                                    dcc.Loading(
                                        id="loading-export",
                                        type="default",
                                        children=[
                                            html.Div(
                                                id='Output-export-message', 
                                                className="mt-4")]
                                    )
                                ])
                            ])
                            ],
                            id="export-tables-tab",
                            style={"display": "none"}
                            ),

                        # Compare images tab
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Select Files to Compare image", 
                                            style={"backgroundColor": COLORS['primary'], "color": "white", "fontSize": "18px"}),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id='file-compare-images-dropdown-list',
                                        options=schema_image_names_with_labels,
                                        placeholder="Select 2 files to compare image",     
                                        multi = True,                                   
                                        className="mb-2",
                                        style={
                                            "width": "100%",
                                            "borderRadius": "4px",
                                        }
                                    ),
                                    dcc.Loading(
                                        id='image-dropdown-loading',
                                        children=[
                                            dcc.Dropdown(
                                                id='image-name-drop-down',
                                                options=[],
                                                placeholder="Select an image name to compare image",                                        
                                                className="mb-2",
                                                style={
                                                    "width": "100%",
                                                    "borderRadius": "4px",
                                                }
                                            ),
                                        ]
                                    ),                                    
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Button(
                                                [html.I(className="fas fa-file-export me-1"), "Compare Images"], 
                                                id="Compare-images-button", 
                                                color="primary", 
                                                n_clicks=0,
                                                style={"margin": "0 auto", "display": "block", "width": "200px"}),
                                            width = "auto"
                                            ),
                                        dbc.Col(
                                            dbc.Button(
                                                [html.I(className="fas fa-file-export me-1"), "Export Images"], 
                                                id="Export-images-button", 
                                                color="success", 
                                                n_clicks=0,
                                                style={"margin": "0 auto", "display": "block", "width": "200px"}),
                                            width = "auto"   
                                            ),
                                        dcc.Download(id="download-image")                                   
                                        ], 
                                        justify = "center",                                        
                                    ),
                                    dcc.Loading(
                                        id='Image-compare-loading',
                                        children=[
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.CardImg(
                                                        id = "result-image",
                                                        src = "",
                                                        style = {"width": "100%", "heigh":"auto", "border":"1px solid"}
                                                    )
                                                ]),
                                            ],
                                            className="mt-1",
                                            style={"height":"100%"}
                                            )
                                        ]
                                    )                                    
                                    ,
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button(
                                                "<<",
                                                id="previous-image-button",
                                                color="secondary",
                                                className ='m-0'
                                            ),
                                        ],
                                        width=1,
                                        style={"display":"flex", "justify-content":"end"}),
                                        dbc.Col([
                                            dbc.Input(
                                                id="input-picture-index",
                                                type="number",
                                                min=1,
                                                max=100,
                                                style={"border" : "1px solid", "width": "70px"}
                                            )
                                        ],
                                        width=1,
                                        style={"display":"flex", "justify-content":"center"}),
                                        dbc.Col([
                                            dbc.Button(
                                                ">>",
                                                id="next-image-button",
                                                color="secondary",
                                                className ='m-0'
                                            )
                                        ],
                                        width=1,
                                        style={"display":"flex", "justify-content":"start"})                                       
                                    ],  
                                    justify="center",
                                    className='g-0 mt-2')
                                ])
                            ])
                            ],
                            id="compare-image-tab",
                            style={"display": "none"}
                            ),
                        
                        # Compare texts tab
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Select Files To Compare Texts", 
                                            style={"backgroundColor": COLORS['primary'], "color": "white", "fontSize": "18px"}),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id='file-compare-text-dropdown-list',
                                        options=schema_names_with_labels,
                                        placeholder="Select files to compare (minimum 2)",
                                        multi=True,
                                        className="mb-4",
                                        style={
                                            "width": "100%",
                                            "borderRadius": "4px",
                                        }
                                    ),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Button(
                                                [html.I(className="fas fa-sync-alt me-2"), "Reflect Doc"], 
                                                id="Reflect-button-for-text", 
                                                color="primary", 
                                                n_clicks=0, 
                                                className="me-2",
                                                style={"width": "160px"}
                                            ), 
                                            width="auto"
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [html.I(className="fas fa-file-archive me-2"), "Export Excel"], 
                                                id="Output-Zip-button-for-text", 
                                                color="success", 
                                                n_clicks=0, 
                                                disabled=True,
                                                style={"width": "160px"}
                                            ), 
                                            width="auto"
                                        ),
                                    ],
                                    justify="center", 
                                    className="mb-4"),
                                ])
                            ], className="mb-4"),
                            
                            dcc.Loading(
                                id="loading-compare-text",
                                type="default",
                                children=[                          
                                    dbc.Card([
                                        dbc.CardHeader("Comparison Results", 
                                                    style={"backgroundColor": COLORS['primary'], "color": "white", "fontSize": "18px"}),
                                        dbc.CardBody([
                                            html.Iframe(
                                                id='iframe-url-for-text', 
                                                src='', 
                                                style={
                                                    "width": "100%", 
                                                    "height": "400px", 
                                                    "border": "none", 
                                                    "borderRadius": "4px",
                                                    "backgroundColor": "white"
                                                }
                                            ),
                                        ]),
                                    ])
                                ]),                            
                            dcc.Download(id="download-for-text"),
                            html.Div(id='Output-message-for-text', className="mt-3")
                            ],
                            id="compare-text-tab",
                            style={"display": "none"},
                            ),
                                           
                        dcc.Store(id='img_names'),
                        dcc.Store(id='convert_names')
                    ],
                    style={
                        "borderRadius": "8px",
                        "backgroundColor": "white",
                        "padding": "20px",
                        "width": "calc(100% - 200px)",
                        "minHeight": "calc(100vh - 200px)",
                        "boxShadow": "0 2px 10px rgba(0,0,0,0.05)"
                    })
                ],
                id="main-page",
                style={
                "display": "flex",
                "backgroundColor": COLORS['accent'], 
                "padding": "20px",
                "borderRadius": "8px",
                "boxShadow": "0 2px 10px rgba(0,0,0,0.1)",
                }
            )

    # Main layout
    dash_app.layout = dbc.Container([
        dcc.Location(id='url', pathname='/login', refresh=False),
        # Header
        html.Div([
            html.H1(
                "ERS COMPARISON SYSTEM", 
                style={
                    "height": "80px",
                    "fontSize": "42px",  
                    "fontWeight": "600",  
                    "textAlign": "center", 
                    "color": "white",
                    "backgroundColor": COLORS['primary'],  
                    "borderRadius": "8px",        
                    "padding": "15px 0",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "letterSpacing": "1px"
                }
            )
        ], className="mb-4"),
        
        html.Div([
            # Login page
            html.Div(id="login-container", children=[login_page]),
        
            # Main content (tabs, etc)
            html.Div(id="main-container", children=[main_page])
        ], id='main-content',)
    ], fluid=True)

    # Authentication callback
    @dash_app.callback(
        [Output("login-status", "children"),
        Output("login-container", "style"),
        Output("main-container", "style"),
        Output('url', "pathname")],
        [Input("login-button", "n_clicks"),
        Input("logout-button", "n_clicks")],
        [State("username", "value"), 
        State("password", "value")]
    )
    def manage_authentication(login_clicks, logout_clicks, username, password):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "login-button" and login_clicks:
            if username in credential and credential[username] == password:
                return "", {'display':'none'}, {'display':'block'}, '/import'
            else:
                return "Invalid username or password", {'display':'block'}, {'display':'none'}, '/login'
        
        elif trigger_id == "logout-button" and logout_clicks:
            return "", {'display':'block'}, {'display':'none'}, '/login'
        
        # Default case - no buttons clicked yet
        return "", {'display':'block'}, {'display':'none'}, '/login'

    # Tab navigation callback
    @dash_app.callback(
        [Output("import-tab", "style"),
        Output("compare-tab", "style"),        
        Output("export-tables-tab", "style"),
        Output("compare-image-tab", "style")],
        Output("compare-text-tab", "style"),
        Input("url", "pathname")       
    )
    def display_tabs(pathname):
        if pathname == "/compare":
            return {"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}
        elif pathname == "/export-tables":
            return {"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}
        elif pathname == "/compare-images":
            return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}
        elif pathname == "/compare-texts":
            return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}    
    
    # Upload files callback
    @dash_app.callback(
        Output('datatable-upload-container', 'data'),
        Output('datatable-upload-container', 'columns'),
        Output('file-compare-dropdown-list', 'options'),
        Output('file-export-dropdown-list', 'options'),
        Output('file-compare-images-dropdown-list', 'options'),
        Output('file-compare-text-dropdown-list', 'options'),
        Output('message-board', 'children'),
        Input('Upload-data', 'contents'),
        Input('Upload-data', 'filename')
    )
    def upload_table(contents, filename):
        if not contents:
            text_dropdown_options = [pdf for pdf in files_name if 'raw_pdf' in get_table_names(pdf['value'].split(".")[0])]
            return [], [], files_name, files_name, schema_image_names_with_labels, text_dropdown_options, "Your uploaded files will be showed here \u2193"
            
        uploaded_files = []
        for i in range(len(filename)):
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            uploaded_files.append({
                'id': len(uploaded_files) + 1,
                'filename': filename[i],
                'upload_time': current_time
            })
            content_type, content_string = contents[i].split(',')
            decoded = base64.b64decode(content_string)
            file_path = os.path.join(temp_folder, filename[i])
            with open(file_path, "wb") as pdf:
                pdf.write(decoded)
        
        pdf_dropdown_options = files_name + \
                            [{'label': html.Span([file['filename']], style={'color':'Red'}), 'value': file['filename']} if ".pdf" in file['filename']\
                            else {'label': html.Span([file['filename']], style={'color':'Green'}), 'value': file['filename']}
                            for file in uploaded_files]
        pdf_dropdown_options = [dict(t) for t in {tuple(d.items()) for d in pdf_dropdown_options}]

        pdf_dropdown_options = sorted(pdf_dropdown_options, key=lambda x: x['value'])
        
        text_dropdown_options = [pdf for pdf in pdf_dropdown_options if 'raw_pdf' in get_table_names(pdf['value'].split(".")[0]) or pdf['value'] in filename]

        image_dropdown_options = schema_image_names_with_labels + [{'label': html.Span([file['filename']], style={'color':'Red'}), 'value': file['filename']} for file in uploaded_files if ".pdf" in file['filename']]
        
        existing_image_schemas = get_schema_names(host=host, port=port, database=dbname_PTC, username=username, password=password)

        for pdf in image_dropdown_options:
            if pdf['value'].split(".")[0] not in existing_image_schemas:
                pdf_path = os.path.join(temp_folder, pdf['value'])
                extract_images(pdf_path)

        return uploaded_files, [
            {'name': "ID", "id": "id"},
            {"name": "File Name", "id": "filename"},
            {"name": "Upload Time", "id": "upload_time"}
        ], pdf_dropdown_options, pdf_dropdown_options, image_dropdown_options, text_dropdown_options, ""

    # Compare table callback
    @dash_app.callback(
        Output('iframe-url', 'src'),
        Output('Output-Zip-button', 'disabled'),
        State('file-compare-dropdown-list', 'value'),
        Input('Reflect-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def compare_tables(selected_files, n_clicks):
        if not selected_files or len(selected_files) < 2:
            default_path = os.path.join(temp_folder, "no_files_selected.html")
            with open(default_path, "w") as f:
                f.write("""
                <div style="text-align: center; padding: 40px; font-family: Arial, sans-serif;">
                    <h3 style="color: #1e5185;">Please select at least two files and click Reflect</h3>
                    <p style="color: #666;">Use the dropdown above to select PDF files for comparison.</p>
                </div>
                """)
            return f"/temp/no_files_selected.html", True

        docs_name = "_".join([selected_file.split(".")[0] for selected_file in selected_files if selected_file])

        result_filename = f"comparison_results_{docs_name}.html"
        file_path = os.path.join(temp_folder, result_filename)

        if os.path.exists(file_path):
            return f"/temp/{result_filename}", False

        selected_files = [os.path.join(temp_folder, selected_file) for selected_file in selected_files if selected_file]
        
        global current_comparison
        if not any(".csv" in file for file in selected_files):
            current_comparison, notifications = compare_pdf(selected_files)
        elif not any(".pdf" in file for file in selected_files):
            current_comparison, notifications = compare_lsw(selected_files)
        else:
            current_comparison, notifications = compare_lws_to_pdf(selected_files)

        if not current_comparison:
            with open(file_path, "w") as f:
                f.write("""
                <div style="text-align: center; padding: 40px; font-family: Arial, sans-serif;">
                    <h3 style="color: #1e5185;">There is no difference between pdfs within tables</h3>
                </div>
                """)
            return f"/temp/{result_filename}", True
        comparison_html = ""
        for table in current_comparison:
            table.columns = table.columns.str.title()
            comparison_html += table.to_html(index=False, classes="custom-table") + "<br><hr>"
            
        for notification in notifications:
            comparison_html += f"<p>{notification}</p>"

        custom_style = """
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    color: #333;
                    background-color: #f6f9fc;
                    margin: 0;
                    padding: 0;
                    width: 100%;
                }
                .container {
                    width: 100%;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    box-sizing: border-box;
                    padding: 0;
                }
                .custom-table {
                    width: 100%; 
                    margin: 0 0 20px 0;
                    border-collapse: collapse;
                    font-size: 14px;
                    word-wrap: break-word;
                    white-space: nowrap;
                    table-layout: fixed;
                    overflow-wrap: break-word;
                }
                .custom-table th {
                    position: sticky;
                    top: 0;
                    z-index: 100;
                    text-align: center; 
                    font-weight: 600; 
                    background-color: #1e5185; 
                    color: white;
                    padding: 12px; 
                    border: 1px solid #dee2e6;
                    white-space: normal;  
                    word-wrap: break-word;
                }
                .custom-table td {
                    text-align: left; 
                    padding: 10px 12px; 
                    border: 1px solid #dee2e6;
                    vertical-align: top;
                    white-space: normal;
                    word-break: break-word;
                }
                .custom-table tbody tr:nth-of-type(odd) {
                    background-color: #f8fafc;
                }                
                .custom-table tbody tr:hover {
                    background-color: #e9f2fd;
                    cursor: pointer;
                }
                hr {
                    border: 0;
                    height: 1px;
                    background-color: #dee2e6;
                    margin: 30px 0;
                }
                h2 {
                    color: #1e5185;
                    margin-top: 0;
                }
            </style>
        """

        html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Comparison Results</title>
                <style>
                    body, html {{
                        margin: 0;
                        padding: 0;
                        width: 100%;
                        height: 100%;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    }}
                    .container {{
                        width: 100%;
                        min-height: 100%;
                        padding: 20px;
                        box-sizing: border-box;
                        background: white;
                    }}
                    /* Keep your existing table styles */
                </style>
            </head>
            <body>
                <div class="container">
                    <h2><i class="fas fa-table"></i> Comparison Results</h2>
                    {comparison_html}
                </div>
            </body>
            </html>
            """
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(custom_style + html_template)

        return f"/temp/{result_filename}", False

    # Serve temp files
    @dash_app.server.route('/temp/<path:filename>')
    def serve_temp_files(filename):
        return send_from_directory(temp_folder, filename)

    # Output excel result callback
    @dash_app.callback(
        Output('download', 'data', allow_duplicate=True),
        Output('Output-message', 'children', allow_duplicate=True),
        Input('Output-Zip-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def output_excel(n_clicks):
        if n_clicks:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                for i, table in enumerate(current_comparison):
                    if 'Title' in table.columns.tolist():
                        sheet_name = table['Title'].iloc[0]
                    elif 'Contents' not in table.columns.tolist():
                        sheet_name = 'LWS'
                    else:
                        sheet_name = table['Contents'].iloc[0] + " - " + str(i)
                    sheet_name = re.sub(r'[\[\]\:\*\?\/\\]', '_', sheet_name)
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:28] + "..."
                    table.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Add formatting to the Excel workbook
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#1e5185',
                        'font_color': 'white',
                        'border': 1
                    })
                    
                    # Apply header format
                    for col_num, value in enumerate(table.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    
                    # Set column widths
                    for idx, col in enumerate(table.columns):
                        max_len = max(table[col].astype(str).str.len().max(), len(str(col)) + 2)
                        worksheet.set_column(idx, idx, min(max_len, 30))
                        
            excel_buffer.seek(0)
            excel_filename = "comparison_result.xlsx"
            return dcc.send_bytes(excel_buffer.getvalue(), filename=excel_filename), dbc.Alert([
                html.I(className="fas fa-check-circle me-2"), 
                "Excel file downloaded successfully"
            ], color="success", dismissable=True)    
   
    # Output Tables callback
    @dash_app.callback(
        Output('download-tables', 'data', allow_duplicate=True),
        Output('Output-export-message', 'children', allow_duplicate=True),
        Input('Output-Tables-button', 'n_clicks'),
        State('file-export-dropdown-list', 'value'),
        prevent_initial_call=True
    )
    def output_tables(n_clicks, schema_names):
        if n_clicks and schema_names:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, mode='w') as zf:
                for schema_name in schema_names:
                    schema_name = schema_name[:-4]
                    tables_name = get_table_names(schema_name, host=host, port=port, database=dbname_ERS, username=username, password=password)
                    tables = [fetch_data_from_db(table_name, schema_name, host=host, port=port, database=dbname_ERS, username=username, password=password) for table_name in tables_name if table_name != 'raw_pdf']
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        for i, table in enumerate(tables):
                            if 'Title' in table.columns.tolist():
                                sheet_name = table['Title'].iloc[0]
                            else:
                                # print(table.columns)
                                sheet_name = table['contents'].iloc[0] + " - " + str(i)
                            sheet_name = "".join([char if char not in "[]:*?\/" else "_" for char in sheet_name])
                            if len(sheet_name) > 31:
                                sheet_name = sheet_name[:28] + "..."
                            table.to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            # Add Excel formatting
                            workbook = writer.book
                            worksheet = writer.sheets[sheet_name]
                            header_format = workbook.add_format({
                                'bold': True,
                                'bg_color': '#1e5185',
                                'font_color': 'white',
                                'border': 1
                            })
                            
                            # Apply header format
                            for col_num, value in enumerate(table.columns.values):
                                worksheet.write(0, col_num, value, header_format)
                    
                    excel_buffer.seek(0)
                    excel_filename = str(schema_name) + ".xlsx"
                    zf.writestr(excel_filename, excel_buffer.getvalue())
                    
            zip_buffer.seek(0)
            return dcc.send_bytes(zip_buffer.getvalue(), filename="Tables.zip"), dbc.Alert("Tables downloaded successfully", color="success")
        else:
            return "", dbc.Alert("Choose at least 1 pdf", color="success")

    #Update image name dropdown list
    @dash_app.callback(
        Output('image-name-drop-down', 'options'),
        Output('img_names', 'data'),
        Output('convert_names', 'data'),
        Input('file-compare-images-dropdown-list', 'value'),
        prevent_initial_call=True
    )
    def update_image_name_dropdown(schemas):
        convert_dict, image_names = convert_img_name(schemas)
        return image_names, image_names, convert_dict

    # Compare image callback
    @dash_app.callback(
        Output('result-image', 'src', allow_duplicate=True),
        Output('input-picture-index', 'value', allow_duplicate=True),
        Input('Compare-images-button', 'n_clicks'),
        State('file-compare-images-dropdown-list', 'value'),
        State('image-name-drop-down', 'value'),
        State('img_names', 'data'),
        State('convert_names', 'data'),
        prevent_initial_call = True
    )
    def compare_picture(n_clicks, schemas, img_name, image_names, convert_names):
        if n_clicks<=0:
            return "", 0
        
        schemas = [schema.split(".")[0] for schema in schemas]

        if convert_names:
            if len(get_images_name(schemas[0])) >= len(get_images_name(schemas[1])):
                img_name1 = img_name
                img_name2 = convert_names[img_name]
            else:
                img_name1 = convert_names[img_name]
                img_name2 =  img_name
        else:
            img_name1 = img_name
            img_name2 = img_name

        index = image_names.index(img_name)+1

        img1_path = os.path.join(temp_folder, f'img1.{img_name1}')
        img2_path = os.path.join(temp_folder, f'img2.{img_name2}')

        if not os.path.exists(img1_path):
            img1, img1_ext = fetch_image_data_from_db(schemas[0], img_name1)
            with open(img1_path, "wb") as file1:
                file1.write(img1)

        if not os.path.exists(img2_path):
            img2, img2_ext = fetch_image_data_from_db(schemas[1], img_name2)
            with open(img2_path, "wb") as file2:
                file2.write(img2)

        img = ImageRender(img1_path, img2_path)

        data_url = img.get_img_url()

        return data_url, index

    # navigate between pictures
    @dash_app.callback(
        Output('result-image', 'src', allow_duplicate=True),
        Output('input-picture-index', 'value', allow_duplicate=True),
        Output('image-name-drop-down', 'value'),
        State('input-picture-index', 'value'),
        State('file-compare-images-dropdown-list', 'value'),
        State('img_names', 'data'),
        State('convert_names', 'data'),
        Input('input-picture-index', 'n_submit'),
        Input('previous-image-button', 'n_clicks'),
        Input('next-image-button', 'n_clicks'),
        prevent_initial_call = True
    )
    def navigate_between_images(index, schemas, image_names, convert_names, n_sub, p_n_clicks, n_n_clicks):  
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        schemas = [schema.split(".")[0] for schema in schemas]

        max_index = len(image_names)

        if trigger_id == 'input-picture-index':
            pass

        elif trigger_id == 'previous-image-button':
                index -= 1

        elif trigger_id == 'next-image-button':
                index += 1

        if index <= 1:
            index = 1
        
        if index >= max_index:
            index = max_index

        img_name = image_names[index-1]

        if convert_names:
            if len(get_images_name(schemas[0])) >= len(get_images_name(schemas[1])):
                img_name1 = img_name
                img_name2 = convert_names[img_name]
            else:
                img_name1 = convert_names[img_name]
                img_name2 =  img_name
        else:
            img_name1 = img_name
            img_name2 = img_name

        img1_path = os.path.join(temp_folder, f'img1.{img_name1}')
        img2_path = os.path.join(temp_folder, f'img2.{img_name2}')

        if not os.path.exists(img1_path):
            img1, img1_ext = fetch_image_data_from_db(schemas[0], img_name1)
            with open(img1_path, "wb") as file1:
                file1.write(img1)

        if not os.path.exists(img2_path):
            img2, img2_ext = fetch_image_data_from_db(schemas[1], img_name2)
            with open(img2_path, "wb") as file2:
                file2.write(img2)

        img = ImageRender(img1_path, img2_path)

        data_url = img.get_img_url()

        return (
            data_url, 
            index,
            img_name
            )

    # download image
    @dash_app.callback(
        Output('download-image', 'data'),
        Input('Export-images-button', 'n_clicks'),
        State('file-compare-images-dropdown-list', 'value'),
        State('image-name-drop-down','value')
    )
    def output_images(n_clicks, schemas, img_name):
        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if not ctx.triggered:
            return dash.no_update

        schemas = [schema.split(".")[0] for schema in schemas]
        if button_id == 'Export-images-button' and n_clicks:
            merged_img = export_images(schemas[0], schemas[1], img_name)

            img_io = io.BytesIO()
            merged_img.save(img_io, format="PNG")
            img_io.seek(0)

        return dcc.send_bytes(img_io.getvalue(), filename=f'Image comparison.png')

    # Compare texts callback
    @dash_app.callback(
        Output('iframe-url-for-text', 'src'),
        Output('Output-Zip-button-for-text', 'disabled'),
        State('file-compare-text-dropdown-list', 'value'),
        Input('Reflect-button-for-text', 'n_clicks'),
        prevent_initial_call=True
    )
    def compare_pdf_texts(selected_files, n_clicks):                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        if not selected_files or len(selected_files) < 2:
            default_path = os.path.join(temp_folder, "no_files_selected.html")
            with open(default_path, "w") as f:
                f.write("""
                <div style="text-align: center; padding: 40px; font-family: Arial, sans-serif;">
                    <h3 style="color: #1e5185;">Please select at least two files and click Reflect</h3>
                    <p style="color: #666;">Use the dropdown above to select PDF files for comparison.</p>
                </div>
                """)
            return f"/temp/no_files_selected.html", True

        docs = "_".join([selected_file.split(".")[0] for selected_file in selected_files if selected_file])
        result_filename = f"texts_comparison_results_{docs}.html"
        file_path = os.path.join(temp_folder, result_filename)

        if os.path.exists(file_path):
            return f"/temp/{result_filename}", False

        selected_files = [os.path.join(temp_folder, selected_file) for selected_file in selected_files if selected_file]
        
        global text_comparison_result
        text_comparison_result = compare_texts(selected_files)
        if text_comparison_result.empty:
            with open(file_path, "w") as f:
                f.write("""
                <div style="text-align: center; padding: 40px; font-family: Arial, sans-serif;">
                    <h3 style="color: #1e5185;">There is no difference between texts</h3>
                </div>
                """)
            return f"/temp/{result_filename}", True

        text_comparison_result.columns = text_comparison_result.columns.str.title()
        comparison_html = text_comparison_result.to_html(index=False, classes="custom-text") + "<br><hr>"

        # Custom CSS for the text comparison results
        custom_style = """
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    color: #333;
                    background-color: #f6f9fc;
                    margin: 0;
                    padding: 0;
                    width: 100%;
                }
                .container {
                    width: 100%;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    box-sizing: border-box;
                    padding: 0;
                }
                .custom-text {
                    width: 100%; 
                    margin: 0 0 20px 0;
                    border-collapse: collapse;
                    font-size: 14px;
                }
                .custom-text th {
                    position: sticky;
                    top: 0;
                    z-index: 100;
                    text-align: center; 
                    font-weight: 600; 
                    background-color: #1e5185; 
                    color: white;
                    padding: 12px; 
                    border: 1px solid #dee2e6;
                    white-space: normal;       /* Allows wrapping */
                    word-wrap: break-word;     /* Ensures long words break */
                }
                .custom-text td {
                    text-align: left; 
                    padding: 10px 12px; 
                    border: 1px solid #dee2e6;
                    vertical-align: top;
                }
                .custom-text tbody tr:nth-of-type(odd) {
                    background-color: #f8fafc;
                }                
                .custom-text tbody tr:hover {
                    background-color: #e9f2fd;
                    cursor: pointer;
                }
                hr {
                    border: 0;
                    height: 1px;
                    background-color: #dee2e6;
                    margin: 30px 0;
                }
                h2 {
                    color: #1e5185;
                    margin-top: 0;
                }
            </style>
        """
        # Create the HTML template with the comparison results
        html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Comparison Results</title>
                <style>
                    body, html {{
                        margin: 0;
                        padding: 0;
                        width: 100%;
                        height: 100%;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    }}
                    .container {{
                        width: 100%;
                        min-height: 100%;
                        padding: 20px;
                        box-sizing: border-box;
                        background: white;
                    }}
                    /* Keep your existing table styles */
                </style>
            </head>
            <body>
                <div class="container">
                    <h2><i class="fas fa-table"></i> Comparison Results</h2>
                    {comparison_html}
                </div>
            </body>
            </html>
            """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(custom_style + html_template)

        return f"/temp/{result_filename}", False
    
    # Output excel result callback for text comparison
    @dash_app.callback(
        Output('download-for-text', 'data', allow_duplicate=True),
        Output('Output-message-for-text', 'children', allow_duplicate=True),
        Input('Output-Zip-button-for-text', 'n_clicks'),
        prevent_initial_call=True
    )
    def output_excel_for_text(n_clicks):
        if n_clicks:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                text_comparison_result.columns = text_comparison_result.columns.str.title()
                text_comparison_result.to_excel(writer, sheet_name="Text compare result", index=False)
               
                # Set the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets["Text compare result"]
                
                # Add formatting to the Excel workbook
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#1e5185',
                    'font_color': 'white',
                    'border': 1
                })
                # Apply header format
                for col_num, value in enumerate(text_comparison_result.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Set column widths
                for idx, col in enumerate(text_comparison_result.columns):
                    max_len = max(text_comparison_result[col].astype(str).str.len().max(), len(str(col)) + 2)
                    worksheet.set_column(idx, idx, min(max_len, 30))
            
            excel_buffer.seek(0)
            excel_filename = "text_comparison_result.xlsx"
            return dcc.send_bytes(excel_buffer.getvalue(), filename=excel_filename), dbc.Alert([
                html.I(className="fas fa-check-cricle me-2"),
                "Excel file downloaded successfully"
            ], color="success", dismissable=True)

    if __name__ == '__main__':
        app.run(debug=False, port=5000, host='0.0.0.0')

run_web()