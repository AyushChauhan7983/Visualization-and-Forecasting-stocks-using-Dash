from datetime import datetime as dt

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# model
from model import prediction


def get_stock_price_fig(df):
    fig = go.Figure(go.Candlestick(
        x=df.Date,
        open=df.Open,
        high=df.High,
        low=df.Low,
        close=df.Close

    ))

    return fig


def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                     x="Date",
                     y="EWA_20",
                     title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig


app = dash.Dash(
    __name__,
    external_stylesheets=[
      #  "https://fonts.googleapis.com/css2?family=Roboto&display=swap"
    ])
server = app.server
# html layout of site
app.layout = html.Div(
    [
        html.Div(
            [
                # Navigation
                #Heading of the page
                html.H1(children= "Welcome to the Stock Dash App", className="start",style={
                    'text-align' : 'center',
                    'color' : '#000000'
                }),

                #Text area to input the stock code
                html.Div([
                    html.P("Input stock code: ",style={
                        'margin' : '5px',
                        'fontSize' : 20
                    }),
                    html.Div([
                        dcc.Input(id="dropdown_tickers", type="text",placeholder='enter the stock code',required=True,style={
                            'margin' : '5px',
                            'width' : '180px',
                            'height' : '25px',
                        }),
                        html.Button("Submit", id='submit',style={
                            'margin' : '4px',
                            'color' : 'blue'
                        }),
                    ],
                             className="form")
                ],
                         className="input-place"),

                html.Br(),
                #Date-Picker-Range to pick the date
                html.P('Pick the start date', style={
                    'margin': '5px',
                    'fontSize': 20
                }),
                html.Div([
                    dcc.DatePickerRange(id='my-date-picker-range',
                                        min_date_allowed=dt(2000, 1, 1),
                                        max_date_allowed=dt.now(),
                                        initial_visible_month=dt.now(),
                                        end_date=dt.now().date()),
                ],
                         className="date"),

                html.Br(),
                html.Div([
                html.P('Click the "Stock Price" button to visualise the stock',style={
                        'margin' : '5px',
                        'fontSize' : 20
                    }),
                    #Button to show stock price
                    html.Button(
                        "Stock Price", className="stock-btn", id="stock",style=
                    {
                       'margin' : '5px',
                       'color': 'blue',
                       'fontSize': 20}
                    ),

                html.Br(),
                html.Br(),
                html.P('Click the "Indicators" button to plot the EMA vs Date graph',style={
                        'margin' : '5px',
                        'fontSize' : 20
                    }),
                    #Button to indicate the
                    html.Button("Indicators",
                                className="indicators-btn",
                                id="indicators",
                                style={
                                    'margin' : '5px',
                                    'color' : 'blue',
                                    'fontSize' : 20
                                }),

                    html.Br(),
                    html.Br(),
                    html.P('Input number of days:',style={
                        'margin' : '5px',
                        'fontSize' : 20
                    }),
                    dcc.Input(id="n_days",
                              type="text",
                              placeholder="enter the number of days",
                              required=True,
                              style={
                              'margin' : '5px',
                              'width' : '180px',
                              'height' : '25px'
                        }),

                    #Button to forecast the stock
                    html.Button(
                        "Forecast", className="forecast-btn", id="forecast",style={
                            'margin' : '4px',
                            'color' : 'blue'
                        })
                ],
                         className="buttons"),
                # here
            ],
            className="nav"),

        # content
        html.Div(
            [
                html.Div(
                    [  # header
                        html.Img(id="logo",style={
                            'margin-top' : '15px'
                        }),
                        html.P(id="ticker")
                    ],
                    className="header"),
                html.Div(id="description", className="decription_ticker"),
                html.Div([], id="graphs-content"),
                html.Div([], id="main-content"),
                html.Div([], id="forecast-content")
            ],
            className="content"),
    ],
    className="container")


# callback for company info
@app.callback([
    Output("description", "children"),
    Output("logo", "src"),
    Output("ticker", "children"),
    Output("stock", "n_clicks"),
    Output("indicators", "n_clicks"),
    Output("forecast", "n_clicks")
], [Input("submit", "n_clicks")], [State("dropdown_tickers", "value")])


def update_data(n, val):  # inpur parameter(s)
    if n == None:
        return "Hey there! Please enter a legitimate stock code to get details.", "https://melmagazine.com/wp-content/uploads/2019/07/Screen-Shot-2019-07-31-at-5.47.12-PM.png", "Stonks", None, None, None
        # raise PreventUpdate
    else:
        if val == None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info
            df = pd.DataFrame().from_dict(inf, orient="index").T
            var = df[['logo_url', 'shortName', 'longBusinessSummary']]
            return df['longBusinessSummary'].values[0], df['logo_url'].values[
                0], df['shortName'].values[0], None, None, None


# callback for stock graphs
@app.callback([
    Output("graphs-content", "children"),
], [
    Input("stock", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])

#method to show the stock price
def stock_price(n, start_date, end_date, val):
    if n == None:
        return [""]
        #raise PreventUpdate
    if val == None:
        raise PreventUpdate
    else:
        if start_date != None:
            df = yf.download(val, str(start_date), str(end_date))
        else:
            df = yf.download(val)

    df.reset_index(inplace=True)
    fig = get_stock_price_fig(df)
    return [dcc.Graph(figure=fig)]


# callback for indicators
@app.callback([Output("main-content", "children")], [
    Input("indicators", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])

#method to show the indicators
def indicators(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        return [""]

    if start_date == None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]


# callback for forecast
@app.callback([Output("forecast-content", "children")],
              [Input("forecast", "n_clicks")],
              [State("n_days", "value"),
               State("dropdown_tickers", "value")])

#method to forecast
def forecast(n, n_days, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)
    return [dcc.Graph(figure=fig)]

#method for prediction
def prediction(stock, n_days):
    import yfinance as yf
    import plotly.graph_objs as go
    # model
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR
    from datetime import date, timedelta
    # load the data

    df = yf.download(stock, period='60d')
    df.reset_index(inplace=True)
    df['Day'] = df.index

    days = list()
    for i in range(len(df.Day)):
        days.append([i])

    # Splitting the dataset

    X = days
    Y = df[['Close']]

    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.1,shuffle=False)

    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.001, 0.01, 0.1, 1, 100, 1000],
            'epsilon': [
                0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,
                50, 100, 150, 1000
            ],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]
        },
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1)

    y_train = y_train.values.ravel()
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf',
                   C=best_params["C"],
                   epsilon=best_params["epsilon"],
                   gamma=best_params["gamma"],
                   max_iter=-1)

    # Support Vector Regression Models

    # RBF model
    # rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=4.0)
    rbf_svr = best_svr

    rbf_svr.fit(x_train, y_train)

    output_days = list()
    for i in range(1, n_days):
        output_days.append([i + x_test[-1][0]])

    dates = []
    current = date.today()
    for i in range(n_days):
        current += timedelta(days=1)
        dates.append(current)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rbf_svr.predict(output_days),
            mode='lines+markers',
            name='data'))
    fig.update_layout(
        title="Predicted Close Price of next " + str(n_days - 1) + " days",
        xaxis_title="Date",
        yaxis_title="Closed Price",
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)