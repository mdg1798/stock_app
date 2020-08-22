import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output , State
import dash_bootstrap_components as dbc
from returns1 import App, update_figure
from prob1 import App1, update_figure1
from Effects1 import App2, update_figure2
from hist1 import App3, update_figure3
from linear_regress1 import App4, update_figure4
from bestreturn1 import App5, update_figure5
from compare1 import App6,update_figure6
from ci1 import App7, update_figure7 
from rang1 import App8, update_figure8
from help1 import Help
from homepage import Homepage
import os
import flask
# In[2]:

server=flask.Flask(__name__)
#server.secret_key=os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR],server=server)


# In[3]:


app.config.suppress_callback_exceptions = True


# In[4]:


app.layout = html.Div([
    dcc.Location(id = 'url', refresh = False),
    html.Div(id = 'page-content')
])


# In[5]:


@app.callback(Output('page-content', 'children'),
            [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/logreturns':
        return App()
    elif pathname == '/probplot':
        return App1()
    elif pathname == '/histplot':
        return App3()
    elif  pathname =='/effect':
        return App2()
    elif pathname == '/residual':
        return App4()
    elif pathname == '/bestreturn':
        return App5()
    elif pathname == '/compare':
        return App6()
    elif pathname == '/population':
        return App7()
    elif pathname == '/range':
        return App8()
    elif pathname == '/help':
        return Help()
    else:
        return Homepage()


# In[6]:


@app.callback(Output('r', 'children'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_ticker', 'value'),
                          State('my_date_picker', 'start_date'),
                     State('my_date_picker', 'end_date'),
                     State('my_ticker_symbol','value'),
                     State('confidence','value'),
                    ])
def update_graph8(n_clicks, value, start_date,end_date,my_ticker_symbol, confidence):
    graph8 = update_figure8(n_clicks, value, start_date,end_date,my_ticker_symbol,confidence)
    return graph8


# In[7]:


@app.callback(Output('out_t', 'children'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_ticker', 'value'),
                          State('my_date_picker', 'start_date'),
                     State('my_date_picker', 'end_date'),
                     State('my_ticker_symbol1','value'),
                     State('my_ticker_symbol2','value'),
                    ])
def update_graph7(n_clicks, value, start_date,end_date,my_ticker_symbol1,my_ticker_symbol2):
    graph7 = update_figure7(n_clicks, value, start_date,end_date,my_ticker_symbol1,my_ticker_symbol2)
    return graph7


# In[8]:


@app.callback(Output('two_plot', 'src'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_ticker', 'value'),
                          State('my_date_picker', 'start_date'),
                     State('my_date_picker', 'end_date'),
                     State('my_ticker_symbol1','value'),
                     State('my_ticker_symbol2','value'),
                    ])
def update_graph6(n_clicks, value, start_date,end_date,my_ticker_symbol1,my_ticker_symbol2):
    graph6 = update_figure6(n_clicks, value, start_date,end_date,my_ticker_symbol1,my_ticker_symbol2)
    return graph6


# In[9]:


@app.callback(Output('b_plot', 'src'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_ticker', 'value'),
                          State('my_date_picker', 'start_date'),
                     State('my_date_picker', 'end_date'),
                     State('my_ticker_symbol','value'),
                    ])
def update_graph5(n_clicks, value, start_date,end_date,my_ticker_symbol):
    graph5 = update_figure5(n_clicks, value, start_date,end_date,my_ticker_symbol)
    return graph5


# In[10]:


@app.callback(Output('cur_plot', 'src'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_ticker', 'value'),
                          State('my_date_picker', 'start_date'),
                     State('my_date_picker', 'end_date'),
                     State('my_ticker_symbol','value'),
                    ])
def update_graph4(n_clicks, value, start_date,end_date,my_ticker_symbol):
    graph4 = update_figure4(n_clicks, value, start_date,end_date,my_ticker_symbol)
    return graph4


# In[11]:


@app.callback(Output('output', 'children'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_ticker', 'value'),
                          State('my_date_picker', 'start_date'),
                     State('my_date_picker', 'end_date'),
                     State('my_ticker_symbol','value'),
                    ])
def update_graph1(n_clicks, value, start_date,end_date,my_ticker_symbol):
    graph1 = update_figure1(n_clicks, value, start_date,end_date,my_ticker_symbol)
    return graph1


# In[12]:


@app.callback(Output('out', 'children'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_effect', 'value'),
                        
                    ])
def update_graph2(n_clicks, value):
    table1 = update_figure2(n_clicks, value)
    return table1


# In[13]:


####r_1

@app.callback(Output('graph1', 'figure'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_ticker', 'value'),
                          State('my_date_picker', 'start_date'),
                     State('my_date_picker', 'end_date'),
                     State('my_ticker_symbol','value'),
                     State('my_month','value')
                    ])
def update_graph(n_clicks, value, start_date,end_date,my_ticker_symbol,month1):
    graph = update_figure(n_clicks, value, start_date,end_date,my_ticker_symbol,month1)
    return graph


# In[14]:


@app.callback(Output('prob_plot', 'src'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_ticker', 'value'),
                          State('my_date_picker', 'start_date'),
                     State('my_date_picker', 'end_date'),
                     State('my_ticker_symbol','value'),
                    ])
def update_graph1(n_clicks, value, start_date,end_date,my_ticker_symbol):
    graph11 = update_figure1(n_clicks, value, start_date,end_date,my_ticker_symbol)
    return graph11


# In[15]:


@app.callback(Output('hist_graph', 'children'),
                    [Input('submit-button', 'n_clicks')],
                    [State('my_ticker', 'value'),
                          State('my_date_picker', 'start_date'),
                     State('my_date_picker', 'end_date'),
                     State('my_ticker_symbol','value'),
                    ])
def update_graph12(n_clicks, value, start_date,end_date,my_ticker_symbol):
    graph12 = update_figure3(n_clicks, value, start_date,end_date,my_ticker_symbol)
    return graph12


# In[16]:


#if __name__ == '__main__':
#    app.run_server(debug=True)


# In[17]:


if __name__ == "__main__":
    #app.run_server(port=8017)
    app.server.run(host='0.0.0.0')


# In[ ]:





# In[ ]:




