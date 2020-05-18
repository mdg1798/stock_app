#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

# In[77]:


import numpy as np  
import pandas as pd  
from pandas_datareader import data as wb  
import matplotlib.pyplot as plt  
from scipy.stats import norm, iqr
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
#import datetime as datetime


# In[78]:


### Data
import pandas as pd
import pickle
### Graphing
import plotly.graph_objects as go
### Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
## Navbar
from navbar import Navbar


# In[79]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas_datareader.data as web
import plotly.graph_objs as go
from datetime import datetime
import pandas as pd
import numpy as np
from navbar import Navbar


# In[80]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.graph_objs as go


# In[81]:


ticker = ["AMZN","JPM","BAC","AAPL","WFC","T","MSFT","GOOGL","JNJ","^GSPC"]


# In[82]:


options=[]
options1=[]
month = ["April", "May","June","July","August","September", "October","November","December","January","February","March"]
for tic in ticker:
        #{'label': 'user sees', 'value': 'script sees'}
        mydict = {}
        mydict['label'] = tic #Apple Co. AAPL
        mydict['value'] = tic
        options.append(mydict)
for m in month:
        #{'label': 'user sees', 'value': 'script sees'}
        mydict1 = {}
        mydict1['label'] = m #Apple Co. AAPL
        mydict1['value'] = m
        options1.append(mydict1)
nav = Navbar()
body = html.Div([
                    html.H1('Analysis of Calender Effect on Stock Prices'),
                    dcc.Markdown(''' --- '''), 
                    html.H1('Relative Returns Comparison', style={'background-color':'lightblue','color':'white'}),
                    html.Div([html.H4('Enter single/multiple stocks symbols to create a ticker:',style={'paddingRight': '30px'}),
                    dcc.Dropdown(
                              id='my_ticker',
                              options = options,
                              value = ['AMZN'], 
                              multi = True
                              # style={'fontSize': 24, 'width': 75}
                    ),
                    
                    html.H4('Enter Stock(make sure it is in ticker):',style={'paddingRight': '30px'}),
                    html.H4('Enter single/multiple month(s) for analysis:')],style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                   html.Div([
                       html.H4('Enter start/end date:(please inlcude a entire year for proper analysis)'),
                        dcc.DatePickerRange(id='my_date_picker',
                                            min_date_allowed = datetime(1990,1,1),
                                            max_date_allowed = datetime.today(),
                                            start_date = datetime(2010, 1, 1),
                                            end_date = datetime.today()
                        ),
                       dcc.Dropdown(
                              id='my_ticker_symbol',
                              options = options,
                              value ='AMZN', 
                              multi = False,
                              style={'margin-top':'30px'}
                    ),

                    dcc.Dropdown(
                              id='my_month',
                              options = options1,
                              value =['January'], 
                              multi = True,
                              style={'margin-top':'30px'}
                    )

                    
                    ], style={'display':'inline-block', 'padding-left':'30px','padding-right':'15px'}), 
                    html.Div([
                        html.Button(id='submit-button',
                                    n_clicks = 0,
                                    children = 'Submit',
                                    style = {'fontSize': 24, 'marginLeft': '30px'}

                        )

                    ], style={'display': 'inline-block'}),
                        dcc.Graph(id='graph1'),
                    dcc.Markdown(''' --- '''),




                    dcc.Markdown(''' --- '''), 

                    # Cumulative Returns Over Time section



    ])




def App():
    layout= html.Div([
    nav,
    body
    ])
    return layout


#     @app.callback(Output('my_graph1', 'figure'),
#                     [Input('submit-button', 'n_clicks')],
#                     [State('my_ticker_symbol', 'value'),
#                           State('my_date_picker', 'start_date'),
#                           State('my_date_picker', 'end_date'),
#                           State('yaxis-type','value'),

#                     ])
def update_figure(n_clicks, value, start_date,end_date,my_ticker_symbol, my_month):
        dataOpen = pdr.DataReader(value, 'yahoo',start_date,end_date)["Open"]
        dataClose = pdr.DataReader(value, 'yahoo',start_date,end_date)["Close"]
        dataLogReturns = np.log(dataClose)-np.log(dataOpen)
        for i in value:
            dataLogReturns = dataLogReturns.rename(columns={i: i+"_returns"})
        
        dataLogReturns = dataLogReturns.assign(DateTime=dataLogReturns.index)
        filter1 = dataLogReturns["DateTime"].dt.day_name()
        dataLogReturns= dataLogReturns.assign(Day = filter1)
        filter2 = dataLogReturns["DateTime"].dt.month_name()
        dataLogReturns= dataLogReturns.assign(Month = filter2)
        filter3 = dataLogReturns["DateTime"].dt.year
        dataLogReturns= dataLogReturns.assign(Year = filter3)


        dataLogReturnsmonthly = dataLogReturns.groupby([dataLogReturns.Year,dataLogReturns.Month]).sum()
        dataLogReturnsmonthly = dataLogReturnsmonthly.reindex(month,level=1)
        dataLogReturnsmonthly.reset_index(inplace=True)

    
        j=my_ticker_symbol+'_returns'
        traces=[]
        for i in my_month:
                    
                    df_by_month = dataLogReturnsmonthly[dataLogReturnsmonthly['Month'] == i]
                    traces.append(dict(
                        x=df_by_month['Year'],

                        y=df_by_month[j],
                        #text=df_by_continent['country'],
                        mode='markers',
                        name=i
                    ))

        return {
            'data': traces,
            'layout': dict(
                #title={'text':j},
                #yaxis={'title': 'Years',},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest',
                transition = {'duration': 500},
            )
        }
    # @app.callback(Output('container', 'children'), [Input('submit-button', 'n_clicks')])
    # def display_graphs(n_clicks):
    #     graphs = []
    #     for i in range(n_clicks):
    #         a=update_figure(n_clicks, value, start_date,end_date)
    #         graphs.append(a)
    #     return html.Div(graphs)
   # return app


# In[ ]:





# In[ ]:





# In[ ]:




