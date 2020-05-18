#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[16]:


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
import datetime as datetime
from io import BytesIO
import base64
from scipy.stats import linregress
from sklearn.metrics import r2_score

# In[11]:


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


# In[12]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.graph_objs as go


# In[13]:


ticker = ["AMZN","JPM","BAC","AAPL","WFC","T","MSFT","GOOGL","JNJ","^GSPC"]


# In[14]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# In[15]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas_datareader.data as web
import plotly.graph_objs as go
from datetime import datetime
import pandas as pd
import numpy as np

def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)
# In[16]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
options = []


# In[22]:

nav=Navbar()
for tic in ticker:
        #{'label': 'user sees', 'value': 'script sees'}
        mydict = {}
        mydict['label'] = tic #Apple Co. AAPL
        mydict['value'] = tic
        options.append(mydict)
body = html.Div([
                    html.H1('Analysis of Calender Effect on Stock Prices'),
                    dcc.Markdown(''' --- '''), 
                    html.H1('Probability Plots',style={'color':'black','background-color':'lightblue'}),
                    html.Div([html.H4('Enter single/multiple stocks symbols to create a ticker:', style={'paddingRight': '30px'}),
                    dcc.Dropdown(
                              id='my_ticker',
                              options = options,
                              value = ['AMZN'], 
                              multi = True
                              # style={'fontSize': 24, 'width': 75}
                    ),
                              
                    html.H4('Enter Stock (make sure it is in ticker):',style={'paddingRight': '30px'})

                    ], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                    html.Div([html.H3('Enter start / end date:'),
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
                              style={'margin-top':"30px"}
                    )
                              ], style={'display':'inline-block','padding-left':'30px','padding-right':'30px'}),
                    html.Div([
                        html.Button(id='submit-button',
                                    n_clicks = 0,
                                    children = 'Submit',
                                    style = {'fontSize': 24, 'marginLeft': '30px'}

                        )

                    ], style={'display': 'inline-block'}),
                   html.Div([html.Img(id='prob_plot',src='')],
                             id='p_graph'
                ),
                    dcc.Markdown(''' --- '''),




                    dcc.Markdown(''' --- '''), 

                    # Cumulative Returns Over Time section



    ])




def App1():
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
def update_figure1(n_clicks,value, start_date,end_date, my_ticker_symbol):
        dataAdjClose = pdr.DataReader(value, 'yahoo',start_date,end_date)["Adj Close"]
        dataOpen = pdr.DataReader(value, 'yahoo',start_date,end_date)["Open"]
        dataClose = pdr.DataReader(value, 'yahoo',start_date,end_date)["Close"]
        dataLogReturns = np.log(dataClose)-np.log(dataOpen)
        month = ["April", "May","June","July","August","September", "October","November","December","January","February","March"]
        for i in value:
            dataLogReturns = dataLogReturns.rename(columns={i: i+"_returns"})
        dataLogReturns = dataLogReturns.assign(DateTime=dataLogReturns.index)
        filter1 = dataLogReturns["DateTime"].dt.day_name()
        dataLogReturns= dataLogReturns.assign(Day = filter1)
        filter2 = dataLogReturns["DateTime"].dt.month_name()
        dataLogReturns= dataLogReturns.assign(Month = filter2)
        filter3 = dataLogReturns["DateTime"].dt.year
        dataLogReturns= dataLogReturns.assign(Year = filter3)
        
        j=my_ticker_symbol+"_returns"
        fig=plt.figure(figsize=(10,10))
        stats.probplot(dataLogReturns[j], dist="norm", fit=True, rvalue=True, plot=plt)
        plt.title(my_ticker_symbol+" Probability Table")
        plt.xlabel('Theoretical quantiles')
        plt.ylabel('Ordered Values')
        fig1=fig_to_uri(fig)
        return fig1

    


# In[ ]:





# In[ ]:




