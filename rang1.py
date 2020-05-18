#!/usr/bin/env python
# coding: utf-8

# In[13]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import math
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
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output , State
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
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

ticker = ["AMZN","JPM","BAC","AAPL","WFC","MSFT","JNJ","^DJI","^GSPC","^RUT","^IXIC"]


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


# In[16]:



# In[2]:


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


# In[9]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
options = []


nav=Navbar()
for tic in ticker:
        #{'label': 'user sees', 'value': 'script sees'}
        mydict = {}
        mydict['label'] = tic #Apple Co. AAPL
        mydict['value'] = tic
        options.append(mydict)
body = html.Div([   
                    html.H1('Analysis of Calender Effect on Stock Prices',style={'color':'white'}),
                    dcc.Markdown(''' --- '''), 
                    html.H2('Range of Mean and Variance with Confindence Interval as User Input', style={'color':'black','background-color':'lightblue'}),
                    html.Div([html.H4('Enter single/multiple stocks symbols to create a ticker:', style={'paddingRight': '30px','color':'#7B7D7D'}),
                    dcc.Dropdown(
                              id='my_ticker',
                              options = options,
                              value = ["AMZN","JPM"], 
                              multi = True
                              # style={'fontSize': 24, 'width': 75}
                    ),#,style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                    html.H4('Select a confidence interval:                            ',style={'paddingRight': '30px'}),
                    dcc.Input(id='confidence', value=0.95, type="number"),
                    

                    ], style={'display': 'inline-block', 'verticalAlign':'top'}),
                    html.Div([html.H4('Enter start / end date (please include an entire year for proper analysis:'),
                        dcc.DatePickerRange(id='my_date_picker',
                                            min_date_allowed = datetime(2000,1,1),
                                            max_date_allowed = datetime.today(),
                                            start_date = datetime(2000, 1, 1),
                                            end_date = datetime.today()
                        )],style={'display': 'inline-block', 'verticalAlign':'top'}),

                    html.Div([html.H4 ('Enter a stock symbol for analysis (make sure its in the ticker in the above selection):',style={'paddingRight': '30px'}),
                    dcc.Dropdown(
                              id='my_ticker_symbol',
                              options = options,
                              value ='AMZN', 
                              multi = False
                              # style={'fontSize': 24, 'width': 75}
                    )
                              ], style={'display':'inline-block'}),
                    html.Div([
                        html.Button(id='submit-button',
                                    n_clicks = 0,
                                    children = 'Submit',
                                    style = {'fontSize': 24, 'marginLeft': '30px'}

                        )

                    ], style={'display': 'inline-block'}),
                    html.Div(dcc.Markdown(''' --- '''),),
                    
                 html.Div(id="r", children=[]
                           ),
                    dcc.Markdown(''' --- '''),




                    dcc.Markdown(''' --- '''), 

                    # Cumulative Returns Over Time section



    ])




def App8():
    layout= html.Div([
    nav,
    body
    ])
    return layout


#   @app.callback(Output('my_graph1', 'figure'),
#                     [Input('submit-button', 'n_clicks')],
#                     [State('my_ticker_symbol', 'value'),
#                           State('my_date_picker', 'start_date'),
#                           State('my_date_picker', 'end_date'),
#                           State('yaxis-type','value'),

#                     ])
def update_figure8(n_clicks,value, start_date,end_date, my_ticker_symbol,confidence):
        month = ["January","February","March","April", "May","June","July","August","September", "October","November","December"]
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
        i=my_ticker_symbol+"_returns"
        res_mean, res_var, res_std = stats.bayes_mvs(dataLogReturns[i], alpha=confidence)
        res_mean1=list(res_mean.minmax)
        res_var1=list(res_var.minmax)
        table_header = [ html.Thead(html.Tr([html.Th("Stock"), html.Th("Mean Range (Min,Max)"),html.Th("Variance Range (Min,Max)")]))]
        row1 = html.Tr([html.Td(my_ticker_symbol), html.Td("Min:"+str(res_mean1[0])+" Max:"+str(res_mean1[1])),html.Td("Min:"+str(res_var1[0])+" Max:"+str(res_var1[1]))])
        
        table_body = [html.Tbody([row1])]

        table = dbc.Table(table_header + table_body, bordered=True , dark=True, hover=True)
        return table
    

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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




