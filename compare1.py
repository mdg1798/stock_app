#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:


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



# In[3]:


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


# In[8]:


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
options = []

nav=Navbar()
# In[22]:


for tic in ticker:
        #{'label': 'user sees', 'value': 'script sees'}
        mydict = {}
        mydict['label'] = tic #Apple Co. AAPL
        mydict['value'] = tic
        options.append(mydict)
body = html.Div([
                    html.H1('Analysis of Calender Effect on Stock Prices'),
                    dcc.Markdown(''' --- '''), 
                    html.H1('Comparisons between regression of stocks',style={'color':'black','background-color':'lightblue'}),
                    html.Div([html.H4('Enter single/multiple stocks symbols to create a ticker:', style={'paddingRight': '30px'}),
                    dcc.Dropdown(
                              id='my_ticker',
                              options = options,
                              value = ["AMZN","JPM"], 
                              multi = True
                              # style={'fontSize': 24, 'width': 75}
                    ),
                     html.H4('Enter Stock1 (make sure it is in ticker):',style={'paddingRight': '30px'}),
                    html.H4('Enter Stock2 (make sure it is in ticker):',style={'paddingRight': '30px'})
                    

                    ], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%'}),
                    html.Div([html.H3('Enter start/end date(please include an entire year for proper analysis:'),
                        dcc.DatePickerRange(id='my_date_picker',
                                            min_date_allowed = datetime(2000,1,1),
                                            max_date_allowed = datetime.today(),
                                            start_date = datetime(2000, 1, 1),
                                            end_date = datetime.today()
                        ),

                     
                    dcc.Dropdown(
                              id='my_ticker_symbol1',
                              options = options,
                              value ='AMZN', 
                              multi = False,
                              style={'margin-top':'30px'}
                    ),
                              dcc.Dropdown(
                              id='my_ticker_symbol2',
                              options = options,
                              value ='JPM', 
                              multi = False,
                              style={'margin-top':'30px'}
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
                    html.Div([html.Img(id='two_plot',src='')],
                             id='two_graph'
                ),
                    dcc.Markdown(''' --- '''),




                    dcc.Markdown(''' --- '''), 

                    # Cumulative Returns Over Time section



    ])




def App6():
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
def update_figure6(n_clicks,value, start_date,end_date, my_ticker_symbol1,my_ticker_symbol2):
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
    
        i=my_ticker_symbol1+"_returns"
        j=my_ticker_symbol2+"_returns"
        X = dataLogReturns[i].values.reshape(-1, 1)  # values converts it into a numpy array
        Y = dataLogReturns[j].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        fig=plt.figure(figsize=(10,10))
        slope = linear_regressor.coef_
        intercept = linear_regressor.intercept_
        r2 = r2_score(Y, Y_pred, sample_weight=None, multioutput='uniform_average')  
        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.title(my_ticker_symbol1+" and "+my_ticker_symbol2+" - regression of the log-returns.")
        plt.xlabel(i)
        plt.ylabel(j)
        plt.text(-0.2,0.75,"Slope:"+str(slope[0][0])+"\nIntercept:"+str(intercept[0])+"\nR2 Score:"+str(r2),fontsize=10)
        
        out_url=fig_to_uri(fig)

        
        
        return out_url

    

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




