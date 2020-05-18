#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import math


# In[3]:


ticker = ["AMZN","JPM","BAC","AAPL","WFC","MSFT","JNJ","^DJI","^GSPC","^RUT","^IXIC"]


start = datetime.datetime(1985,1,1)
end = datetime.datetime(2018,12,31)


dataAdjClose = pdr.DataReader(ticker, 'yahoo',start,end)["Adj Close"]
dataOpen = pdr.DataReader(ticker, 'yahoo',start,end)["Open"]
dataClose = pdr.DataReader(ticker, 'yahoo',start,end)["Close"]


# In[4]:


dataLogReturns = np.log(dataClose)-np.log(dataOpen)


# In[5]:


for i in ticker:
    dataLogReturns = dataLogReturns.rename(columns={i: i+"_returns"})


# In[6]:


dataLogReturns = dataLogReturns.assign(DateTime=dataLogReturns.index)
filter1 = dataLogReturns["DateTime"].dt.day_name()
dataLogReturns= dataLogReturns.assign(Day = filter1)
filter2 = dataLogReturns["DateTime"].dt.month_name()
dataLogReturns= dataLogReturns.assign(Month = filter2)
filter3 = dataLogReturns["DateTime"].dt.year
dataLogReturns= dataLogReturns.assign(Year = filter3)


# In[7]:


month = ["April", "May","June","July","August","September", "October","November","December","January","February","March"]


# In[8]:


dataLogReturnsmonthly = dataLogReturns.groupby([dataLogReturns.Year,dataLogReturns.Month]).sum()
dataLogReturnsmonthly = dataLogReturnsmonthly.reindex(month,level=1)
dataLogReturnsmonthly.reset_index(inplace=True)


# In[9]:


dataLogReturnsmonthly.head()


# In[ ]:





# In[ ]:





# In[10]:


dataLogReturnsmonthly = dataLogReturns.groupby([dataLogReturns.Year,dataLogReturns.Month]).sum()
dataLogReturnsmonthly = dataLogReturnsmonthly.reindex(month,level=1)
dataLogReturnsmonthly.reset_index(inplace=True)


# In[11]:


mont = ["January","February","March","April", "May","June","July","August","September", "October","November","December"]
test = list()
for i in mont:
    root=list()
    filter = dataLogReturnsmonthly[dataLogReturnsmonthly.Month == i]
    Jan = filter.iloc[:,2:]
    Jan.reset_index(inplace=True)
    Jan = Jan.drop(columns="index")
    Janmean = Jan.mean()
    Janstd = Jan.std()
    n=len(Jan)

    filter = dataLogReturnsmonthly[dataLogReturnsmonthly.Month != i]
    NotJan = filter.iloc[:,2:]
    NotJan.reset_index(inplace=True)
    NotJan = NotJan.drop(columns="index")
    NotJanmean = NotJan.mean()
    NotJanstd = NotJan.std()
    m=len(NotJan)
    
    diff = (Janmean - NotJanmean)
    
    for j in range(0,len(Janstd)):
        root.append(math.sqrt(((Janstd[j]**2)/n)+((NotJanstd[j]**2)/m))) 
    ts = diff[-4:]/root[-4:]
    test.append(ts)
Test = list(test)
Test


# In[12]:


hypothesis=[['Accepted', 'Accepted', 'Accepted', 'Accepted', 'Accepted',
        'Rejected', 'Accepted', 'Rejected', 'Rejected', 'Accepted',
        'Rejected', 'Accepted'],
       ['Accepted', 'Accepted', 'Accepted', 'Accepted', 'Accepted',
        'Accepted', 'Accepted', 'Accepted', 'Rejected', 'Accepted',
        'Accepted', 'Accepted'],
       ['Accepted', 'Accepted', 'Accepted', 'Accepted', 'Accepted',
        'Accepted', 'Accepted', 'Accepted', 'Accepted', 'Accepted',
        'Accepted', 'Accepted'],
       ['Accepted', 'Accepted', 'Accepted', 'Accepted', 'Accepted',
        'Accepted', 'Accepted', 'Accepted', 'Accepted', 'Accepted',
        'Accepted', 'Rejected']]


# In[13]:


hy=np.array(hypothesis)
hy.T.shape


# In[14]:


a=[]
for i in range(0,12):
        if abs(Test[i][0]) >1.64:
            a.append("Valid")
        else:
            a.append("not so convincing")


# In[15]:


eff=[['not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'Valid',
  'not so convincing',
  'Valid',
  'Valid',
  'not so convincing',
  'Valid',
  'not so convincing'],
 ['not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'Valid',
  'not so convincing',
  'not so convincing',
  'not so convincing'],
 ['not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing'],
 ['not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'not so convincing',
  'Valid'],]


# In[16]:


a12=[]
a12.append(a)
a12


# In[17]:


b12=[]
b12.append(a)
b12


# In[18]:


e=np.array(eff)
e.T.shape


# In[19]:


Test1=np.array(Test)
Test1.shape


# In[20]:




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
nav=Navbar()
options = []

effect=["January Effect","February Effect","March Effect","April Effect","May Effect","June Effect","July Effect","August Effect","September Effect","October Effect","November Effect","Decemeber Effect"]
val=[0,1,2,3,4,5,6,7,8,9,10,11]
j=0
for i in effect:
        #{'label': 'user sees', 'value': 'script sees'}
        mydict = {}
        mydict['label'] = i #Apple Co. AAPL
        mydict['value'] = val[j]
        j=j+1
        options.append(mydict)
j=[]
body = html.Div([
                    html.H1('Analysis of Calender Effect on Stock Prices'),
                    dcc.Markdown(''' --- '''), 
                    html.H1('Monthly Effects Analysis', style={'color':'black','background-color':'lightblue'}),
                    html.Div([html.H3('Enter a Effect:', style={'paddingRight': '30px'}),
                    dcc.Dropdown(
                              id='my_effect',
                              options = options,
                              value = 0, 
                              multi = False
                              # style={'fontSize': 24, 'width': 75}
                    )
                    

                    ], style={'display': 'inline-block', 'verticalAlign':'top', 'width': '30%', 'margin-bottom':'30px'}),
                    html.Div([
                        html.Button(id='submit-button',
                                    n_clicks = 0,
                                    children = 'Submit',
                                    style = {'fontSize': 24, 'marginLeft': '30px'}

                        )

                    ], style={'display': 'inline-block'}),
                    html.Div(id="out", children=[]
                ),
                    dcc.Markdown(''' --- '''),
                   html.P('Hence we can see that September gives use the lowest returns, hence September effect is visbile upto some extent.',style={'color':'white'}),
                   html.P('Rusell 500 index shows December effect, which has the highest returns.', style={'color':'lightblue'}),
                   




                    dcc.Markdown(''' --- '''), 



    ])




def App2():
    layout= html.Div([
    nav,
    body
    ])
    return layout


def update_figure2(n_clicks,value):
    table_header = [ html.Thead(html.Tr([html.Th("Stock"), html.Th("Z Score"),html.Th("Hypothesis Result(CI=90%)"),html.Th("Effect Result")]))]
    i=value
    row1 = html.Tr([html.Td("DGI"), html.Td(Test[i][0]),html.Td("Hence we have "+hy.T[i][0]+" the null hypothesis"),html.Td("Hence this effect is: "+e.T[i][0])])
    row2 = html.Tr([html.Td("GSPC"), html.Td(Test[i][1]),html.Td("Hence we have "+hy.T[i][1]+" the null hypothesis"),html.Td("Hence this effect is: "+e.T[i][1]) ])
    row3 = html.Tr([html.Td("IXIC"), html.Td(Test[i][2]),html.Td("Hence we have "+hy.T[i][2]+" the null hypothesis"),html.Td("Hence this effect is: "+e.T[i][2])])
    row4 = html.Tr([html.Td("RUT"), html.Td(Test[i][3]),html.Td("Hence we have "+hy.T[i][3]+" the null hypothesis"),html.Td("Hence this effect is: "+e.T[i][3]) ])

    table_body = [html.Tbody([row1, row2, row3, row4])]

    table = dbc.Table(table_header + table_body, bordered=True,hover=True, dark=True)
        
    return table


# In[ ]:





# In[ ]:




