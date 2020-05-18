#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[28]:


import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_gif_component as Gif
from dash.dependencies import Input, Output , State
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from io import BytesIO
import base64


# In[29]:


from navbar import Navbar
nav = Navbar()


# In[35]:


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


# In[36]:


img = mpimg.imread('./my_g.png')
fig= plt.figure(figsize=(7,5))
plt.imshow(img)
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_visible(False)
cur_axes.axes.get_yaxis().set_visible(False)
out_url = fig_to_uri(fig)


# In[40]:


body = dbc.Container(
    [
       dbc.Row(
           [
               dbc.Col(
                  [
                     html.H2("Welcome"),
                     html.P(""" To our Stock Analysis App, the user can select their tickers, individual stocks, compare them, analyze monthly effects on indexes...   """,style={'color':'white'}),
                     html.P(""" Perform Regression Analysis and see the trends by analyzing  over 20 years of data... """,style={'color':'pink'}),
                     html.P(""" Checkout all the different graphs for analysis and find the best month for your investment...""",style={'color':'lightblue'}),
                     html.P(""" If you have any issue while running the components of the app, please refer to the help manual in the navigation bar.""", style={'color':'orange'})
                   ],
                  md=4,
               ),
              dbc.Col(
                 [
                     html.H2("Stock Market Analysis"),
                     html.Img(id = 'pic', src =out_url),
                        ]
                     ),
                ]
            )
       ],
className="mt-4",
)


# In[41]:


def Homepage():
    layout = html.Div([
    nav,
    body
    ])
    return layout


# In[42]:


app = dash.Dash(__name__, external_stylesheets = [dbc.themes.UNITED])
app.layout = Homepage()
if __name__ == "__main__":
    app.run_server()


# In[ ]:





# In[ ]:





# In[ ]:




