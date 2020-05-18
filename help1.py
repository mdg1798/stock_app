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
                     html.P(
                         """\
 Welcome to the help manual for the app:"""
                           ),
                     html.P("""1). First select the stocks you want to add to your ticker.""",style={'color':'white'}),
                     html.P("""2). Select a timeline, please include an entire year for proper analysis.""",style={'color':'pink'}),
                     html.P("""3). Select the stock symbol which you want to analyze, make sure you had included that stock in your ticker or else, no result will be generated.""",style={'color':'lightblue'}),
                     html.P("""4). Click on the submit button, it sometimes takes some time to generate the result."""),
                     html.P("""Thank You for using our app, if you have any queries, reach us at:""",style={'color':'yellow'}),
                     html.P("""Modhuli Goswami- mdg2197@columbia.edu""",style={'color':'green'}),
               
                   ],
                  md=4,
               ),
              dbc.Col(
                 [
                     html.H2("Stock Market Analysis"),
                     html.Img(id = 'pic1', src =out_url),
                        ]
                     ),
                ]
            )
       ],
className="mt-4",
)


# In[41]:


def Help():
    layout = html.Div([
    nav,
    body
    ])
    return layout


# In[42]:


app = dash.Dash(__name__, external_stylesheets = [dbc.themes.UNITED])
app.layout = Help()
if __name__ == "__main__":
    app.run_server()


# In[ ]:





# In[ ]:





# In[ ]:




