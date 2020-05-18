#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:


import dash_bootstrap_components as dbc
def Navbar():
    navbar = dbc.NavbarSimple(
           children=[
              dbc.NavItem(dbc.NavLink("Help Manual", href="/help")),
              dbc.NavItem(dbc.NavLink("Range of mean and variance", href="/range")),
              dbc.NavItem(dbc.NavLink("Monthly Effects", href="/effect")),
              dbc.NavItem(dbc.NavLink("Comparsion of Population Mean", href="/population")),
              dbc.NavItem(dbc.NavLink("Best Monthly Returns",href='/bestreturn')),
              dbc.DropdownMenu(
                 nav=True,
                 in_navbar=True,
                 label="Plots",
                 children=[
                    dbc.DropdownMenuItem("Log Returns Plot", href="/logreturns"),
                    dbc.DropdownMenuItem("Probability Plot",href="/probplot"),
                    dbc.DropdownMenuItem("Histogram Plot",href="/histplot"),
                          ],
                      ),
               dbc.DropdownMenu(
                 nav=True,
                 in_navbar=True,
                 label="Regression Model",
                 children=[
                    dbc.DropdownMenuItem("Residuals and Returns Plot ", href="/residual"),
                    dbc.DropdownMenuItem("Comparison Plots",href="/compare"),
                          ],
                      ),
                    ],
          brand="Home",
          brand_href="/home",
          sticky="top",
        )
    return navbar


# In[ ]:





# In[ ]:





# In[ ]:




