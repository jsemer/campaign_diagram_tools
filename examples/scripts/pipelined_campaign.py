#!/usr/bin/env python
# coding: utf-8


# In[2]:


#
# Import the campaign diagram class
#
import sys

sys.path.append("../../")
from campaign_diagram import *


# In[3]:


# Example usage with multiple Kernel instances

kernel1a = Kernel(name='EinsumA',
                  duration=3,
                  compute_util=0.7,
                  bw_util=0.25)

kernel1b = Kernel(name='EinsumB',
                  duration=10,
                  compute_util=0.2,
                  bw_util=0.9)

kernel1c = Kernel(name='EinsumC',
                  duration=2,
                  compute_util=0.6,
                  bw_util=0.4)


# In[4]:


# Create the plot with a list of kernel instances

cascade1 = Cascade([kernel1a, kernel1b, kernel1c])

CampaignDiagram(cascade1).draw()


# In[5]:


# Define tiles for pipeline example

repeat = 4

kernel2a = Kernel(name='EinsumA',
                  duration=5,
                  compute_util=0.8,
                  bw_util=0.2)


kernel2b = Kernel(name='EinsumB',
                  duration=10,
                  compute_util=0.2,
                  bw_util=0.6)

original_cascade = Cascade([kernel2a, kernel2b])



# In[6]:


CampaignDiagram(original_cascade).draw()


# In[7]:


# Unpipelined processing

tiled_cascade = original_cascade.split(repeat)
CampaignDiagram(tiled_cascade).draw()


# In[8]:


# Pipelined diagram (undilated)

pipelined_cascade = tiled_cascade.pipeline()
    
CampaignDiagram(pipelined_cascade).draw()


# In[9]:


# Pipelined diagram (dilated)

pipelined_cascade2 = tiled_cascade.pipeline(spread=True)
#print(f"{pipelinedcascade}")
    
CampaignDiagram(pipelined_cascade2).draw()


# In[ ]:





# In[ ]:





# In[ ]:




