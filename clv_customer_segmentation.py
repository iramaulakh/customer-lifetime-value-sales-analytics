#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas numpy scikit-learn matplotlib plotly dash


# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import plotly.express as px
from IPython.display import display

#step 1 - load and filter dataset
file_path = "Sales data - Order Delivered.csv"  #replace with your file
df = pd.read_csv(file_path, low_memory=False)
df = df[df['order_status'] == 'COMPLETED'].copy()
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df_relevant = df[['store_id', 'order_date', 'gross_gmv']].dropna()

#step 2 - aggregate customer behavior
clv = df_relevant.groupby('store_id').agg(
    total_revenue=('gross_gmv', 'sum'),
    order_count=('gross_gmv', 'count'),
    avg_order_value=('gross_gmv', 'mean'),
    first_order=('order_date', 'min'),
    last_order=('order_date', 'max')
).reset_index()

#step 3 - calculate recency, tenure, and CLV
max_date = df_relevant['order_date'].max()
clv['recency_days'] = (max_date - clv['last_order']).dt.days
clv['tenure_days'] = (clv['last_order'] - clv['first_order']).dt.days.fillna(0)
clv['estimated_clv'] = clv['avg_order_value'] * clv['order_count']

#step 4 - RFM Scores
clv['R'] = pd.qcut(clv['recency_days'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
clv['F'] = pd.qcut(clv['order_count'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
clv['M'] = pd.qcut(clv['total_revenue'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
clv['RFM_Score'] = clv['R'].astype(str) + clv['F'].astype(str) + clv['M'].astype(str)

#step 5 - KMeans clustering into 10 segments
features = clv[['estimated_clv', 'order_count', 'recency_days']]
features_scaled = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clv['clv_cluster'] = kmeans.fit_predict(features_scaled)

cluster_map = clv.groupby('clv_cluster')['estimated_clv'].mean().sort_values().reset_index()
cluster_map['segment'] = [f'Segment {i+1}' for i in range(10)]
clv = clv.merge(cluster_map[['clv_cluster', 'segment']], on='clv_cluster')

#step 6 - predict future CLV using linear regression
reg = LinearRegression()
X = clv[['order_count']]
y = clv['estimated_clv']
reg.fit(X, y)
clv['predicted_clv'] = reg.predict(X)

#step 7 - save CSV
clv.to_csv("customer_clv_full_analysis.csv", index=False)

#step 8 - plot charts
fig1 = px.histogram(clv, x='segment', title="Customer Segments by CLV")
fig2 = px.scatter(clv, x='order_count', y='estimated_clv', color='segment', title="CLV vs Order Count")
fig3 = px.box(clv, x='segment', y='recency_days', title="Recency Distribution by Segment")
fig4 = px.scatter(clv, x='order_count', y='predicted_clv', title="Predicted Future CLV")
fig5 = px.histogram(clv, x='RFM_Score', title="RFM Score Frequency")


fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()


# In[3]:


#########TOP 5 Customers #######################
import plotly.express as px

#top 5 highest CLV customers
top_5_summary = clv.sort_values(by='estimated_clv', ascending=False).head(5)
top_5_summary = top_5_summary[['store_id', 'segment', 'estimated_clv', 'order_count', 'recency_days']]

#Top 5 Customers by Estimated CLV
fig_top5 = px.bar(
    top_5_summary.sort_values(by='estimated_clv'),
    x='estimated_clv',
    y='store_id',
    color='segment',
    orientation='h',
    title='Top 5 Customers by Estimated CLV',
    labels={'estimated_clv': 'Estimated CLV', 'store_id': 'Customer ID'}
)
fig_top5.show()


# In[4]:


######### REpeated Buyers #########################


#flitering High CLV (above median) AND multiple orders (>1)
high_clv_multi_orders = clv[
    (clv['estimated_clv'] > clv['estimated_clv'].median()) &
    (clv['order_count'] > 1)
]
#scatter plot for high-value repeat buyers
fig_repeat = px.scatter(
    high_clv_multi_orders,
    x='order_count',
    y='estimated_clv',
    color='segment',
    size='recency_days',  #bubble size reflects recency (lower is better)
    hover_data=['store_id', 'recency_days'],
    title='High-Value Repeat Buyers: CLV vs Order Count',
    labels={'order_count': 'Number of Orders', 'estimated_clv': 'Estimated CLV'}
)
fig_repeat.update_traces(marker=dict(sizemode='diameter', sizeref=1))
fig_repeat.show()


#top 10 matching customers
high_clv_multi_orders[['store_id', 'segment', 'estimated_clv', 'order_count', 'recency_days']].head(10)



# In[5]:


############################ CHURN RISK CUSTOMERS #################

#thresholds
clv_threshold = clv['estimated_clv'].median()
recency_threshold = clv['recency_days'].quantile(0.75)

#filter: High CLV + High Recency = Churn Risk
churn_risk = clv[
    (clv['estimated_clv'] > clv_threshold) &
    (clv['recency_days'] > recency_threshold)
]


import plotly.express as px

fig_churn = px.scatter(
    churn_risk,
    x='recency_days',
    y='estimated_clv',
    color='segment',
    hover_data=['store_id', 'order_count'],
    title='🛑 Churn Risk Customers: High CLV but Inactive',
    labels={'recency_days': 'Days Since Last Purchase', 'estimated_clv': 'Estimated CLV'}
)
fig_churn.show()
#top churn-risk customers
churn_risk[['store_id', 'segment', 'estimated_clv', 'order_count', 'recency_days']].head(10)


# In[ ]:





# In[ ]:




