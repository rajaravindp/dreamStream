import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

logo_path = "tsla_logo.png"

st.set_page_config(layout="wide", page_icon=logo_path, page_title="Tesla Used Cars Dashboard")
# st.set_page_config(layout="wide", page_icon=":bar_chart:", page_title="Tesla Used Cars Dashboard")
st.title(':bar_chart: Tesla Used Cars Dashboard')
# Add my name to the top right corner
st.markdown('<p style="text-align: right;">By Aravind Raj Palepu</p>', unsafe_allow_html=True)
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Load data
os.chdir(r'c:\Users\Aravind\OneDrive\Documents\sem3\pyProjects\streamlitApp')  
df = pd.read_csv('ms_mx_my_m3_tsla_df.csv')

# 3 columns layout
col1, col2, col3 = st.columns(3)

# Set title for sidebar
st.sidebar.title('Filters')

# Create sidebar for model
model = st.sidebar.multiselect('Model', df['model'].unique())
# Create sidebar for year
year = st.sidebar.multiselect('Model Year', df['year'].unique())
# Create sidebar for drivetrain
drivetrain = st.sidebar.multiselect('Drivetrain', df['dtrain'].unique())
# Create sidebar for Accident History
accident = st.sidebar.multiselect('Accident History', df['accident_history'].unique())
# Create sidebar for paintjob
paintjob = st.sidebar.multiselect('PaintJob', df['paintJob'].unique())

# Filter data
df_filtered = df.copy()
if model:
    df_filtered = df_filtered[df_filtered['model'].isin(model)]
if year:
    df_filtered = df_filtered[df_filtered['year'].isin(year)]
if accident:
    df_filtered = df_filtered[df_filtered['accident_history'].isin(accident)]
if drivetrain:
    df_filtered = df_filtered[df_filtered['dtrain'].isin(drivetrain)]
if paintjob:
    df_filtered = df_filtered[df_filtered['paintJob'].isin(paintjob)]
if model and year:
    df_filtered = df_filtered[df_filtered['model'].isin(model) & df_filtered['year'].isin(year)]
if model and accident:
    df_filtered = df_filtered[df_filtered['model'].isin(model) & df_filtered['accident_history'].isin(accident)]
if year and accident:
    df_filtered = df_filtered[df_filtered['year'].isin(year) & df_filtered['accident_history'].isin(accident)]
if year and drivetrain:
    df_filtered = df_filtered[df_filtered['year'].isin(year) & df_filtered['dtrain'].isin(drivetrain)]
if model and drivetrain:
    df_filtered = df_filtered[df_filtered['model'].isin(model) & df_filtered['dtrain'].isin(drivetrain)]
if accident and drivetrain:
    df_filtered = df_filtered[df_filtered['accident_history'].isin(accident) & df_filtered['dtrain'].isin(drivetrain)]
if paintjob and drivetrain:
    df_filtered = df_filtered[df_filtered['paintJob'].isin(paintjob) & df_filtered['dtrain'].isin(drivetrain)]
if model and year and drivetrain:
    df_filtered = df_filtered[df_filtered['model'].isin(model) & df_filtered['year'].isin(year) & df_filtered['dtrain'].isin(drivetrain)]
if model and year and accident and drivetrain:
    df_filtered = df_filtered[df_filtered['model'].isin(model) & df_filtered['year'].isin(year) & df_filtered['accident_history'].isin(accident)]
if model and year and accident and drivetrain and paintjob:
    df_filtered = df_filtered[df_filtered['model'].isin(model) & df_filtered['year'].isin(year) & df_filtered['accident_history'].isin(accident) & df_filtered['dtrain'].isin(drivetrain) & df_filtered['paintJob'].isin(paintjob)]


# Fid 1 - Avg price per model
with col1:
    st.markdown('<center><h3>Average Price per Model</h3></center>', unsafe_allow_html=True)
    price_df = df_filtered.groupby(by= ['model'], as_index=False)['price'].mean()
    normalized_price = (price_df['price'] - price_df['price'].min()) / (price_df['price'].max() - price_df['price'].min())

    fig = px.bar(price_df, x='price', y='model',
                text= ['${:,.2f}'.format(price) for price in price_df['price']],
                template='seaborn', 
                hover_data={'price': ':,.2f'}, 
                color=normalized_price,
                color_continuous_scale='reds',
                orientation='h')
    fig.update_traces(hovertemplate="Model: %{y}<br>Price: $%{x:,.2f}<extra></extra>")
    fig.update_xaxes(tickprefix="$", showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, height=200)


# Fig 2 - Model wise accidents count
with col2:
    st.markdown('<center><h3>Accident Count by Model</h3></center>', unsafe_allow_html=True)
    accident_df = df_filtered.groupby(by= ['model'], as_index=False)['accident_history'].count()
    custom_reds = ['#800000','#FF9999', '#F5F5DC','#FF3333']
    fig = px.pie(accident_df, values='accident_history', hole= 0.5, names='model', template='seaborn', color_discrete_sequence=custom_reds)
    fig.update_traces(textposition='outside', text= accident_df['model'])
    
    # Add text "Accident Count" inside the hollow of the donut chart
    fig.add_annotation(text='Accident Count', x=0.5, y=0.5, showarrow=False, font_size=20, font_color='white', textangle=0)
    
    st.plotly_chart(fig, use_container_width=True)


# Fig 3 - Average Price of models by DAS
with col3:
    st.markdown('<center><h3>Average Price of models by DAS</h3></center>', unsafe_allow_html=True)
    # Group by Drive Assistance Mode and calculate the count and average price
    das_grouped = df_filtered.groupby('DAS').agg(count=('DAS', 'count'), price=('price', 'mean')).reset_index()

    # Define custom red color palette
    custom_reds = ['#800000','#FF9999', '#F5F5DC','#FF3333']

    # Create a bar plot using custom red color palette
    fig3 = px.bar(das_grouped, x='DAS', y='price', 
                color='count',
                color_continuous_scale=custom_reds,
                hover_data={'DAS': True, 'count': True, 'price': True})
    
    fig3.update_yaxes(title_text='Avg Price', showgrid=False)
    fig3.update_xaxes(tickangle=45, showgrid=False)
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)

# Fig 4 - Average Price of Models in Each State
colm1, colm2 = st.columns(2)
with colm1:
    states = ['CA', 'AZ', 'IL', 'FL', 'MD', 'WA', 'NJ', 'GA', 'TN', 'MN', 'VA',
            'MA', 'NC', 'NY', 'CO', 'OH', 'OR', 'PA']

    # Calculate the average price for each state and model
    avg_price_per_state_model = df_filtered[df_filtered['state'].isin(states)].groupby(['state', 'model'])['price'].mean().reset_index()
    # Define the custom reds color palette
    custom_reds = ['#800000', '#B22222', '#F5F5DC','#FF3333', '#FFA07A', '#FFB6C1']

    st.markdown('<center><h3>Average Price of Models by State</h3></center>', unsafe_allow_html=True)
    # Create a choropleth map with custom reds color palette
    fig4 = px.choropleth(avg_price_per_state_model, 
                        locations='state', 
                        locationmode='USA-states',
                        color='price',
                        hover_name='model', 
                        scope='usa',
                        color_continuous_scale=custom_reds)

    # Update the layout of the plot
    fig4.update_layout(
        width=1000, 
        height=600, 
        paper_bgcolor='black', 
        plot_bgcolor='black',
        title=dict(text='View in Full Screen for a Better View', x=0.5, y=0.05, xanchor='center', yanchor="bottom", font=dict(color='white')),
        geo=dict(bgcolor='black')
    )
    # Update tooltip content
    fig4.update_traces(
        hovertemplate='<b>State</b>: %{location}<br><b>Average Price</b>: $%{z:.2f}')
    st.plotly_chart(fig4, use_container_width=True)

# Fig 5 - Average Price of Models by Drivetrain
with colm2:
    custom_reds = ['#800000', '#B22222', '#FF6347']
    st.markdown('<center><h3>Average Price of Models by Drivetrain</h3></center>', unsafe_allow_html=True)
    df_grouped = df_filtered.groupby(['dtrain', 'model']).agg(price=('price', 'mean'), count=('model', 'count')).reset_index()

    # Create traces for each drivetrain category
    traces = []
    for i, dtrain in enumerate(df_grouped['dtrain'].unique()):
        df_subset = df_grouped[df_grouped['dtrain'] == dtrain]
        trace = go.Barpolar(r=df_subset['price'], theta=df_subset['model'], name=dtrain, 
                            marker_color=custom_reds[i % len(custom_reds)],
                            customdata=df_subset[['model', 'count']])
        traces.append(trace)

    # Create a radial bar chart
    fig5 = go.Figure(data=traces)
    fig5.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor='rgba(0,0,0,0)', tickcolor='black'),
            angularaxis=dict(gridcolor='rgba(0,0,0,0)'),
            bgcolor='black'
        ),
        showlegend=True,
        height=600, 
        width=500, 
        paper_bgcolor='Black', 
        plot_bgcolor='Black',
        legend=dict(x=0, y=0), 
    )

    # Update tooltip content
    fig5.update_traces(
    hovertemplate=(
        '<b style="color: White; font-weight: bold;">Model</b>: %{customdata[0]}<br>'
        '<b style="color: White; font-weight: bold;">Number of Listings</b>: %{customdata[1]:,}<br>'
        '<b style="color: White; font-weight: bold;">Average Price</b>: $%{r:,.2f}'
    )
)

    st.plotly_chart(fig5, use_container_width=True)
