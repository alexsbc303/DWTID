import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import nltk
from textblob import TextBlob
import re
from datetime import datetime, timedelta
import warnings
import json
import chardet
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Dark Web Threat Intelligence Research Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_threat_keywords() -> dict:
    """Load threat keywords from a JSON file"""
    try:
        with open('threat_keywords.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading threat keywords: {e}")
        return {}

def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding']
    except Exception as e:
        st.warning(f"Could not detect encoding for {file_path}: {e}")
        return 'latin-1'  # Fallback to a robust encoding

@st.cache_data
def load_datasets():
    """Load and preprocess the datasets"""
    st.info("Attempting to load datasets by automatically detecting encoding...")
    dark_market_df = None
    forum_df = None
    
    try:
        market_encoding = detect_encoding('dark_market_output_v2.csv')
        st.info(f"Detected encoding for 'dark_market_output_v2.csv': {market_encoding}")
        try:
            dark_market_df = pd.read_csv('dark_market_output_v2.csv', encoding=market_encoding, on_bad_lines='skip', engine='python')
        except UnicodeDecodeError:
            st.warning(f"Decoding with {market_encoding} failed. Falling back to latin-1 for 'dark_market_output_v2.csv'.")
            dark_market_df = pd.read_csv('dark_market_output_v2.csv', encoding='latin-1', on_bad_lines='skip', engine='python')
    except Exception as e:
        st.error(f"[DEBUG] Error loading 'dark_market_output_v2.csv': {e}")
        
    try:
        forum_encoding = detect_encoding('Clearnedup_ALL_7.csv')
        st.info(f"Detected encoding for 'Clearnedup_ALL_7.csv': {forum_encoding}")
        try:
            forum_df = pd.read_csv('Clearnedup_ALL_7.csv', encoding=forum_encoding, on_bad_lines='skip', engine='python')
        except UnicodeDecodeError:
            st.warning(f"Decoding with {forum_encoding} failed. Falling back to latin-1 for 'Clearnedup_ALL_7.csv'.")
            forum_df = pd.read_csv('Clearnedup_ALL_7.csv', encoding='latin-1', on_bad_lines='skip', engine='python')
    except Exception as e:
        st.error(f"[DEBUG] Error loading 'Clearnedup_ALL_7.csv': {e}")

    if dark_market_df is None or forum_df is None:
        st.warning("One or both datasets failed to load.")
        return None, None
        
    return dark_market_df, forum_df

@st.cache_data
def preprocess_data(dark_market_df, forum_df):
    """Preprocess and clean the data"""
    # Clean dark market data
    if dark_market_df is not None:
        dark_market_df['Description'] = dark_market_df['Description'].fillna('')
        dark_market_df['Price_USD'] = pd.to_numeric(
            dark_market_df['Price'].str.extract(r'(\d+\.?\d*)')[0], 
            errors='coerce'
        )
    
    # Clean forum data
    if forum_df is not None:
        forum_df['Post Content'] = forum_df['Post Content'].fillna('')
        forum_df['datetime'] = pd.to_datetime(forum_df['datetime'], errors='coerce')
    
    return dark_market_df, forum_df

def detect_threat_keywords(text: str, threat_keywords: dict) -> list[str]:
    """Detect cybersecurity threat keywords"""
    detected_threats = []
    text_lower = str(text).lower()
    
    for category, keywords in threat_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_threats.append(category)
                break
    
    return detected_threats

def create_network_graph(df, column1, column2, limit=20):
    """Create network graph showing relationships"""
    G = nx.Graph()
    
    # Get top connections
    connections = df.groupby([column1, column2]).size().reset_index(name='weight')
    connections = connections.sort_values('weight', ascending=False).head(limit)
    
    for _, row in connections.iterrows():
        G.add_edge(row[column1], row[column2], weight=row['weight'])
    
    return G

def sentiment_analysis(text: str) -> str:
    """Perform sentiment analysis on text"""
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    except Exception:
        return 'neutral'

def main():
    st.title("🔒 Dark Web Threat Intelligence Research Dashboard")
    st.markdown("### Academic Research Tool for Cybersecurity Threat Analysis")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Academic Research Disclaimer**: This tool is designed for educational and research purposes only. 
    It analyzes existing datasets to understand cybersecurity threats and dark web activities for defensive research.
    """)
    
    # Load data
    dark_market_df, forum_df = load_datasets()
    threat_keywords = load_threat_keywords()

    if dark_market_df is None or forum_df is None or not threat_keywords:
        st.error("Unable to load datasets or keywords. Please ensure CSV and JSON files are in the correct location.")
        return

    # Preprocess data
    dark_market_df, forum_df = preprocess_data(dark_market_df, forum_df)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Page",
        ["Overview", "Threat Detection", "Market Analysis", "Forum Analysis", "Network Analysis", "ML Classification"]
    )

    if page == "Overview":
        show_overview(dark_market_df, forum_df)
    elif page == "Threat Detection":
        show_threat_detection(dark_market_df, forum_df, threat_keywords)
    elif page == "Market Analysis":
        show_market_analysis(dark_market_df)
    elif page == "Forum Analysis":
        show_forum_analysis(forum_df)
    elif page == "Network Analysis":
        show_network_analysis(dark_market_df, forum_df)
    elif page == "ML Classification":
        show_ml_classification(dark_market_df, forum_df, threat_keywords)

def show_overview(dark_market_df, forum_df):
    """Show overview dashboard"""
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Market Listings", len(dark_market_df))
    
    with col2:
        st.metric("Forum Posts", len(forum_df))
    
    with col3:
        unique_sellers = dark_market_df['Seller'].nunique()
        st.metric("Unique Sellers", unique_sellers)
    
    with col4:
        unique_forums = forum_df['Forum Name'].nunique()
        st.metric("Forums Analyzed", unique_forums)
    
    # Data samples
    st.subheader("Market Data Sample")
    st.dataframe(dark_market_df[['Title', 'Price', 'Seller', 'Category']].head())
    
    st.subheader("Forum Data Sample")
    st.dataframe(forum_df[['Forum Name', 'Thread Title', 'Username', 'User Type']].head())

def show_threat_detection(dark_market_df, forum_df, threat_keywords):
    """Show threat detection analysis"""
    st.header("🚨 Threat Detection Engine")
    
    # Analyze market listings
    st.subheader("Market Threat Analysis")
    
    market_threats = []
    for _, row in dark_market_df.iterrows():
        text = f"{row['Title']} {row['Description']}"
        threats = detect_threat_keywords(text, threat_keywords)
        if threats:
            market_threats.append({
                'Title': row['Title'],
                'Seller': row['Seller'],
                'Category': row['Category'],
                'Threats': threats,
                'Price': row.get('Price_USD', 0)
            })
    
    threat_df = pd.DataFrame(market_threats)
    
    if len(threat_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Threat category distribution
            threat_counts = {}
            for threats in threat_df['Threats']:
                for threat in threats:
                    threat_counts[threat] = threat_counts.get(threat, 0) + 1
            
            threat_chart_df = pd.DataFrame(list(threat_counts.items()), columns=['Threat Type', 'Count'])
            fig = px.bar(threat_chart_df, x='Threat Type', y='Count', title='Detected Threat Categories')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top threat sellers
            seller_threats = threat_df.groupby('Seller').size().sort_values(ascending=False).head(10)
            fig = px.bar(x=seller_threats.values, y=seller_threats.index, orientation='h',
                        title='Top Sellers by Threat Activity')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed threat listings
        st.subheader("Detected Threat Listings")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            threat_filter = st.multiselect("Filter by Threat Type", options=list(threat_counts.keys()))
        with col2:
            seller_filter = st.selectbox("Filter by Seller", options=['All'] + list(threat_df['Seller'].unique()))
        
        filtered_df = threat_df.copy()
        if threat_filter:
            filtered_df = filtered_df[filtered_df['Threats'].apply(lambda x: any(t in threat_filter for t in x))]
        if seller_filter != 'All':
            filtered_df = filtered_df[filtered_df['Seller'] == seller_filter]
        
        st.dataframe(filtered_df)

def show_market_analysis(dark_market_df):
    """Show market analysis"""
    st.header("💰 Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        category_counts = dark_market_df['Category'].value_counts().head(10)
        fig = px.pie(values=category_counts.values, names=category_counts.index, 
                     title='Product Categories Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price analysis
        price_data = dark_market_df[dark_market_df['Price_USD'].notna() & (dark_market_df['Price_USD'] > 0)]
        if len(price_data) > 0:
            fig = px.histogram(price_data, x='Price_USD', title='Price Distribution', nbins=50)
            fig.update_xaxis(range=[0, price_data['Price_USD'].quantile(0.95)])
            st.plotly_chart(fig, use_container_width=True)
    
    # Top sellers analysis
    st.subheader("Seller Analysis")
    seller_stats = dark_market_df.groupby('Seller').agg({
        'Title': 'count',
        'Price_USD': 'mean'
    }).sort_values('Title', ascending=False).head(20)
    seller_stats.columns = ['Listings Count', 'Average Price']
    
    fig = px.scatter(seller_stats, x='Listings Count', y='Average Price', 
                     title='Seller Activity vs Average Price',
                     hover_data={'Listings Count': True, 'Average Price': True})
    st.plotly_chart(fig, use_container_width=True)

def show_forum_analysis(forum_df):
    """Show forum analysis"""
    st.header("💬 Forum Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Forum activity
        forum_activity = forum_df['Forum Name'].value_counts().head(10)
        fig = px.bar(x=forum_activity.index, y=forum_activity.values, 
                     title='Forum Activity (Posts Count)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # User types
        user_types = forum_df['User Type'].value_counts()
        fig = px.pie(values=user_types.values, names=user_types.index, 
                     title='User Types Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    if 'datetime' in forum_df.columns:
        st.subheader("Temporal Analysis")
        
        # Filter valid dates
        valid_dates = forum_df[forum_df['datetime'].notna()].copy()
        if len(valid_dates) > 0:
            valid_dates['date'] = valid_dates['datetime'].dt.date
            daily_posts = valid_dates.groupby('date').size().reset_index(name='posts')
            
            fig = px.line(daily_posts, x='date', y='posts', title='Forum Activity Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    # Content analysis
    st.subheader("Content Analysis")
    
    # Word cloud of post content
    if st.button("Generate Word Cloud"):
        st.warning("Generating word cloud from a sample of 20,000 posts for performance reasons.")
        sample_df = forum_df.sample(min(20000, len(forum_df)))
        text_content = ' '.join(sample_df['Post Content'].dropna().astype(str))
        
        if len(text_content) > 0:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_content)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

def show_network_analysis(dark_market_df, forum_df):
    """Show network analysis"""
    st.header("🕸️ Network Analysis")
    
    st.subheader("Seller-Category Network")
    
    # Create network graph for sellers and categories
    G = create_network_graph(dark_market_df, 'Seller', 'Category', 30)
    
    if len(G.nodes()) > 0:
        # Calculate network metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Network Nodes", len(G.nodes()))
        with col2:
            st.metric("Network Edges", len(G.edges()))
        with col3:
            density = nx.density(G)
            st.metric("Network Density", f"{density:.3f}")
        
        # Visualize network
        pos = nx.spring_layout(G, k=0.9, iterations=50)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_size = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node} (Degree: {G.degree(node)})")
            node_size.append(G.degree(node) * 5 + 10)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
        
        node_trace.marker.color = node_adjacencies

        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Seller-Category Network',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Network graph showing connections between Sellers and Product Categories",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        
        st.plotly_chart(fig, use_container_width=True)

def prepare_classification_data(df: pd.DataFrame, text_column: str, threat_keywords: dict) -> tuple[list, list]:
    """Prepare text data and labels for classification from a dataframe."""
    text_data = []
    labels = []
    for _, row in df.iterrows():
        text = str(row[text_column])
        threats = detect_threat_keywords(text, threat_keywords)
        text_data.append(text)
        if threats:
            labels.append('threat')
        else:
            labels.append('normal')
    return text_data, labels


def show_ml_classification(dark_market_df: pd.DataFrame, forum_df: pd.DataFrame, threat_keywords: dict) -> None:
    """Show ML classification analysis"""
    st.header("🤖 Machine Learning Classification")
    
    st.subheader("Post Classification Model")
    
    # Prepare data for classification
    if st.button("Train Classification Model"):
        with st.spinner("Training model..."):
            # Prepare data
            dark_market_df['combined_text'] = dark_market_df['Title'] + ' ' + dark_market_df['Description'].fillna('')
            market_text, market_labels = prepare_classification_data(dark_market_df, 'combined_text', threat_keywords)
            
            forum_sample = forum_df.sample(min(1000, len(forum_df)))
            forum_text, forum_labels = prepare_classification_data(forum_sample, 'Post Content', threat_keywords)

            text_data = market_text + forum_text
            labels = market_labels + forum_labels
            
            if len(text_data) > 10:
                # Vectorize text
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                X = vectorizer.fit_transform(text_data)
                y = np.array(labels)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                model = MultinomialNB()
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    accuracy = (y_pred == y_test).mean()
                    st.metric("Model Accuracy", f"{accuracy:.3f}")
                    
                    # Classification report
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                
                with col2:
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, text_auto=True, aspect="auto", 
                                   title="Confusion Matrix",
                                   labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.subheader("Top Predictive Features")
                feature_names = vectorizer.get_feature_names_out()
                feature_importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
                top_features = np.argsort(feature_importance)[-20:]
                
                feature_df = pd.DataFrame({
                    'Feature': [feature_names[i] for i in top_features],
                    'Importance': [feature_importance[i] for i in top_features]
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                           title='Top 20 Predictive Features')
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()