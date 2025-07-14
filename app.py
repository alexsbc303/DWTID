import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import nltk
import re
from datetime import datetime
import warnings
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gc
import pickle
from functools import lru_cache
try:
    from gensim import corpora
    from gensim.models import LdaModel
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configuration
MAX_SAMPLE_SIZE = 25000
CACHE_TTL = 3600  # 1 hour

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

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_datasets_optimized():
    """Load datasets with memory optimization and clean UI."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"🔄 Loading datasets...")
    
    try:
        # Load market data
        progress_bar.progress(0.2)
        status_text.text("📊 Loading market data...")
        crawled_result_df = pd.read_csv('dark_market_output_v2.csv', 
                                      encoding='latin-1', 
                                      on_bad_lines='skip', 
                                      engine='c',
                                      low_memory=False)
        
        # Load forum data
        progress_bar.progress(0.6)
        status_text.text("💬 Loading all forum data (this may take a moment)...")
        preprocessed_forum_df = pd.read_csv('Clearnedup_ALL_7.csv', 
                                          encoding='latin-1', 
                                          on_bad_lines='skip', 
                                          engine='c',
                                          low_memory=False)
        
        gc.collect()
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Datasets loaded! Market: {len(crawled_result_df):,}, Forum: {len(preprocessed_forum_df):,}")
        
        return crawled_result_df, preprocessed_forum_df
        
    except Exception as e:
        st.error(f"❌ Error loading datasets: {e}")
        return None, None
    finally:
        progress_bar.empty()
        status_text.empty()

@st.cache_data
def load_or_process_data(threat_keywords):
    """Load preprocessed data if it exists, otherwise process and save it."""
    crawled_pkl_path = 'crawled_processed.pkl'
    forum_pkl_path = 'forum_processed.pkl'

    try:
        with open(crawled_pkl_path, 'rb') as f:
            crawled_result_df = pickle.load(f)
        with open(forum_pkl_path, 'rb') as f:
            preprocessed_forum_df = pickle.load(f)
        st.success("Loaded pre-processed data from cache.")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        st.info("📂 No cache found, processing fresh data...")
        crawled_result_df, preprocessed_forum_df = load_datasets_optimized()
        
        if crawled_result_df is None or preprocessed_forum_df is None:
            return None, None

        # Always run preprocessing
        crawled_result_df, preprocessed_forum_df = preprocess_data_full(crawled_result_df, preprocessed_forum_df, threat_keywords)
        
        with open(crawled_pkl_path, 'wb') as f:
            pickle.dump(crawled_result_df, f)
        with open(forum_pkl_path, 'wb') as f:
            pickle.dump(preprocessed_forum_df, f)
            
        st.success("Data processed and cached successfully.")

    # Ensure threat columns are always present
    if crawled_result_df is not None and preprocessed_forum_df is not None:
        crawled_result_df, preprocessed_forum_df = preprocess_data_full(crawled_result_df, preprocessed_forum_df, threat_keywords)

    return crawled_result_df, preprocessed_forum_df

@st.cache_data(ttl=CACHE_TTL)
def preprocess_data_full(crawled_result_df, preprocessed_forum_df, threat_keywords):
    """The full preprocessing pipeline - optimized version with clean UI."""
    
    # Pre-compile regexes for faster threat detection with optimization
    @lru_cache(maxsize=128)
    def compile_regex_pattern(pattern):
        """Cache compiled regex patterns for reuse."""
        return re.compile(pattern, re.IGNORECASE)
    
    compiled_regexes = {}
    for category, keywords in threat_keywords.items():
        # Use word boundaries and optimize pattern
        pattern = r'\b(?:' + '|'.join(re.escape(keyword) for keyword in keywords) + r')\b'
        compiled_regexes[category] = compile_regex_pattern(pattern)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # --- Preprocess Crawled Market Data ---
        if crawled_result_df is not None:
            status_text.text(f"🏪 Processing {len(crawled_result_df):,} market records...")
            progress_bar.progress(0.1)
            crawled_result_df = process_market_data(crawled_result_df, compiled_regexes)
            progress_bar.progress(0.4)
        
        # --- Preprocess Forum Data ---
        if preprocessed_forum_df is not None:
            status_text.text(f"💬 Processing {len(preprocessed_forum_df):,} forum records...")
            progress_bar.progress(0.5)
            preprocessed_forum_df = process_forum_data(preprocessed_forum_df, compiled_regexes)
            progress_bar.progress(0.9)
        
        # Final optimization
        gc.collect()
        progress_bar.progress(1.0)
        status_text.text(f"✅ All data processing complete!")
        
        return crawled_result_df, preprocessed_forum_df
        
    except Exception as e:
        st.error(f"❌ Data processing failed: {e}")
        return crawled_result_df, preprocessed_forum_df
    finally:
        progress_bar.empty()
        status_text.empty()

@st.cache_data
def process_market_data(df, compiled_regexes):
    """Process market data efficiently."""
    df = df.copy()
    df['Description'] = df['Description'].fillna('')
    df['Title'] = df['Title'].fillna('')
    df['Price_USD'] = pd.to_numeric(
        df['Price'].str.extract(r'(\d+\.?\d*)')[0], 
        errors='coerce'
    )
    df['combined_text'] = (df['Title'].astype(str) + ' ' + df['Description'].astype(str)).str.lower()
    
    # Vectorized threat detection
    for category, regex in compiled_regexes.items():
        df[category] = df['combined_text'].str.contains(regex, na=False)
    
    # Sample-based sentiment analysis for performance
    df['sentiment'] = compute_sentiment_efficient(df['combined_text'])
    
    return df

@st.cache_data
def process_forum_data(df, compiled_regexes):
    """Process forum data efficiently."""
    df = df.copy()
    df['Post Content'] = df['Post Content'].fillna('')
    df['Thread Title'] = df['Thread Title'].fillna('')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['combined_text'] = (df['Thread Title'].astype(str) + ' ' + df['Post Content'].astype(str)).str.lower()

    # Vectorized threat detection
    for category, regex in compiled_regexes.items():
        df[category] = df['combined_text'].str.contains(regex, na=False)
    
    # Efficient sentiment analysis
    df['sentiment'] = compute_sentiment_efficient(df['combined_text'])
    
    return df



@st.cache_data(ttl=CACHE_TTL)
def compute_sentiment_efficient(text_series, sample_size=None):
    """Compute sentiment efficiently using optimized sampling and vectorization."""
    if sample_size is None:
        sample_size = MAX_SAMPLE_SIZE
        
    analyzer = SentimentIntensityAnalyzer()
    
    # Quick exit for empty series
    if len(text_series) == 0:
        return pd.Series(dtype=float)
    
    # Show progress only for very large datasets
    show_progress = len(text_series) > 20000
    progress_bar = st.progress(0) if show_progress else None
    status_text = st.empty() if show_progress else None
    
    try:
        if show_progress:
            status_text.text("🔄 Analyzing sentiment...")
            progress_bar.progress(0.1)
        
        unique_texts = text_series.drop_duplicates()
        
        # Sample if dataset is too large
        effective_sample_size = min(sample_size, len(unique_texts))
        
        if len(unique_texts) > effective_sample_size:
            if show_progress:
                status_text.text(f"📊 Sampling {effective_sample_size:,} texts...")
                progress_bar.progress(0.2)
            sampled_texts = unique_texts.sample(n=effective_sample_size, random_state=42)
        else:
            sampled_texts = unique_texts
        
        # Compute sentiment for unique texts
        sentiment_map = {}
        for i, text in enumerate(sampled_texts):
            text_str = str(text).strip()
            if len(text_str) > 0:
                try:
                    sentiment_map[text] = analyzer.polarity_scores(text_str)['compound']
                except Exception:
                    sentiment_map[text] = 0.0
            else:
                sentiment_map[text] = 0.0
            
            if show_progress and i % 1000 == 0:
                progress = 0.2 + (i / len(sampled_texts)) * 0.7
                progress_bar.progress(progress)
        
        if show_progress:
            status_text.text("🔄 Mapping sentiment...")
            progress_bar.progress(0.9)
        
        # Map back to original series
        result = text_series.map(sentiment_map).fillna(0.0)
        
        if show_progress:
            status_text.text(f"✅ Sentiment analysis complete!")
            progress_bar.progress(1.0)
        
        return result
        
    except Exception as e:
        st.error(f"❌ Sentiment analysis failed: {e}")
        return pd.Series([0.0] * len(text_series), index=text_series.index)
    finally:
        if show_progress:
            progress_bar.empty()
            status_text.empty()
        gc.collect()

def get_threat_categories(threat_keywords: dict) -> list:
    return list(threat_keywords.keys())

def create_network_graph(df, column1, column2, limit=20):
    """Create network graph showing relationships"""
    G = nx.Graph()
    
    connections = df.groupby([column1, column2]).size().reset_index(name='weight')
    connections = connections.sort_values('weight', ascending=False).head(limit)
    
    for _, row in connections.iterrows():
        G.add_edge(row[column1], row[column2], weight=row['weight'])
    
    return G

def main():
    st.title("🔒 Dark Web Threat Intelligence Dashboard")
    st.markdown("### MSP24021")
    
    # Load data with clean UI
    threat_keywords = load_threat_keywords()
    
    if not threat_keywords:
        st.error("❌ Cannot load threat keywords. Please check threat_keywords.json file.")
        return
        
    crawled_result_df, preprocessed_forum_df = load_or_process_data(threat_keywords)

    if crawled_result_df is None or preprocessed_forum_df is None or not threat_keywords:
        st.error("Unable to load or process datasets. Please ensure CSV files are in the correct location.")
        return

    threat_categories = get_threat_categories(threat_keywords)

    st.sidebar.title("Navigation")
    analysis_choice = st.sidebar.radio(
        "Select Analysis View",
        ('Threat Dashboard', 'Crawled Result Analysis', 'Preprocessed Forum Datasets Analysis')
    )

    if analysis_choice == 'Threat Dashboard':
        show_threat_dashboard(crawled_result_df, preprocessed_forum_df, threat_keywords)
    
    elif analysis_choice == 'Crawled Result Analysis':
        st.header("Analysis of Crawled Market Data")
        show_overview(crawled_result_df, "market")
        show_threat_analysis(crawled_result_df, threat_categories)
        show_crawled_result_analysis(crawled_result_df)
        
        # Enhanced Market Listing Topics Analysis
        st.header("📊 Market Listing Topics Analysis")
        
        # Always show meaningful keyword analysis first
        show_meaningful_keyword_analysis(crawled_result_df, "Market Keywords & Threat Intelligence")
        
        # Then show topic modeling if available
        if GENSIM_AVAILABLE:
            st.markdown("---")
            show_topic_modeling(crawled_result_df, text_column='combined_text', title="Advanced Topic Modeling (LDA)")
        
        show_network_analysis(crawled_result_df, "Seller", "Category")
        show_ml_classification(crawled_result_df, threat_categories)
    
    elif analysis_choice == 'Preprocessed Forum Datasets Analysis':
        st.header("Analysis of Preprocessed Forum Data")
        show_overview(preprocessed_forum_df, "forum")
        show_threat_analysis(preprocessed_forum_df, threat_categories)
        show_forum_analysis(preprocessed_forum_df)
        
        # Enhanced Forum Discussion Topics Analysis
        st.header("💬 Forum Discussion Topics Analysis")
        
        # Always show meaningful keyword analysis first
        show_meaningful_keyword_analysis(preprocessed_forum_df, "Forum Keywords & Discussion Intelligence")
        
        # Then show topic modeling if available
        if GENSIM_AVAILABLE:
            st.markdown("---")
            show_topic_modeling(preprocessed_forum_df, text_column='Post Content', title="Advanced Topic Modeling (LDA)")
        
        show_network_analysis(preprocessed_forum_df, "Username", "Forum Name")
        show_ml_classification(preprocessed_forum_df, threat_categories)

def show_overview(df, dataset_type):
    """Show overview dashboard"""
    st.header("📊 Dataset Overview")
    
    if dataset_type == "market":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Market Listings", len(df))
        with col2:
            st.metric("Unique Sellers", df['Seller'].nunique())
        with col3:
            st.metric("Unique Categories", df['Category'].nunique())
        st.subheader("Market Data Sample")
        st.dataframe(df[['Title', 'Price', 'Seller', 'Category']].head())
    elif dataset_type == "forum":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Forum Posts", len(df))
        with col2:
            st.metric("Unique Users", df['Username'].nunique())
        with col3:
            st.metric("Forums Analyzed", df['Forum Name'].nunique())
        st.subheader("Forum Data Sample")
        st.dataframe(df[['Forum Name', 'Thread Title', 'Username', 'User Type']].head())

def show_threat_analysis(df, threat_categories):
    """Show threat detection analysis"""
    st.header("🚨 Threat Analysis")
    
    threat_df = df[df[threat_categories].any(axis=1)]
    
    if not threat_df.empty:
        # Threat distribution chart
        threat_counts = threat_df[threat_categories].sum().sort_values(ascending=False)
        fig = px.bar(threat_counts, x=threat_counts.index, y=threat_counts.values, title='Detected Threat Categories')
        st.plotly_chart(fig, use_container_width=True)

        if 'datetime' in threat_df.columns:
            st.subheader("Threat Mentions Over Time")
            threat_temporal = threat_df.set_index('datetime').resample('M')[threat_categories].sum()
            fig = px.line(threat_temporal, x=threat_temporal.index, y=threat_temporal.columns, title='Threat Mentions per Month')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔍 Advanced Threat Filtering")
        
        # Enhanced filtering options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_threats = st.multiselect(
                "Filter by Threat Type", 
                options=threat_categories,
                help="Select one or more threat types to filter the results"
            )
        
        with col2:
            if 'sentiment' in threat_df.columns:
                sentiment_filter = st.selectbox(
                    "Sentiment Filter",
                    options=["All", "Positive (>0.7)", "Neutral (0.3-0.7)", "Negative (<0.3)"],
                    help="Filter by sentiment analysis results"
                )
            else:
                sentiment_filter = "All"
        
        with col3:
            if 'Price_USD' in threat_df.columns:
                price_range = st.slider(
                    "Price Range (USD)",
                    min_value=0,
                    max_value=int(threat_df['Price_USD'].max()) if threat_df['Price_USD'].max() > 0 else 1000,
                    value=(0, int(threat_df['Price_USD'].max()) if threat_df['Price_USD'].max() > 0 else 1000),
                    help="Filter by price range"
                )
            else:
                price_range = None
        
        # Apply filters
        filtered_df = threat_df.copy()
        
        if selected_threats:
            threat_mask = filtered_df[selected_threats].any(axis=1)
            filtered_df = filtered_df[threat_mask]
        
        if sentiment_filter != "All" and 'sentiment' in filtered_df.columns:
            if sentiment_filter == "Positive (>0.7)":
                filtered_df = filtered_df[filtered_df['sentiment'] > 0.7]
            elif sentiment_filter == "Negative (<0.3)":
                filtered_df = filtered_df[filtered_df['sentiment'] < 0.3]
            elif sentiment_filter == "Neutral (0.3-0.7)":
                filtered_df = filtered_df[(filtered_df['sentiment'] >= 0.3) & (filtered_df['sentiment'] <= 0.7)]
        
        if price_range and 'Price_USD' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['Price_USD'] >= price_range[0]) & 
                (filtered_df['Price_USD'] <= price_range[1])
            ]
        
        st.subheader(f"Filtered to {len(filtered_df):,} results")
        
        if not filtered_df.empty:
            # Display options
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Filter out Unnamed columns
                available_columns = [col for col in filtered_df.columns if not col.startswith('Unnamed')]
                
                # Set default columns based on analysis type
                if 'Forum Name' in available_columns:
                    # Forum analysis defaults
                    default_columns = ['Forum Name', 'Thread Title', 'Username', 'User Type', 'Post Content', 'datetime', 'sentiment']
                else:
                    # Market analysis defaults
                    default_columns = ['Title', 'Description', 'Price', 'Seller', 'Category', 'sentiment', 'Seller Location', 'Availability', 'Quantity in Stock', 'Price_USD']
                
                display_columns = st.multiselect(
                    "Select columns to display",
                    options=available_columns,
                    default=[col for col in default_columns if col in available_columns],
                    help="Choose which columns to show in the table"
                )
            
            with col2:
                max_rows = st.number_input(
                    "Max rows to display",
                    min_value=10,
                    max_value=1000,
                    value=50,
                    step=10
                )
            
            if display_columns:
                st.dataframe(filtered_df[display_columns].head(max_rows))
            else:
                st.dataframe(filtered_df.head(max_rows))
            
            # Download option
            if st.button("📥 Download Filtered Results as CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"threat_analysis_{len(filtered_df)}_results.csv",
                    mime="text/csv"
                )
        else:
            st.info("No data matches the selected filters.")

        # Word cloud section with button
        st.subheader("📊 Threat Word Cloud")
        if st.button("🔄 Generate Word Cloud", help="Click to generate word cloud from filtered threat data"):
            text_content = ' '.join(filtered_df['combined_text'].dropna())
            if text_content and len(text_content.strip()) > 0:
                try:
                    with st.spinner("Generating word cloud..."):
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_content)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
                except ValueError as e:
                    st.warning(f"Could not generate word cloud: {e}")
            else:
                st.warning("No text content available for word cloud generation.")
        
    else:
        st.info("No threats detected in the data based on the current keywords.")

def show_crawled_result_analysis(crawled_result_df):
    """Show market analysis"""
    st.header("💰 Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_counts = crawled_result_df['Category'].value_counts().head(10)
        fig = px.pie(values=category_counts.values, names=category_counts.index, title='Product Categories Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Shows the distribution of different product categories in the marketplace, helping identify the most common types of offerings.")
    
    with col2:
        price_data = crawled_result_df[crawled_result_df['Price_USD'].notna() & (crawled_result_df['Price_USD'] > 0)]
        if len(price_data) > 0:
            fig = px.histogram(price_data, x='Price_USD', title='Price Distribution', nbins=50)
            fig.update_xaxes(range=[0, price_data['Price_USD'].quantile(0.95)])
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💵 Displays the price distribution across all listings, excluding extreme outliers to show typical pricing patterns.")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Seller Analysis")
        seller_stats = crawled_result_df.groupby('Seller').agg({'Title': 'count', 'Price_USD': 'mean'}).sort_values('Title', ascending=False).head(20)
        seller_stats.columns = ['Listings Count', 'Average Price']
        
        fig = px.scatter(seller_stats, x='Listings Count', y='Average Price', title='Seller Activity vs Average Price', hover_data={'Listings Count': True, 'Average Price': True})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🏪 Analyzes seller behavior by comparing their activity level (number of listings) with their average pricing strategy.")

    with col4:
        st.subheader("Sentiment Analysis")
        sentiment_counts = crawled_result_df['sentiment'].apply(lambda p: 'positive' if p > 0.7 else ('negative' if p < 0.3 else 'neutral')).value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment of Market Listings')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("😊 Shows the overall emotional tone of marketplace listings, indicating how sellers present their products (positive, neutral, or negative language).")

def show_forum_analysis(forum_df):
    """Show forum analysis"""
    st.header("💬 Forum Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Forum Activity")
        forum_activity = forum_df['Forum Name'].value_counts().head(10)
        fig = px.bar(x=forum_activity.index, y=forum_activity.values, title='Forum Activity (Posts Count)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📊 Shows which forums have the most discussion activity. Higher post counts indicate more active communities or trending topics.")
    
    with col2:
        st.subheader("User Types Distribution")
        user_types = forum_df['User Type'].value_counts()
        fig = px.pie(values=user_types.values, names=user_types.index, title='User Types Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("👥 Breakdown of user categories in forum discussions. Different user types may indicate varying levels of expertise or community roles.")
    
    # Second row with Temporal Analysis and Sentiment Analysis in 2 columns
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📈 Temporal Analysis")
        if 'datetime' in forum_df.columns:
            valid_dates = forum_df[forum_df['datetime'].notna()].copy()
            if len(valid_dates) > 0:
                valid_dates['date'] = valid_dates['datetime'].dt.date
                daily_posts = valid_dates.groupby('date').size().reset_index(name='posts')
                fig = px.line(daily_posts, x='date', y='posts', title='Forum Activity Over Time')
                st.plotly_chart(fig, use_container_width=True)
                st.caption("⏰ Timeline of forum posting activity. Spikes may indicate significant events, trending topics, or coordinated activities in the dark web community.")
            else:
                st.info("No valid datetime data available for temporal analysis.")
        else:
            st.info("Datetime column not available for temporal analysis.")
    
    with col4:
        st.subheader("😊 Sentiment Analysis")
        if 'sentiment' in forum_df.columns:
            sentiment_counts = forum_df['sentiment'].apply(lambda p: 'positive' if p > 0.1 else ('negative' if p < -0.1 else 'neutral')).value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment of Forum Posts')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("💭 Emotional tone analysis of forum discussions. Sentiment patterns can reveal community mood, satisfaction levels, or reaction to events.")
        else:
            st.info("Sentiment data not available for analysis.")

def show_network_analysis(df, source_col, target_col):
    """Show network analysis"""
    st.header("🕸️ Network Analysis")
    st.subheader(f"{source_col}-{target_col} Network")
    
    limit = st.slider("Number of connections to show", 5, 100, 30)
    G = create_network_graph(df, source_col, target_col, limit)
    
    if len(G.nodes()) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Network Nodes", len(G.nodes()))
        with col2:
            st.metric("Network Edges", len(G.edges()))
        with col3:
            density = nx.density(G)
            st.metric("Network Density", f"{density:.3f}")
        
        pos = nx.spring_layout(G, k=0.9, iterations=50)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        node_x, node_y, node_text, node_size = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node} (Degree: {G.degree(node)})")
            node_size.append(G.degree(node) * 5 + 10)

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=node_size, colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'), line_width=2))
        
        node_adjacencies = [len(adj[1]) for adj in G.adjacency()]
        node_trace.marker.color = node_adjacencies

        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=f'<br>{source_col}-{target_col} Network', titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), annotations=[dict(text=f"Network graph showing connections between {source_col} and {target_col}", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)], xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        st.plotly_chart(fig, use_container_width=True)

def prepare_classification_data(df: pd.DataFrame, threat_categories: list) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare data for classification."""
    df['is_threat'] = df[threat_categories].any(axis=1)
    return df['combined_text'], df['is_threat']

def show_ml_classification(df: pd.DataFrame, threat_categories: list) -> None:
    """Show ML classification analysis with automatic training and results."""
    st.header("🤖 Machine Learning Classification")
    st.subheader("Threat Classification Results")
    
    with st.spinner("Training and evaluating classification model..."):
        X, y = prepare_classification_data(df, threat_categories)
        
        if len(X) > 10 and y.sum() > 0:
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                
                model = MultinomialNB()
                model.fit(X_train_vec, y_train)
                
                y_pred = model.predict(X_test_vec)
                
                col1, col2 = st.columns(2)
                with col1:
                    accuracy = (y_pred == y_test).mean()
                    st.metric("Model Accuracy", f"{accuracy:.3f}")
                    
                    # Show class distribution
                    threat_count = y.sum()
                    total_count = len(y)
                    st.metric("Threat Posts", f"{threat_count:,} ({threat_count/total_count:.1%})")
                    st.metric("Non-Threat Posts", f"{total_count-threat_count:,} ({(total_count-threat_count)/total_count:.1%})")
                    
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                
                with col2:
                    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
                    fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"), x=model.classes_, y=model.classes_)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Top Predictive Features")
                try:
                    threat_idx = list(model.classes_).index(True)
                    feature_names = vectorizer.get_feature_names_out()
                    feature_importance = model.feature_log_prob_[threat_idx] - model.feature_log_prob_[1 - threat_idx]
                    top_features = np.argsort(feature_importance)[-20:]
                    
                    feature_df = pd.DataFrame({
                        'Feature': [feature_names[i] for i in top_features], 
                        'Importance': [feature_importance[i] for i in top_features]
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h', title='Top 20 Predictive Features for "Threat" Class')
                    st.plotly_chart(fig, use_container_width=True)
                except (ValueError, IndexError) as e:
                    st.warning(f"Could not calculate feature importance: {e}")
                    
            except Exception as e:
                st.error(f"Classification failed: {e}")
        else:
            st.warning("Insufficient data for classification. Need more samples with threat labels.")

def show_topic_modeling(df, text_column='combined_text', title="Topic Modeling", num_topics=5, num_words=10):
    """Enhanced topic modeling with better UI controls and visualization."""
    st.subheader(f"🔬 {title}")
    
    if not GENSIM_AVAILABLE:
        st.error("Topic modeling libraries are not available. Please install gensim and nltk.")
        show_meaningful_keyword_analysis(df, title.replace("Topics", "Keywords"))
        return
    
    # Enhanced UI Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_topics = st.slider("📊 Number of Topics", min_value=3, max_value=15, value=8)
    with col2:
        num_words = st.slider("🔤 Words per Topic", min_value=5, max_value=20, value=10)
    with col3:
        sample_size = st.selectbox("📄 Sample Size", 
                                  options=[1000, 5000, 10000, 25000, "All"],
                                  index=2)
    
    if st.button("🚀 Run Topic Analysis", type="primary"):
        with st.spinner("Running LDA analysis..."):
            try:
                # Download NLTK data if needed
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    nltk.download('stopwords', quiet=True)
                    stop_words = set(stopwords.words('english'))
                
                # Text preprocessing
                def preprocess_text(text):
                    try:
                        tokens = word_tokenize(str(text).lower())
                    except:
                        try:
                            nltk.download('punkt', quiet=True)
                            nltk.download('punkt_tab', quiet=True)
                            tokens = word_tokenize(str(text).lower())
                        except:
                            tokens = re.findall(r'\b\w+\b', str(text).lower())
                    return [word for word in tokens if word.isalpha() and len(word) > 2 and word not in stop_words]

                # Use combined_text if available
                if 'combined_text' in df.columns:
                    text_data = df['combined_text'].dropna()
                elif text_column in df.columns:
                    text_data = df[text_column].dropna()
                else:
                    st.error(f"Column '{text_column}' not found in the dataset.")
                    return
                
                # Sample data
                if sample_size != "All":
                    text_data = text_data.sample(n=min(sample_size, len(text_data)), random_state=42)
                
                # Process documents
                processed_docs = []
                for text in text_data:
                    doc = preprocess_text(text)
                    if len(doc) > 3:
                        processed_docs.append(doc)
                
                if len(processed_docs) < 10:
                    st.warning("Not enough meaningful text data for topic modeling.")
                    show_meaningful_keyword_analysis(df, title.replace("Topics", "Keywords"))
                    return
                    
                # Create dictionary and corpus
                dictionary = corpora.Dictionary(processed_docs)
                dictionary.filter_extremes(no_below=2, no_above=0.8)
                
                if len(dictionary) < 10:
                    st.warning("Insufficient vocabulary for topic modeling.")
                    show_meaningful_keyword_analysis(df, title.replace("Topics", "Keywords"))
                    return
                
                corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
                
                # Train LDA model
                lda_model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=42,
                    passes=10,
                    per_word_topics=True
                )
                
                st.success(f"✅ Successfully identified {num_topics} topics!")
                
                # Display topics in tabs
                topic_tabs = st.tabs([f"Topic {i+1}" for i in range(num_topics)])
                
                for i, tab in enumerate(topic_tabs):
                    with tab:
                        topic_words = lda_model.show_topic(i, num_words)
                        
                        st.markdown(f"### 🏷️ Topic {i+1}")
                        
                        # Display as metrics
                        words_cols = st.columns(min(5, len(topic_words)))
                        for j, (word, prob) in enumerate(topic_words):
                            col_idx = j % len(words_cols)
                            with words_cols[col_idx]:
                                st.metric(word.title(), f"{prob:.3f}")
                        
                        # Word cloud
                        try:
                            topic_dict = dict(topic_words)
                            wordcloud = WordCloud(
                                width=800, height=400,
                                background_color='white',
                                colormap='viridis'
                            ).generate_from_frequencies(topic_dict)
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'Topic {i+1} Word Cloud', fontsize=16)
                            st.pyplot(fig)
                            plt.close()
                        except ImportError:
                            # Fallback bar chart
                            words, probs = zip(*topic_words)
                            fig = px.bar(
                                x=list(probs), y=list(words),
                                orientation='h',
                                title=f'Topic {i+1} Word Probabilities'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Summary metrics
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📚 Vocabulary Size", f"{len(dictionary):,}")
                with col2:
                    st.metric("📄 Documents Processed", f"{len(processed_docs):,}")
                with col3:
                    avg_doc_length = sum(len(doc) for doc in processed_docs) / len(processed_docs)
                    st.metric("📏 Avg. Document Length", f"{avg_doc_length:.1f} words")
                    
            except Exception as e:
                st.error(f"Topic modeling failed: {e}")
                show_meaningful_keyword_analysis(df, title.replace("Topics", "Keywords"))

def show_meaningful_keyword_analysis(df, title="Keyword Analysis"):
    """Show meaningful keywords and adjectives for analysis when topic modeling is not available."""
    st.header(f"🔑 {title}")
    
    # Use combined_text if available, otherwise find appropriate text column
    text_data = None
    if 'combined_text' in df.columns:
        text_data = df['combined_text'].dropna()
    elif 'Post Content' in df.columns:
        text_data = df['Post Content'].dropna()
    elif 'Description' in df.columns:
        text_data = df['Description'].dropna()
    elif 'Title' in df.columns:
        text_data = df['Title'].dropna()
    else:
        st.warning("No suitable text column found for analysis.")
        return
    
    # Sample data for performance - use 50K for forum data, 10K for others
    if 'Post Content' in df.columns or 'Forum' in title:
        # Forum data - use larger sample
        sample_size = min(50000, len(text_data))
    else:
        # Market data - use standard sample
        sample_size = min(10000, len(text_data))
    
    text_sample = text_data.sample(n=sample_size, random_state=42) if len(text_data) > sample_size else text_data
    
    # Combine all text
    all_text = ' '.join(text_sample.astype(str)).lower()
    
    try:
        import re
        
        # Define meaningful keyword categories for dark web analysis
        threat_keywords = {
            'Malware Types': ['ransomware', 'trojan', 'virus', 'worm', 'rootkit', 'spyware', 'adware', 'keylogger', 'stealer', 'rat', 'backdoor', 'botnet'],
            'Attack Methods': ['exploit', 'vulnerability', 'zero-day', 'phishing', 'social engineering', 'brute force', 'ddos', 'injection', 'bypass', 'penetration'],
            'Tools & Software': ['crypter', 'loader', 'builder', 'scanner', 'cracker', 'generator', 'checker', 'validator', 'parser', 'scraper', 'grabber'],
            'Target Systems': ['windows', 'android', 'ios', 'linux', 'mac', 'server', 'database', 'network', 'mobile', 'web application'],
            'Criminal Services': ['tutorial', 'guide', 'method', 'service', 'seller', 'vendor', 'escrow', 'marketplace', 'forum', 'private'],
            'Security Evasion': ['undetectable', 'fud', 'private', 'exclusive', 'stealth', 'hidden', 'encrypted', 'obfuscated', 'packed', 'armored'],
            'Payment Terms': ['bitcoin', 'crypto', 'payment', 'price', 'cost', 'money', 'cash', 'transfer', 'wallet', 'exchange'],
            'Quality Indicators': ['tested', 'working', 'updated', 'fresh', 'new', 'premium', 'professional', 'reliable', 'guaranteed', 'verified']
        }
        
        # Quality adjectives that indicate sophistication
        quality_adjectives = ['premium', 'professional', 'advanced', 'sophisticated', 'exclusive', 'private', 'custom', 'enterprise', 'military', 'commercial', 'industrial']
        
        # Threat intensity adjectives
        intensity_adjectives = ['dangerous', 'powerful', 'devastating', 'lethal', 'destructive', 'aggressive', 'severe', 'critical', 'extreme', 'massive']
        
        # Process all categories and collect results
        category_results = {}
        for category, keywords in threat_keywords.items():
            found_keywords = []
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', all_text))
                if count > 0:
                    found_keywords.append((keyword, count))
            
            if found_keywords:
                category_results[category] = sorted(found_keywords, key=lambda x: x[1], reverse=True)
        
        # Process adjectives
        quality_found = []
        for adj in quality_adjectives:
            count = len(re.findall(r'\b' + re.escape(adj) + r'\b', all_text))
            if count > 0:
                quality_found.append((adj, count))
        quality_found.sort(key=lambda x: x[1], reverse=True)
        
        intensity_found = []
        for adj in intensity_adjectives:
            count = len(re.findall(r'\b' + re.escape(adj) + r'\b', all_text))
            if count > 0:
                intensity_found.append((adj, count))
        intensity_found.sort(key=lambda x: x[1], reverse=True)
        
        # === TOP SUMMARY SECTION ===
        st.subheader("📈 Analysis Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_threats = sum(len(keywords) for keywords in category_results.values())
        total_quality = len(quality_found)
        total_intensity = len(intensity_found)
        
        with col1:
            st.metric("🎯 Threat Terms", total_threats, help="Total unique threat-related terms found")
        with col2:
            st.metric("⭐ Quality Indicators", total_quality, help="Terms indicating sophistication/quality")
        with col3:
            st.metric("⚡ Intensity Markers", total_intensity, help="Terms indicating threat severity")
        with col4:
            st.metric("📄 Documents", f"{sample_size:,}", help="Total documents analyzed")
        
        # === CATEGORY BREAKDOWN WITH TABS ===
        if category_results or quality_found or intensity_found:
            # Create tabs for different categories
            all_categories = list(category_results.keys())
            if quality_found:
                all_categories.append("Sophistication Indicators")
            if intensity_found:
                all_categories.append("Threat Intensity Markers")
            
            if all_categories:
                tabs = st.tabs(all_categories)
                
                # Show threat categories
                for i, category in enumerate(list(category_results.keys())):
                    with tabs[i]:
                        keywords = category_results[category]
                        
                        # Choose emoji based on category
                        emoji_map = {
                            'Malware Types': '🦠',
                            'Attack Methods': '⚔️',
                            'Tools & Software': '🛠️',
                            'Security Evasion': '🥷',
                            'Criminal Services': '🏪',
                            'Target Systems': '🎯',
                            'Payment Terms': '💰',
                            'Quality Indicators': '⭐'
                        }
                        
                        emoji = emoji_map.get(category, '📌')
                        st.markdown(f"## {emoji} {category}")
                        st.caption(f"Found {len(keywords)} terms in this category")
                        
                        # Create columns for better layout
                        if len(keywords) > 4:
                            num_cols = min(3, (len(keywords) + 2) // 3)
                            cols = st.columns(num_cols)
                            
                            for j, (keyword, count) in enumerate(keywords):
                                col_idx = j % num_cols
                                with cols[col_idx]:
                                    percentage = (count / sample_size) * 100
                                    st.metric(
                                        keyword.title(), 
                                        f"{count:,}", 
                                        help=f"Found {count} times in {sample_size:,} documents ({percentage:.2f}%)"
                                    )
                        else:
                            # Single column for few items
                            for keyword, count in keywords:
                                percentage = (count / sample_size) * 100
                                st.metric(
                                    keyword.title(), 
                                    f"{count:,}",
                                    help=f"Found {count} times in {sample_size:,} documents ({percentage:.2f}%)"
                                )
                
                # Add sophistication indicators tab
                if quality_found:
                    tab_idx = len(category_results)
                    with tabs[tab_idx]:
                        st.markdown("## 🌟 Sophistication Indicators")
                        st.caption(f"Found {len(quality_found)} quality/sophistication terms")
                        
                        # Show as metrics in columns
                        if len(quality_found) > 4:
                            num_cols = min(3, (len(quality_found) + 2) // 3)
                            cols = st.columns(num_cols)
                            
                            for j, (adj, count) in enumerate(quality_found):
                                col_idx = j % num_cols
                                with cols[col_idx]:
                                    percentage = (count / sample_size) * 100
                                    st.metric(
                                        adj.title(),
                                        f"{count:,}",
                                        help=f"Found {count} times in {sample_size:,} documents ({percentage:.2f}%)"
                                    )
                        else:
                            for adj, count in quality_found:
                                percentage = (count / sample_size) * 100
                                st.metric(
                                    adj.title(),
                                    f"{count:,}",
                                    help=f"Found {count} times in {sample_size:,} documents ({percentage:.2f}%)"
                                )
                
                # Add intensity markers tab
                if intensity_found:
                    tab_idx = len(category_results) + (1 if quality_found else 0)
                    with tabs[tab_idx]:
                        st.markdown("## ⚡ Threat Intensity Markers")
                        st.caption(f"Found {len(intensity_found)} intensity/severity terms")
                        
                        # Show as metrics in columns
                        if len(intensity_found) > 4:
                            num_cols = min(3, (len(intensity_found) + 2) // 3)
                            cols = st.columns(num_cols)
                            
                            for j, (adj, count) in enumerate(intensity_found):
                                col_idx = j % num_cols
                                with cols[col_idx]:
                                    percentage = (count / sample_size) * 100
                                    st.metric(
                                        adj.title(),
                                        f"{count:,}",
                                        help=f"Found {count} times in {sample_size:,} documents ({percentage:.2f}%)"
                                    )
                        else:
                            for adj, count in intensity_found:
                                percentage = (count / sample_size) * 100
                                st.metric(
                                    adj.title(),
                                    f"{count:,}",
                                    help=f"Found {count} times in {sample_size:,} documents ({percentage:.2f}%)"
                                )
        
        else:
            st.info("No significant keywords found in the analyzed text.")
                
    except Exception as e:
        st.error(f"Keyword analysis failed: {e}")
        st.info("Unable to perform meaningful keyword analysis on the current dataset.")

def show_threat_dashboard(crawled_df, forum_df, threat_keywords):
    """Enhanced threat dashboard with education and trending analysis."""
    st.header("🚨 Threat Intelligence Dashboard")
    st.info("The dashboard helps people easily identify dark web threats and recognize malicious activities.")

    # Educational Section
    show_educational_content(threat_keywords)
    
    # Trending Threats from Market Data
    show_trending_threats(crawled_df, threat_keywords)
    
    # Alert Simulation
    show_alert_simulation(crawled_df, forum_df, threat_keywords)

def show_educational_content(threat_keywords):
    """Interactive educational content about dark web threats."""
    st.subheader("🎓 Dark Web Threat Education")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Common Dark Web Threat Categories")
        
        selected_category = st.selectbox(
            "Select a threat category to learn more:",
            list(threat_keywords.keys()) if threat_keywords else ["No categories available"]
        )
        
        if threat_keywords and selected_category in threat_keywords:
            keywords = threat_keywords[selected_category]
            
            # Educational descriptions
            threat_descriptions = {
                'Ransomware': 'Malicious software that encrypts victim files and demands payment for decryption keys.',
                'Malware': 'General term for malicious software designed to damage, disrupt, or gain unauthorized access to systems.',
                'Crypter': 'Tools used to encrypt malware to avoid detection by antivirus software.',
                'RAT': 'Remote Access Trojans that allow attackers to control victim computers remotely.',
                'Stealer': 'Malware designed to steal sensitive information like passwords, credit cards, and personal data.',
                'Botnet': 'Networks of compromised computers controlled remotely for malicious activities.',
                'Exploit': 'Code that takes advantage of software vulnerabilities to gain unauthorized access.'
            }
            
            description = threat_descriptions.get(selected_category, "A category of cybersecurity threats.")
            st.markdown(f"**{selected_category}**: {description}")
            
            st.markdown("**Common keywords associated with this threat:**")
            keyword_chips = ' '.join([f"`{kw}`" for kw in keywords[:10]])  # Show first 10 keywords
            st.markdown(keyword_chips)
            
            if len(keywords) > 10:
                if st.expander(f"Show all {len(keywords)} keywords"):
                    all_keywords = ', '.join(keywords)
                    st.text(all_keywords)
    
    with col2:
        st.markdown("### ⚠️ Warning Signs & Security Precautions")
        
        st.markdown("**Warning Signs of Dark Web Threats:**")
        warning_signs = [
            "💰 **Unusual pricing**: Extremely low prices for expensive software/services",
            "🔐 **Cryptocurrency payments**: Requests for Bitcoin or other cryptocurrencies only",
            "🕵️ **Anonymity emphasis**: Excessive focus on hiding identity and location",
            "📦 **Suspicious file formats**: Executable files (.exe) disguised as documents",
            "🔗 **Shortened URLs**: Links that hide the actual destination",
            "⏰ **Time pressure**: Claims of limited-time offers or urgent action needed",
            "🎯 **Too good to be true**: Promises of easy money or impossible results",
            "📝 **Poor grammar**: Unusual spelling or grammar mistakes in professional contexts"
        ]
        
        for sign in warning_signs:
            st.markdown(f"- {sign}")
        
def show_trending_threats(crawled_df, threat_keywords):
    """Show trending threats from market data."""
    st.subheader("📈 Trending Threat Intelligence")
    
    if crawled_df is not None and not crawled_df.empty:
        threat_categories = list(threat_keywords.keys()) if threat_keywords else []
        
        # Find posts with threats
        threat_df = crawled_df[crawled_df[threat_categories].any(axis=1)] if threat_categories else pd.DataFrame()
        
        if not threat_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Most Active Threat Categories:**")
                threat_counts = threat_df[threat_categories].sum().sort_values(ascending=False).head(5)
                
                for category, count in threat_counts.items():
                    percentage = (count / len(crawled_df)) * 100
                    st.metric(category, f"{count:,} listings", f"{percentage:.1f}% of total")
            
            with col2:
                st.markdown("**Price Analysis of Threat Listings:**")
                if 'Price_USD' in threat_df.columns:
                    avg_price = threat_df['Price_USD'].mean()
                    median_price = threat_df['Price_USD'].median()
                    max_price = threat_df['Price_USD'].max()
                    
                    if pd.notna(avg_price):
                        st.metric("Average Price", f"${avg_price:.2f}")
                        st.metric("Median Price", f"${median_price:.2f}")
                        st.metric("Highest Price", f"${max_price:.2f}")
        else:
            st.info("No threat-related listings found in the current dataset.")
    else:
        st.warning("No market data available for trending analysis.")

def show_alert_simulation(crawled_df, forum_df, threat_keywords):
    """Simulate alerts based on random trending posts for education purposes."""
    st.subheader("🚨 Live Threat Alerts (Simulation from Crawled Data)")
    
    if crawled_df is not None and not crawled_df.empty:
        threat_categories = list(threat_keywords.keys()) if threat_keywords else []
        
        # Get threat posts from market data
        threat_df = crawled_df[crawled_df[threat_categories].any(axis=1)] if threat_categories else pd.DataFrame()
        
        if not threat_df.empty:
            # Select random threat posts for alerts (random every time)
            num_alerts = min(5, len(threat_df))
            import random
            random_seed = random.randint(1, 10000)
            alert_posts = threat_df.sample(n=num_alerts, random_state=random_seed)
            
            st.markdown("**Selected Alert Detections:**")
            
            for idx, (_, post) in enumerate(alert_posts.iterrows()):
                # Determine which threats were detected
                detected_threats = [cat for cat in threat_categories if post.get(cat, False)]
                
                with st.expander(f"🚨 Alert #{idx+1}: {', '.join(detected_threats)} detected", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Title:** {post.get('Title', 'N/A')}")
                        description = post.get('Description', 'N/A')
                        if len(description) > 200:
                            description = description[:200] + "..."
                        st.markdown(f"**Description:** {description}")
                        
                        if 'Price' in post:
                            st.markdown(f"**Price:** {post['Price']}")
                        
                        st.markdown(f"**Seller:** {post.get('Seller', 'Unknown')}")
                        st.markdown(f"**Category:** {post.get('Category', 'Unknown')}")
                    
                    with col2:
                        # AI-based risk assessment with explanations
                        risk_level = len(detected_threats)
                        risk_explanation = generate_risk_explanation(detected_threats, post)
                        
                        if risk_level >= 2:
                            st.error("🔴 HIGH RISK")
                        elif risk_level == 1:
                            st.warning("🟡 MEDIUM RISK")
                        else:
                            st.info("🟢 LOW RISK")
                        
                        st.markdown("**AI Risk Analysis:**")
                        st.markdown(f"*{risk_explanation}*")
                        
                        st.markdown("**Detected Threats:**")
                        for threat in detected_threats:
                            st.markdown(f"- {threat}")
                        
                        # AI-based sentiment analysis with explanations
                        sentiment = post.get('sentiment', 0)
                        sentiment_explanation = generate_sentiment_explanation(sentiment, post)
                        
                        if sentiment > 0.7:
                            st.success("😊 Positive sentiment")
                        elif sentiment < 0.3:
                            st.error("😠 Negative sentiment")
                        else:
                            st.info("😐 Neutral sentiment")
                        
                        st.markdown("**AI Sentiment Analysis:**")
                        st.markdown(f"*{sentiment_explanation}*")
        else:
            st.info("No threat alerts to display. This indicates the monitoring system is working properly.")
    else:
        st.warning("No data available for alert simulation.")

def generate_risk_explanation(detected_threats, post):
    """Generate risk explanation based on detected threats and post content analysis."""
    title = str(post.get('Title', '')).lower()
    description = str(post.get('Description', '')).lower()
    combined_text = f"{title} {description}"
    
    # Base risk assessment
    risk_score = 0
    factors = []
    
    # Threat category analysis
    if len(detected_threats) >= 2:
        risk_score += 30
        factors.append(f"Multiple threat categories ({len(detected_threats)}) indicate sophisticated malicious operations")
    elif len(detected_threats) == 1:
        risk_score += 15
        factors.append(f"Single threat category ({detected_threats[0]}) detected with focused malicious intent")
    
    # Content analysis for additional risk factors
    high_risk_keywords = ['hack', 'crack', 'steal', 'bypass', 'exploit', 'breach', 'penetrate', 'anonymous', 'untraceable']
    medium_risk_keywords = ['tool', 'software', 'program', 'script', 'access', 'remote', 'control', 'password']
    
    high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in combined_text)
    medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in combined_text)
    
    if high_risk_count > 0:
        risk_score += high_risk_count * 5
        factors.append(f"Contains {high_risk_count} high-risk keywords indicating explicit malicious activities")
    
    if medium_risk_count > 0:
        risk_score += medium_risk_count * 2
        factors.append(f"Contains {medium_risk_count} medium-risk technical terms suggesting tool-based threats")
    
    # Price analysis
    price = post.get('Price', '')
    if price and any(char.isdigit() for char in str(price)):
        try:
            price_nums = ''.join(filter(lambda x: x.isdigit() or x == '.', str(price)))
            if price_nums:
                price_num = float(price_nums)
                if price_num < 10:
                    risk_score += 5
                    factors.append(f"Extremely low price (${price_num:.2f}) suggests scam or mass-distribution tactics")
                elif price_num > 500:
                    risk_score += 3
                    factors.append(f"High price (${price_num:.2f}) indicates premium malicious services")
        except:
            pass
    
    # Urgency and marketing tactics
    urgency_keywords = ['urgent', 'limited', 'exclusive', 'now', 'today', 'fast']
    urgency_count = sum(1 for keyword in urgency_keywords if keyword in combined_text)
    if urgency_count > 0:
        risk_score += urgency_count * 2
        factors.append(f"Uses {urgency_count} urgency tactics to pressure potential victims")
    
    # Technical sophistication indicators
    tech_keywords = ['encryption', 'algorithm', 'protocol', 'api', 'database', 'network']
    tech_count = sum(1 for keyword in tech_keywords if keyword in combined_text)
    if tech_count > 2:
        risk_score += 5
        factors.append(f"Technical sophistication ({tech_count} advanced terms) suggests professional criminal operations")
    
    # Generate explanation based on risk score and factors
    if risk_score >= 25:
        explanation = f"Risk assessment score: {risk_score}/100. This posting presents significant security threats. "
    elif risk_score >= 10:
        explanation = f"Risk assessment score: {risk_score}/100. This posting shows moderate threat indicators. "
    else:
        explanation = f"Risk assessment score: {risk_score}/100. This posting has minimal threat indicators. "
    
    if factors:
        explanation += "Key factors: " + "; ".join(factors[:3]) + "."
    else:
        explanation += "No significant risk factors identified in content analysis."
    
    return explanation

def generate_sentiment_explanation(sentiment_score, post):
    """Generate AI-based sentiment explanation based on content analysis and scoring."""
    title = str(post.get('Title', '')).lower()
    description = str(post.get('Description', '')).lower()
    combined_text = f"{title} {description}"
    
    # Content analysis for sentiment factors
    sentiment_factors = []
    content_score = 0
    
    # Positive sentiment indicators
    positive_words = ['premium', 'quality', 'professional', 'reliable', 'tested', 'guaranteed', 'secure', 'safe', 'trusted', 'verified']
    positive_count = sum(1 for word in positive_words if word in combined_text)
    
    if positive_count > 0:
        content_score += positive_count * 0.1
        sentiment_factors.append(f"{positive_count} positive marketing terms (quality/reliability focus)")
    
    # Service-oriented language
    service_words = ['support', 'help', 'guide', 'tutorial', 'service', 'assistance', 'customer']
    service_count = sum(1 for word in service_words if word in combined_text)
    
    if service_count > 0:
        content_score += service_count * 0.08
        sentiment_factors.append(f"{service_count} customer service oriented terms")
    
    # Negative sentiment indicators
    negative_words = ['hack', 'crack', 'steal', 'destroy', 'damage', 'attack', 'breach', 'penetrate', 'exploit', 'victim']
    negative_count = sum(1 for word in negative_words if word in combined_text)
    
    if negative_count > 0:
        content_score -= negative_count * 0.15
        sentiment_factors.append(f"{negative_count} aggressive/hostile terms indicating malicious intent")
    
    # Anonymity and concealment terms
    concealment_words = ['anonymous', 'untraceable', 'hidden', 'secret', 'stealth', 'bypass', 'evade']
    concealment_count = sum(1 for word in concealment_words if word in combined_text)
    
    if concealment_count > 0:
        content_score -= concealment_count * 0.1
        sentiment_factors.append(f"{concealment_count} concealment/evasion terms")
    
    # Technical neutrality indicators
    technical_words = ['software', 'tool', 'program', 'system', 'method', 'technique', 'algorithm', 'protocol']
    technical_count = sum(1 for word in technical_words if word in combined_text)
    
    if technical_count > 2:
        sentiment_factors.append(f"{technical_count} technical terms suggesting professional approach")
    
    # Generate explanation based on sentiment score and analysis
    if sentiment_score > 0.7:
        explanation = f"Positive sentiment detected (score: {sentiment_score:.2f}). "
        explanation += "This indicates optimistic or promotional language designed to attract customers. "
    elif sentiment_score < 0.3:
        explanation = f"Negative sentiment detected (score: {sentiment_score:.2f}). "
        explanation += "This reflects hostile, aggressive, or explicitly harmful language patterns. "
    else:
        explanation = f"Neutral sentiment detected (score: {sentiment_score:.2f}). "
        explanation += "This suggests factual, technical, or business-like communication without strong emotional bias. "
    
    # Add content analysis insights
    if sentiment_factors:
        explanation += f"Content analysis reveals: {'; '.join(sentiment_factors[:3])}. "
    
    # Contextual interpretation
    if sentiment_score > 0.7:
        explanation += "High positive sentiment in criminal contexts often indicates sophisticated marketing tactics to build trust and credibility."
    elif sentiment_score < 0.3:
        explanation += "Negative sentiment typically correlates with explicit criminal intent and aggressive cyber activities."
    else:
        explanation += "Neutral sentiment may indicate professional criminal services attempting to appear legitimate through objective descriptions."
    
    return explanation

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    main()