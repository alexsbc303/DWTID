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
from datetime import datetime, timedelta
import warnings
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from gensim import corpora
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

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

@st.cache_data
def load_datasets():
    """Load and preprocess the datasets"""
    st.info("Loading datasets...")
    crawled_result_df = pd.read_csv('dark_market_output_v2.csv', encoding='latin-1', on_bad_lines='skip', engine='python', low_memory=True)
    preprocessed_forum_df = pd.read_csv('Clearnedup_ALL_7.csv', encoding='latin-1', on_bad_lines='skip', engine='python', low_memory=True)
    st.success("Datasets loaded successfully.")
    return crawled_result_df, preprocessed_forum_df

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
        return crawled_result_df, preprocessed_forum_df
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        st.info("Pre-processed data not found. Processing from source...")
        crawled_result_df, preprocessed_forum_df = load_datasets()
        
        if crawled_result_df is None or preprocessed_forum_df is None:
            return None, None

        crawled_result_df, preprocessed_forum_df = preprocess_data_full(crawled_result_df, preprocessed_forum_df, threat_keywords)
        
        with open(crawled_pkl_path, 'wb') as f:
            pickle.dump(crawled_result_df, f)
        with open(forum_pkl_path, 'wb') as f:
            pickle.dump(preprocessed_forum_df, f)
            
        st.success("Data processed and cached successfully.")
        return crawled_result_df, preprocessed_forum_df

@st.cache_data
def preprocess_data_full(crawled_result_df, preprocessed_forum_df, threat_keywords):
    """The full preprocessing pipeline - optimized version."""
    # Pre-compile regexes for faster threat detection
    compiled_regexes = {
        category: re.compile('|'.join(keywords), re.IGNORECASE)
        for category, keywords in threat_keywords.items()
    }
    
    # --- Preprocess Crawled Market Data ---
    if crawled_result_df is not None:
        st.info(f"Processing {len(crawled_result_df):,} market records...")
        crawled_result_df = process_market_data(crawled_result_df, compiled_regexes)

    # --- Preprocess Forum Data ---
    if preprocessed_forum_df is not None:
        st.info(f"Processing {len(preprocessed_forum_df):,} forum records...")
        preprocessed_forum_df = process_forum_data(preprocessed_forum_df, compiled_regexes)

    return crawled_result_df, preprocessed_forum_df

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
    """Process forum data efficiently with chunking for large datasets."""
    df = df.copy()
    df['Post Content'] = df['Post Content'].fillna('')
    df['Thread Title'] = df['Thread Title'].fillna('')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['combined_text'] = (df['Thread Title'].astype(str) + ' ' + df['Post Content'].astype(str)).str.lower()

    # Process in chunks for large datasets
    chunk_size = 100000
    if len(df) > chunk_size:
        st.info(f"Processing large dataset in chunks of {chunk_size:,}...")
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            chunk = process_forum_chunk(chunk, compiled_regexes)
            chunks.append(chunk)
            st.progress((i + chunk_size) / len(df))
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = process_forum_chunk(df, compiled_regexes)
    
    return df

@st.cache_data
def process_forum_chunk(chunk, compiled_regexes):
    """Process a chunk of forum data."""
    # Vectorized threat detection
    for category, regex in compiled_regexes.items():
        chunk[category] = chunk['combined_text'].str.contains(regex, na=False)
    
    # Efficient sentiment analysis
    chunk['sentiment'] = compute_sentiment_efficient(chunk['combined_text'])
    
    return chunk

@st.cache_data
def compute_sentiment_efficient(text_series, sample_size=10000):
    """Compute sentiment efficiently using sampling for large datasets."""
    analyzer = SentimentIntensityAnalyzer()
    
    if len(text_series) > sample_size:
        # Sample-based approach for large datasets
        unique_texts = text_series.drop_duplicates()
        if len(unique_texts) > sample_size:
            sampled_texts = unique_texts.sample(n=sample_size, random_state=42)
        else:
            sampled_texts = unique_texts
        
        # Compute sentiment for sample
        sentiment_map = {}
        for text in sampled_texts:
            if len(str(text).strip()) > 0:
                sentiment_map[text] = analyzer.polarity_scores(str(text))['compound']
            else:
                sentiment_map[text] = 0.0
        
        # Map back to full dataset with default for unmapped texts
        result = text_series.map(sentiment_map)
        result = result.fillna(0.0)  # Default sentiment for unmapped texts
    else:
        # Direct computation for smaller datasets
        unique_texts = text_series.drop_duplicates()
        sentiment_map = {}
        for text in unique_texts:
            if len(str(text).strip()) > 0:
                sentiment_map[text] = analyzer.polarity_scores(str(text))['compound']
            else:
                sentiment_map[text] = 0.0
        result = text_series.map(sentiment_map)
    
    return result

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
    
    threat_keywords = load_threat_keywords()
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
        show_alerting_system(preprocessed_forum_df, threat_keywords)
    
    elif analysis_choice == 'Crawled Result Analysis':
        st.header("Analysis of Crawled Market Data")
        show_overview(crawled_result_df, "market")
        show_threat_analysis(crawled_result_df, threat_categories)
        show_crawled_result_analysis(crawled_result_df)
        show_network_analysis(crawled_result_df, "Seller", "Category")
        show_ml_classification(crawled_result_df, threat_categories)
    
    elif analysis_choice == 'Preprocessed Forum Datasets Analysis':
        st.header("Analysis of Preprocessed Forum Data")
        show_overview(preprocessed_forum_df, "forum")
        show_threat_analysis(preprocessed_forum_df, threat_categories)
        show_forum_analysis(preprocessed_forum_df)
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
        threat_counts = threat_df[threat_categories].sum().sort_values(ascending=False)
        fig = px.bar(threat_counts, x=threat_counts.index, y=threat_counts.values, title='Detected Threat Categories')
        st.plotly_chart(fig, use_container_width=True)

        if 'datetime' in threat_df.columns:
            st.subheader("Threat Mentions Over Time")
            threat_temporal = threat_df.set_index('datetime').resample('M')[threat_categories].sum()
            fig = px.line(threat_temporal, x=threat_temporal.index, y=threat_temporal.columns, title='Threat Mentions per Month')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detected Threat Listings")
        selected_threats = st.multiselect("Filter by Threat Type", options=threat_categories)
        
        if selected_threats:
            filtered_df = threat_df[threat_df[selected_threats].all(axis=1)]
        else:
            filtered_df = threat_df
        
        if not filtered_df.empty:
            st.dataframe(filtered_df)
        else:
            st.info("No data matches the selected threat filters.")

        st.subheader("Threat Word Cloud")
        text_content = ' '.join(filtered_df['combined_text'].dropna())
        if text_content and len(text_content.strip()) > 0:
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_content)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            except ValueError as e:
                st.warning(f"Could not generate word cloud: {e}")
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
    
    with col2:
        price_data = crawled_result_df[crawled_result_df['Price_USD'].notna() & (crawled_result_df['Price_USD'] > 0)]
        if len(price_data) > 0:
            fig = px.histogram(price_data, x='Price_USD', title='Price Distribution', nbins=50)
            fig.update_xaxes(range=[0, price_data['Price_USD'].quantile(0.95)])
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Seller Analysis")
    seller_stats = crawled_result_df.groupby('Seller').agg({'Title': 'count', 'Price_USD': 'mean'}).sort_values('Title', ascending=False).head(20)
    seller_stats.columns = ['Listings Count', 'Average Price']
    
    fig = px.scatter(seller_stats, x='Listings Count', y='Average Price', title='Seller Activity vs Average Price', hover_data={'Listings Count': True, 'Average Price': True})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment Analysis")
    sentiment_counts = crawled_result_df['sentiment'].apply(lambda p: 'positive' if p > 0.1 else ('negative' if p < -0.1 else 'neutral')).value_counts()
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment of Market Listings')
    st.plotly_chart(fig, use_container_width=True)

def show_forum_analysis(forum_df):
    """Show forum analysis"""
    st.header("💬 Forum Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forum_activity = forum_df['Forum Name'].value_counts().head(10)
        fig = px.bar(x=forum_activity.index, y=forum_activity.values, title='Forum Activity (Posts Count)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        user_types = forum_df['User Type'].value_counts()
        fig = px.pie(values=user_types.values, names=user_types.index, title='User Types Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    if 'datetime' in forum_df.columns:
        st.subheader("Temporal Analysis")
        valid_dates = forum_df[forum_df['datetime'].notna()].copy()
        if len(valid_dates) > 0:
            valid_dates['date'] = valid_dates['datetime'].dt.date
            daily_posts = valid_dates.groupby('date').size().reset_index(name='posts')
            fig = px.line(daily_posts, x='date', y='posts', title='Forum Activity Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Sentiment Analysis")
    sentiment_counts = forum_df['sentiment'].apply(lambda p: 'positive' if p > 0.1 else ('negative' if p < -0.1 else 'neutral')).value_counts()
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='Sentiment of Forum Posts')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Content Analysis")
    if st.button("Generate Word Cloud"):
        st.warning("Generating word cloud from a sample of 20,000 posts for performance reasons.")
        sample_df = forum_df.sample(min(20000, len(forum_df)))
        text_content = ' '.join(sample_df['Post Content'].dropna().astype(str))
        
        if len(text_content) > 0:
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_content)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            except ValueError as e:
                st.warning(f"Could not generate word cloud: {e}")

    show_topic_modeling(forum_df)

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
    """Show ML classification analysis"""
    st.header("🤖 Machine Learning Classification")
    st.subheader("Post Classification Model")
    
    if st.button("Train Classification Model"):
        with st.spinner("Training model..."):
            X, y = prepare_classification_data(df, threat_categories)
            
            if len(X) > 10 and y.sum() > 0:
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
                    
                    feature_df = pd.DataFrame({'Feature': [feature_names[i] for i in top_features], 'Importance': [feature_importance[i] for i in top_features]}).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h', title='Top 20 Predictive Features for "Threat" Class')
                    st.plotly_chart(fig, use_container_width=True)
                except (ValueError, IndexError) as e:
                    st.warning(f"Could not calculate feature importance: {e}")

def show_topic_modeling(forum_df, num_topics=5, num_words=10):
    """Perform topic modeling on forum posts."""
    st.header("🔬 Topic Modeling")
    
    if not GENSIM_AVAILABLE:
        st.error("Topic modeling is not available. Gensim library failed to import due to dependency issues.")
        return
    
    if st.button("Run Topic Modeling"):
        with st.spinner("Running LDA Topic Modeling... This may take a while."):
            try:
                stop_words = set(stopwords.words('english'))
                def preprocess_text(text):
                    tokens = word_tokenize(text.lower())
                    return [word for word in tokens if word.isalpha() and word not in stop_words]

                sample_df = forum_df.sample(min(10000, len(forum_df)))
                processed_docs = sample_df['Post Content'].apply(preprocess_text)
                
                dictionary = corpora.Dictionary(processed_docs)
                corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
                
                lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
                
                st.subheader("Discovered Topics")
                topics = lda_model.print_topics(num_words=num_words)
                for topic in topics:
                    st.write(f"**Topic {topic[0]}:** {topic[1]}")
            except Exception as e:
                st.error(f"Topic modeling failed: {e}")
                st.info("This may be due to compatibility issues with the current environment.")

def show_alerting_system(forum_df, threat_keywords):
    """Simulate an alerting system for high-risk activities."""
    st.header("🚨 Alerting System")
    st.info("This system highlights recent posts containing high-threat keywords.")

    # Get high-threat keywords (customize as needed)
    high_threat_keywords = threat_keywords.get('Ransomware', []) + threat_keywords.get('Malware', [])

    # Filter for recent posts (e.g., last 30 days)
    if 'datetime' in forum_df.columns and forum_df['datetime'].notna().any():
        recent_df = forum_df[forum_df['datetime'].notna() & (forum_df['datetime'] > (datetime.now() - timedelta(days=30)))]
        
        if not recent_df.empty:
            # Find posts with high-threat keywords
            if high_threat_keywords:
                alert_mask = recent_df['combined_text'].str.contains('|'.join(high_threat_keywords), case=False, na=False)
                alert_df = recent_df[alert_mask]
            else:
                alert_df = pd.DataFrame()

            if not alert_df.empty:
                st.subheader("Recent High-Risk Posts")
                st.dataframe(alert_df[['datetime', 'Forum Name', 'Thread Title', 'Username', 'Post Content']])
            else:
                st.success("No high-risk posts detected in the last 30 days.")
        else:
            st.info("No forum posts from the last 30 days in the dataset.")

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