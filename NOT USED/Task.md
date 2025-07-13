Please assist in making a darkweb Malware Monitoring Prototype in web application utilising 2 set of databases (1 crawled from darkweb called dark_market_output_v2.csv & 1 called Clearnedup_ALL_7.csv from the GitHub repository ([https://github.com/gayanku/darkweb_clearweb_darktopics](https://github.com/gayanku/darkweb_clearweb_darktopics))), following the below parts :

This prototype will monitor dark web forums and marketplaces for malware-related activities guided by the DWTIA framework’s intelligence pipeline and the keyword-based collection strategy.

2. Keyword-Based Malware Detection Engine: Uses static keyword matching, Named Entity Recognition (NER), and topic modeling (LDA) to identify malware types and cybersecurity-related discussions.
3. Structured Threat Intelligence Parser: Extracts key data fields such as malware name, type, seller identity, pricing, and indicators of compromise (e.g. file hashes, IP addresses).
4. Malware Intelligence Dashboard: Built with Streamlit or Dash to visualize trends such as active sellers, commonly advertised malware, and forum activity. NetworkX with static or dummy data to visualize relationships between malware types, sellers, and forums will be adopted. On top of the dashboard, please allow users to filter and search for the malware findings.
5. Alerting System: Issues alerts when specific keywords, malware families, or new variants are detected. Notifications will be simulated through email or dashboard indicators.
6. Post Classification Module: Employs a machine learning classifier (e.g. Scikit-learn or BERT) to categorise posts into malware ads or service requests. You can further use more AI technologies based on the crawled data

Task.md Objectives Verification:

  1. Keyword-Based Malware Detection Engine ✅
    - app_cleaned.py:158-165: Compiled regex patterns for
  efficient keyword detection
    - app_cleaned.py:257: Vectorized threat detection using
  compiled regexes
    - app_cleaned.py:847-981: Topic modeling (LDA)
  implementation with fallback to keyword analysis
    - app_cleaned.py:982-1201: Comprehensive keyword analysis
  with threat categories
  2. Structured Threat Intelligence Parser ✅
    - app_cleaned.py:196-251: Market data processing with price
  extraction
    - app_cleaned.py:265-300: Forum data processing with
  datetime parsing
    - app_cleaned.py:198-217: Data preprocessing pipeline
  extracting key fields
    - app_cleaned.py:253-261: Sentiment analysis for
  intelligence enrichment
  3. Malware Intelligence Dashboard ✅
    - app_cleaned.py:39-60: Streamlit configuration and UI
  styling
    - app_cleaned.py:402-431: Overview dashboards with dataset
  statistics
    - app_cleaned.py:432-566: Advanced threat filtering and
  search capabilities
    - app_cleaned.py:653-739: NetworkX network analysis and
  visualization
    - app_cleaned.py:567-664: Market and forum analysis
  visualizations
  4. Alerting System ✅
    - app_cleaned.py:1337-1475: Alert simulation system with
  configurable thresholds
    - app_cleaned.py:1445-1529: AI-based risk and sentiment
  explanations
    - app_cleaned.py:1271-1335: Trending threat analysis for
  proactive alerting
    - Dashboard indicators and real-time alert generation
  5. Post Classification Module ✅
    - app_cleaned.py:741-845: Machine learning classification
  using TfidfVectorizer and MultinomialNB
    - app_cleaned.py:758-761: Data preparation for binary threat
   classification
    - app_cleaned.py:820-845: Classification results with
  metrics and feature importance
    - Scikit-learn based implementation as specified
  6. Additional AI Technologies ✅
    - app_cleaned.py:253-320: VADER sentiment analysis
    - app_cleaned.py:1379-1529: AI-based risk assessment and
  sentiment explanation
    - app_cleaned.py:1216-1269: Educational content with threat
  intelligence
    - Advanced threat intelligence framework implementation