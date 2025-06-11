import streamlit as st
from streamlit_float import *
import preprocessor, helper, classifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from classifier import WhatsAppClassifier
import seaborn as sns
from datetime import datetime, timedelta
from helper import forecast_message_trends
from helper import evaluate_forecasting_algorithms
import pandas as pd
import matplotlib.font_manager as fm
import os
from sklearn.linear_model import LinearRegression
import preprocessor
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tabulate import tabulate
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from whatsapp_reminder import WhatsAppReminderSystem
from auth import GoogleAuth

# Import chatbot components
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxLLM
from langchain_community.vectorstores import FAISS

# Initialize float feature for floating elements
float_init()

# Initialize session state for filters and chatbot
if 'category_filter' not in st.session_state:
    st.session_state.category_filter = ["Important", "Personal", "Spam", "Media"]
if 'selected_recency' not in st.session_state:
    st.session_state.selected_recency = "All time"
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
# Initialize chatbot visibility state
if 'show_chatbot' not in st.session_state:
    st.session_state.show_chatbot = False
if 'show_reminders' not in st.session_state:
    st.session_state.show_reminders = False
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'auth' not in st.session_state:
    st.session_state.auth = GoogleAuth()

# Create the IBM Watson model for chatbot
llm = WatsonxLLM(
    model_id='ibm/granite-3-8b-instruct',
    apikey='rKEBYvSNbsFhwEbHvIVkzl3eVrwZQoUac_gzmp-vrM3C',
    url='https://eu-de.ml.cloud.ibm.com',
    project_id='2cfa9ea3-b758-4053-933a-06e8fa35276b',
    params={
        'decoding_method': 'sample',
        'max_new_tokens': 200,
        'temperature': 1
    }
)

# Forecasting algorithm summaries for display
FORECAST_ALGO_SUMMARIES = {
    "Linear Regression": (
       "Linear Regression fits a straight line to your historical message counts, showing the average trend over time. The predicted values appear as a flat or gently sloped line, representing the expected average message count. This model does not capture daily fluctuations, seasonality, or sudden spikes-it assumes the future will follow the same general trend as the past. Use this for a simple overview, but consider more advanced models for detailed patterns"
    ),
    "ARIMA": (
       "ARIMA (AutoRegressive Integrated Moving Average) is a classic statistical model for time series forecasting. It predicts future message counts by analyzing both past values and past errors, capturing trends and some short-term patterns. ARIMA can handle data with trends and cycles but requires the data to be stationary (no overall change in mean/variance over time). It’s more flexible than linear regression and can model some ups and downs, but may still miss complex seasonal effects"
    ),
    "Prophet": (
        "Prophet is an advanced forecasting tool developed by Facebook, designed for time series with strong seasonal effects and missing data. It automatically detects and models trends, recurring patterns (like weekly or yearly cycles), and even the impact of holidays or special events. Prophet is robust, user-friendly, and provides interpretable forecasts, making it ideal for chat data with regular activity cycles"
    ),
    "Exponential Smoothing": (
        "Exponential Smoothing (including Holt-Winters) forecasts future values by giving more weight to recent message counts, allowing it to adapt to gradual changes and seasonality. It’s effective for data with consistent trends or regular cycles (like weekly chat activity). The model smooths out random noise but may not capture sudden spikes or complex patterns"
    ),
    "LSTM": (
        "LSTM is a deep learning model specialized for sequential data. It learns from long-term patterns in your message history, making it powerful for capturing complex trends, seasonality, and even irregular spikes. LSTM adapts to non-linear and non-stationary data, often outperforming traditional models when enough data is available. However, it requires more data and computational resources to train effectively"
    ),
}


# Configure matplotlib for emoji display
def set_matplotlib_emoji_font():
    # Try to find a font that can display emojis
    emoji_fonts = ['Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', 'Symbola', 'Arial Unicode MS']
    font_path = None
    for font in emoji_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font))
            if font_path:
                plt.rcParams['font.family'] = font
                break
        except:
            continue
    if not font_path:
        print("Warning: No emoji font found. Emojis might not display correctly.")


# Set emoji font
set_matplotlib_emoji_font()

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

labeled_data_path = "labeled_data.csv"
if not os.path.exists(labeled_data_path):
    st.error(f"Labeled data file '{labeled_data_path}' not found in the project directory. Please add it.")
else:
    labeled_data = pd.read_csv(labeled_data_path)

if not st.session_state.authenticated:
    st.title("WhatsApp Chat Analyzer")
    st.write("Please log in with your Google account to continue.")

    # Check if we're returning from auth flow
    query_params = st.query_params
    if 'code' in query_params:
        with st.spinner("Completing authentication..."):
            if st.session_state.auth.authenticate():
                st.session_state.authenticated = True
                user_info = st.session_state.auth.get_user_info()
                st.session_state.user_email = user_info.get('email')
                st.success(f"Successfully logged in as {st.session_state.user_email}")
                st.rerun()
    else:
        # Show login button
        if st.button("Login with Google"):
            auth_result = st.session_state.auth.authenticate()
            if auth_result:  # True means already authenticated
                st.session_state.authenticated = True
                user_info = st.session_state.auth.get_user_info()
                st.session_state.user_email = user_info.get('email')
                st.success(f"Successfully logged in as {st.session_state.user_email}")
                st.rerun()
            # If auth_result is False, the authenticate method has displayed the login link

    st.stop()

else:
    # Main app content
    st.sidebar.title("WhatsApp Chat Analyzer")

    # Add logout button
    if st.sidebar.button("Logout"):
        st.session_state.auth.logout()
        st.session_state.authenticated = False
        if 'user_email' in st.session_state:
            del st.session_state.user_email
        st.rerun()

    # Display user email
    if 'user_email' in st.session_state:
        st.sidebar.write(f"Logged in as: {st.session_state.user_email}")

    # Add reminder button to sidebar

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    try:
        data = bytes_data.decode("utf-8")  # Try decoding as UTF-8
    except UnicodeDecodeError:
        st.error("Failed to decode the file as UTF-8. Please ensure it's a text file.")
    else:
        df = preprocessor.preprocess(data)  # Process the data only if decoding was successful
        if df.empty:
            st.error("No data could be processed. Please check if the file is a valid WhatsApp chat export.")
        else:
            # Convert date column to datetime if it's not already
            if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])

            # fetch unique users
            user_list = df['user'].unique().tolist()
            if 'group_notification' in user_list:
                user_list.remove('group_notification')
            user_list.sort()
            user_list.insert(0, "Overall")

            selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

            if st.sidebar.button("Show Analysis"):

              tab1, tab2, tab3=st.tabs(["Statical Analysis", "Sentiment Analysis", "Forecasting"])

              with tab1:
                num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
                st.title("Top Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.header("Total Messages")
                    st.title(num_messages)
                with col2:
                    st.header("Total Words")
                    st.title(words)
                with col3:
                    st.header("Media Shared")
                    st.title(num_media_messages)
                with col4:
                    st.header("Links Shared")
                    st.title(num_links)

                # monthly timeline
                st.title("Monthly Timeline")
                timeline = helper.monthly_timeline(selected_user, df)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

                # daily timeline
                st.title("Daily Timeline")
                daily_timeline = helper.daily_timeline(selected_user, df)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

                # activity map
                st.title('Activity Map')
                col1, col2 = st.columns(2)

                with col1:
                    st.header("Most busy day")
                    busy_day = helper.week_activity_map(selected_user, df)
                    fig, ax = plt.subplots()
                    ax.bar(busy_day.index, busy_day.values, color='purple')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                with col2:
                    st.header("Most busy month")
                    busy_month = helper.month_activity_map(selected_user, df)
                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='orange')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                st.title("Weekly Activity Map")
                user_heatmap = helper.activity_heatmap(selected_user, df)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)

                # finding the busiest users in the group(Group level)
                if selected_user == 'Overall':
                    st.title('Most Busy Users')
                    x, new_df = helper.most_busy_users(df)
                    fig, ax = plt.subplots()
                    col1, col2 = st.columns(2)

                    with col1:
                        ax.bar(x.index, x.values, color='red')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)

                    with col2:
                        st.dataframe(new_df)

                # WordCloud
                st.title("Wordcloud")
                df_wc = helper.create_wordcloud(selected_user, df)
                if df_wc is not None:
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)

                # most common words
                most_common_df = helper.most_common_words(selected_user, df)
                if not most_common_df.empty:
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df['Word'], most_common_df['Count'])
                    plt.xticks(rotation='vertical')
                    st.title('Most common words')
                    st.pyplot(fig)

                # emoji analysis
                emoji_df = helper.emoji_helper(selected_user, df)
                st.title("Emoji Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(emoji_df)

                with col2:
                    if not emoji_df.empty:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        # Get top 5 emojis for the pie chart
                        top_emojis = emoji_df.head(5)
                        # Create pie chart with larger text
                        wedges, texts, autotexts = ax.pie(
                            top_emojis['Count'],
                            labels=top_emojis['Emoji'],
                            autopct='%1.1f%%',
                            textprops={'fontsize': 20}  # Increase emoji size
                        )
                        # Set title with appropriate font size
                        ax.set_title("Top 5 Emojis", pad=20, fontsize=15)
                        st.pyplot(fig)
                    else:
                        st.write("No emojis found in the selected messages")

                # Provide a download button for the processed CSV
                csv_data = df.to_csv(index=False).encode('utf-8')
                # Add a download button for the processed data after analysis is displayed
                st.download_button(
                    label="Download Processed Data as CSV",
                    data=csv_data,
                    file_name="processed_chat.csv",
                    mime="text/csv"
                )

              with tab2:
                  st.title("Sentiment Analysis")

                  # Make sure NLTK Vader lexicon is downloaded
                  import nltk
                  import ssl

                  try:
                      # Create unverified HTTPS context to avoid SSL certificate issues
                      try:
                          _create_unverified_https_context = ssl._create_unverified_context
                      except AttributeError:
                          pass
                      else:
                          ssl._create_default_https_context = _create_unverified_https_context

                      nltk.download('vader_lexicon', quiet=True)
                  except Exception as e:
                      st.error(f"Error downloading NLTK resources: {e}")

                  try:
                      # Get sentiment analysis results
                      sentiment_df = helper.analyze_sentiment(df, selected_user)

                      if sentiment_df.empty:
                          st.error("No messages found for sentiment analysis after filtering.")
                      else:
                          # Display overall sentiment distribution
                          st.subheader("Overall Sentiment Distribution")
                          sentiment_counts = sentiment_df['sentiment_category'].value_counts()

                          col1, col2 = st.columns(2)

                          with col1:
                              fig, ax = plt.subplots()
                              ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                                     colors=['#99ff99', '#ff9999', '#66b3ff'])
                              ax.set_title('Sentiment Distribution')
                              st.pyplot(fig)

                          with col2:
                              st.dataframe(sentiment_counts.reset_index().rename(
                                  columns={'index': 'Sentiment', 'sentiment_category': 'Count'}))

                          # Display sentiment over time
                          st.subheader("Sentiment Trends Over Time")

                          # Group by date and calculate average sentiment
                          sentiment_over_time = sentiment_df.groupby('only_date').agg({
                              'positive': 'mean',
                              'negative': 'mean',
                              'neutral': 'mean',
                              'compound': 'mean'
                          }).reset_index()

                          # Plot sentiment over time
                          fig, ax = plt.subplots(figsize=(10, 6))
                          ax.plot(sentiment_over_time['only_date'], sentiment_over_time['positive'],
                                  label='Positive', color='green')
                          ax.plot(sentiment_over_time['only_date'], sentiment_over_time['negative'],
                                  label='Negative', color='red')
                          ax.plot(sentiment_over_time['only_date'], sentiment_over_time['compound'],
                                  label='Compound', color='blue')
                          ax.set_xlabel('Date')
                          ax.set_ylabel('Sentiment Score')
                          ax.legend()
                          plt.xticks(rotation=45)
                          plt.tight_layout()
                          st.pyplot(fig)

                          # User sentiment comparison (if group chat)
                          if selected_user == 'Overall' and len(df['user'].unique()) > 1:
                              st.subheader("User Sentiment Comparison")

                              user_sentiment = sentiment_df.groupby('user').agg({
                                  'compound': 'mean',
                                  'message': 'count'
                              }).reset_index()

                              # Filter users with at least 5 messages
                              user_sentiment = user_sentiment[user_sentiment['message'] >= 5]

                              if len(user_sentiment) > 0:
                                  # Sort by compound score
                                  user_sentiment = user_sentiment.sort_values(by='compound', ascending=False)

                                  fig, ax = plt.subplots(figsize=(10, 6))
                                  bars = ax.bar(user_sentiment['user'], user_sentiment['compound'],
                                                color=[plt.cm.RdYlGn(0.5 + 0.5 * score) for score in
                                                       user_sentiment['compound']])
                                  ax.set_xlabel('User')
                                  ax.set_ylabel('Average Sentiment Score')
                                  plt.xticks(rotation=45)
                                  plt.tight_layout()
                                  st.pyplot(fig)
                              else:
                                  st.warning("No users with 5+ messages for sentiment comparison")

                              # Top users with positive, negative, and neutral messages
                              st.subheader("Top Users by Sentiment")

                              # Get top users for each sentiment category
                              top_positive = sentiment_df[sentiment_df['sentiment_category'] == 'Positive'].groupby(
                                  'user').size().sort_values(ascending=False).head(5)
                              top_negative = sentiment_df[sentiment_df['sentiment_category'] == 'Negative'].groupby(
                                  'user').size().sort_values(ascending=False).head(5)
                              top_neutral = sentiment_df[sentiment_df['sentiment_category'] == 'Neutral'].groupby(
                                  'user').size().sort_values(ascending=False).head(5)



                              col1, col2, col3 = st.columns(3)

                              with col1:
                                  st.markdown("### Most Positive Users")
                                  if not top_positive.empty:
                                      fig, ax = plt.subplots()
                                      ax.bar(top_positive.index, top_positive.values, color='green')
                                      plt.xticks(rotation=45)
                                      ax.set_ylabel('Number of Positive Messages')
                                      st.pyplot(fig)
                                  else:
                                      st.write("No positive messages found")

                              with col2:
                                  st.markdown("### Most Negative Users")
                                  if not top_negative.empty:
                                      fig, ax = plt.subplots()
                                      ax.bar(top_negative.index, top_negative.values, color='red')
                                      plt.xticks(rotation=45)
                                      ax.set_ylabel('Number of Negative Messages')
                                      st.pyplot(fig)
                                  else:
                                      st.write("No negative messages found")

                              with col3:
                                  st.markdown("### Most Neutral Users")
                                  if not top_neutral.empty:
                                      fig, ax = plt.subplots()
                                      ax.bar(top_neutral.index, top_neutral.values, color='blue')
                                      plt.xticks(rotation=45)
                                      ax.set_ylabel('Number of Neutral Messages')
                                      st.pyplot(fig)
                                  else:
                                      st.write("No neutral messages found")

                  except Exception as e:
                      st.error(f"Error in sentiment analysis: {str(e)}")
                      import traceback

                      st.error(f"Detailed error: {traceback.format_exc()}")

              with tab3:
                  st.title("Message Trend Forecasting")


                  @st.fragment
                  def forecasting_content():
                      try:
                          plt.close('all')  # Clear previous figures

                          # Algorithm selection
                          algorithm = st.selectbox(
                              "Select Forecasting Algorithm",
                              ["Linear Regression", "ARIMA", "Prophet", "Exponential Smoothing", "LSTM"],
                              key='forecast_algorithm'
                          )

                          # Initialize session state for slider
                          if 'forecast_days' not in st.session_state:
                              st.session_state.forecast_days = 30

                          # Slider with session state management
                          future_days = st.slider(
                              "Select forecast days", 10, 60,
                              key='forecast_days',
                              help="Choose number of days to forecast"
                          )

                          # Get forecasting data and model
                          message_counts, model = helper.forecast_message_trends(df, algorithm)
                          last_date = df['only_date'].max()

                          # Generate predictions based on algorithm
                          if algorithm == "Linear Regression":
                              # Create numeric representation for prediction
                              last_numeric_day = message_counts['date_numeric'].max()
                              future_numeric_days = np.array(
                                  [(last_numeric_day + i + 1) for i in range(future_days)]
                              ).reshape(-1, 1)
                              future_predictions = model.predict(future_numeric_days)

                          elif algorithm == "ARIMA":
                              future_predictions = model.forecast(steps=future_days).values

                          elif algorithm == "Prophet":
                              future = model.make_future_dataframe(periods=future_days)
                              forecast = model.predict(future)
                              future_predictions = forecast['yhat'][-future_days:].values

                          elif algorithm == "Exponential Smoothing":
                              future_predictions = model.forecast(steps=future_days).values

                          elif algorithm == "LSTM":
                              # Get the model components
                              lstm_model = model['lstm_model']
                              scaler = model['scaler']
                              look_back = model['look_back']
                              last_sequence = model['last_sequence'].copy()

                              # Generate predictions one by one
                              future_predictions = []
                              current_sequence = last_sequence.copy()

                              for _ in range(future_days):
                                  # Predict next value
                                  next_pred = lstm_model.predict(current_sequence, verbose=0)[0][0]
                                  future_predictions.append(next_pred)

                                  # Update sequence for next prediction - FIXED DIMENSION ISSUE
                                  current_sequence = current_sequence.copy()
                                  current_sequence[:, :-1, :] = current_sequence[:, 1:, :]
                                  current_sequence[:, -1, 0] = next_pred

                              # Inverse transform to get original scale
                              future_predictions = scaler.inverse_transform(
                                  np.array(future_predictions).reshape(-1, 1)
                              ).flatten()

                          # Create future date range
                          future_dates = pd.date_range(
                              start=last_date + timedelta(days=1),
                              periods=future_days
                          )

                          # Visualization
                          fig, ax = plt.subplots(figsize=(14, 8))
                          ax.plot(message_counts['only_date'], message_counts['count'], label='Historical Data')
                          ax.plot(future_dates, future_predictions, label='Forecast', linestyle='--')
                          ax.set(xlabel='Date', ylabel='Message Count', title=f'Message Trend Forecast ({algorithm})')
                          ax.legend()
                          plt.xticks(rotation=45)
                          plt.tight_layout()
                          st.pyplot(fig)
                          st.info(FORECAST_ALGO_SUMMARIES.get(algorithm, ""))
                          # Model evaluation metrics
                          st.subheader("Model Evaluation")
                          col1, col2 = st.columns(2)
                          with col1:
                              st.metric("Forecast Period", f"{future_days} days")
                          with col2:
                              st.metric("Last Historical Date", last_date.strftime("%Y-%m-%d"))

                          results_df = evaluate_forecasting_algorithms(df, test_size=30)
                          print("\nForecasting Algorithm Performance Comparison:")
                          print(tabulate(results_df, headers='keys', tablefmt='github', showindex=False))
                      except Exception as e:
                          st.error(f"Forecasting failed: {str(e)}")


                  # Call the fragment function
                  forecasting_content()

            # Add model selection dropdown for training
            if 'classifier' not in st.session_state:
                st.session_state.classifier = WhatsAppClassifier()

            # In the sidebar
            if st.sidebar.button("Train Classifier"):
                if os.path.exists(labeled_data_path):
                    with st.spinner("Training intelligent classifier..."):
                        try:
                            st.session_state.classifier.train(labeled_data_path)
                            st.success("Classifier trained successfully!")
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                else:
                    st.error("Labeled data not found!")

            if st.sidebar.button("Categorize Messages"):
                if 'message' in df.columns:
                    try:
                        with st.spinner("Analyzing messages with AI..."):
                            # First handle system notifications explicitly
                            system_mask = df['user'] == 'group_notification'

                            # Get predictions for non-system messages
                            non_system_messages = df[~system_mask]['message'].tolist()
                            predictions = st.session_state.classifier.predict(non_system_messages)

                            # Create full predictions array
                            full_predictions = []
                            pred_idx = 0
                            for idx, row in df.iterrows():
                                if row['user'] == 'group_notification':
                                    full_predictions.append("System Notification")
                                else:
                                    full_predictions.append(predictions[pred_idx])
                                    pred_idx += 1

                            df['category'] = full_predictions
                            st.session_state.df_categorized = df.copy()
                            st.session_state.categorized = True

                        st.success("Messages categorized successfully!")
                    except Exception as e:
                        st.error(f"Categorization failed: {str(e)}")

            if "categorized" not in st.session_state:
                st.session_state.categorized = False

            if st.session_state.categorized:
                df_categorized = st.session_state.df_categorized

                # Initialize session state for filters (only once)
                if 'category_filter' not in st.session_state:
                    st.session_state.category_filter = ["Important", "Personal", "Spam", "Media", "System Notification"]
                if 'recency_filter' not in st.session_state:
                    st.session_state.recency_filter = "All time"

                # Filter UI (inside expander to avoid unwanted refreshes)
                with st.expander("Filter Options", expanded=True):
                    category_filter = st.multiselect(
                        label="Categories to display",
                        options=["Important", "Personal", "Spam", "Media", "System Notification"],
                        default=st.session_state.category_filter,
                        key='category_filter'
                    )

                    recency_options = {
                        "All time": None,
                        "Last 7 days": 7,
                        "Last 30 days": 30,
                        "Last 90 days": 90
                    }

                    selected_recency = st.selectbox(
                        label="Time period",
                        options=list(recency_options.keys()),
                        index=list(recency_options.keys()).index(st.session_state.recency_filter),
                        key='recency_filter'
                    )
                    if 'category' in labeled_data.columns and 'category' in df.columns:
                        y_true = labeled_data['category']
                        y_pred = df['category'][:len(y_true)]  # Ensure same length

                        # Generate classification report as a dictionary
                        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

                        # Convert to DataFrame and format
                        df_report = pd.DataFrame(report).transpose()
                        df_report[['precision', 'recall', 'f1-score']] = df_report[
                            ['precision', 'recall', 'f1-score']].round(3)
                        df_report['support'] = df_report['support'].astype(int)

                        # Select only desired rows (categories + weighted avg)
                        categories = list(labeled_data['category'].unique())
                        categories.append('weighted avg')
                        df_display = df_report.loc[categories, ['precision', 'recall', 'f1-score', 'support']]
                        df_display.index.name = 'Category'

                        # Print as a markdown-style table in the log/console
                        print("\nClassification Metrics Table:")
                        print(tabulate(df_display.reset_index(), headers='keys', tablefmt='github', showindex=False))


                    def reset_filters():
                        st.session_state.category_filter = ["Important", "Personal", "Spam", "Media", "System Notification"]
                        st.session_state.recency_filter = "All time"


                    col1, col2 = st.columns(2)
                    with col1:
                        apply_filters_btn = st.button("Apply Filters")
                    with col2:
                        reset_filters_btn = st.button("Reset Filters", on_click=reset_filters)

                filtered_df = df_categorized.copy()

                # Apply filters when button clicked
                if apply_filters_btn:
                    if category_filter:
                        filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
                    days_selected = recency_options[selected_recency]
                    if days_selected is not None:
                        cutoff_date = datetime.now() - timedelta(days=days_selected)
                        filtered_df = filtered_df[filtered_df['date'] >= cutoff_date]
                    # Save filtered dataframe in session state to persist across reruns
                    st.session_state.filtered_df_final = filtered_df.copy()

                # Display filtered or original categorized data based on session state
                display_df = (st.session_state.filtered_df_final
                              if 'filtered_df_final' in st.session_state
                              else df_categorized)

                if display_df.empty:
                    st.warning("No messages match the selected filters.")
                else:
                    st.subheader(f"Showing {len(display_df)} messages")
                    st.dataframe(display_df[['date', 'user', 'message', 'category']])

                    csv_download_data = display_df.to_csv(index=False).encode('utf-8')
                    filename_csv_download = ("filtered_chat.csv"
                                             if 'filtered_df_final' in st.session_state
                                             else "categorized_chat.csv")
                    # Download button for displayed dataframe
                    st.download_button(
                        label="Download Displayed Messages as CSV",
                        data=csv_download_data,
                        file_name=filename_csv_download,
                        mime="text/csv"
                    )

                # Category distribution visualization
                category_counts = display_df['category'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
                ax.set_title('Category Distribution')
                st.pyplot(fig)

            # Add reminder button to sidebar
            if st.sidebar.button("Reminders"):
                st.session_state.show_reminders = not st.session_state.show_reminders
                st.rerun()

            # Check if reminders should be shown (independent of Show Analysis)
            if st.session_state.show_reminders:
                # Display reminders UI
                st.title("WhatsApp Reminders")

                if 'reminder_system' not in st.session_state:
                    st.session_state.reminder_system = WhatsAppReminderSystem()

                if st.button("Scan Chat for Reminders"):
                    with st.spinner("Scanning chat for potential reminders..."):
                        reminders = st.session_state.reminder_system.process_chat(df)
                        st.session_state.extracted_reminders = reminders

                        if reminders:
                            st.success(f"Found {len(reminders)} potential reminders!")
                        else:
                            st.info("No reminders found in the chat.")

                if 'extracted_reminders' in st.session_state and st.session_state.extracted_reminders:
                    st.subheader("Extracted Reminders")

                    # Display reminders in a table with confidence scores
                    reminder_data = []
                    for i, reminder in enumerate(st.session_state.extracted_reminders):
                        reminder_data.append({
                            "ID": i + 1,
                            "Title": reminder['title'],
                            "Date & Time": reminder['start_time'].strftime("%Y-%m-%d %I:%M %p"),
                            "Context": reminder['tag'],
                            "From User": reminder['user'],
                            "Confidence": f"{reminder.get('confidence', 'N/A')}%"
                        })
                    reminder_df = pd.DataFrame(reminder_data)
                    st.dataframe(reminder_df)

                    # Add options to select which reminders to add
                    selected_reminders = st.multiselect(
                        "Select reminders to add to calendar",
                        options=list(range(1, len(st.session_state.extracted_reminders) + 1)),
                        default=list(range(1, len(st.session_state.extracted_reminders) + 1)),
                        format_func=lambda x: f"Reminder {x}: {st.session_state.extracted_reminders[x - 1]['title']}"
                    )

                    if st.button("Add Selected Reminders to Google Calendar"):
                        if not selected_reminders:
                            st.warning("No reminders selected.")
                        else:
                            with st.spinner("Adding reminders to Google Calendar..."):
                                # Filter selected reminders
                                reminders_to_add = [st.session_state.extracted_reminders[i - 1] for i in
                                                    selected_reminders]

                                # Add to calendar
                                results = st.session_state.reminder_system.add_reminders_to_calendar(reminders_to_add)

                                # Show results
                                success_count = sum(1 for r in results if r['success'])
                                st.success(
                                    f"Successfully added {success_count} out of {len(results)} reminders to Google Calendar!")

                                # Display detailed results
                                for result in results:
                                    if result['success']:
                                        st.markdown(f"✅ **{result['title']}**: {result['message']}")
                                    else:
                                        st.markdown(f"❌ **{result['title']}**: {result['message']}")

            # Robot/Chatbot Toggle Functionality
            # Robot/Chatbot Toggle Functionality
            robot_container = st.container()
            with robot_container:
                if not st.session_state.show_chatbot:
                    st.markdown("""
                        <div class="robot-label">
                            <span>Hi👋, I'm Whatssie!</span>
                        </div>
                        """, unsafe_allow_html=True)
                    if st.button("🤖", key="robot_button", help="Click to chat with Whatssie"):
                        st.session_state.show_chatbot = True
                        st.rerun()

            # Enhanced floating robot button styling
            if not st.session_state.show_chatbot:
                robot_css = """
                    position: fixed;
                    bottom: 80px;
                    right: 20px;
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    background: linear-gradient(145deg, #546b94, #425678);
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    cursor: pointer;
                    z-index: 1000;
                    transition: all 0.3s ease;
                    animation: pulse 2s infinite;

                    /* Add pulse animation */
                    @keyframes pulse {
                        0% { transform: scale(1); }
                        50% { transform: scale(1.05); }
                        100% { transform: scale(1); }
                    }

                    /* Add hover effect */
                    &:hover {
                        transform: translateY(-3px);
                        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
                    }
                """
                robot_container.float(robot_css)

                # Add additional styles for the robot label
                st.markdown("""
                    <style>
                    .robot-label {
                        position: fixed;
                        bottom: 145px;
                        right: 20px;
                        background-color: rgba(75, 93, 128, 0.9);
                        padding: 8px 12px;
                        border-radius: 15px;
                        z-index: 999;
                        text-align: center;
                        font-size: 14px;
                        color: white;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
                        animation: fadeIn 0.5s ease-in-out;
                    }

                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    </style>
                """, unsafe_allow_html=True)

            # Chatbot UI when activated
            if st.session_state.show_chatbot:
                chatbox_container = st.container()
                with chatbox_container:
                    # Header with controls in Streamlit
                    col1, col2, col3 = st.columns([7, 1, 1])

                    with col1:
                        st.markdown("""
                            <div class="chat-title">
                                <div class="chat-icon">🤖</div>
                                <div class="chat-name">Whatssie</div>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        if st.button("_", key="minimize_chat"):
                            # You could implement a minimize state here
                            pass

                    with col3:
                        if st.button("✖️", key="close_chat"):
                            st.session_state.show_chatbot = False
                            st.rerun()

                    # Initialize chatbot chain if data is processed
                    if not df.empty:
                        try:
                            # Convert DataFrame messages to text for embedding creation
                            chat_content = "\n".join(df['message'].astype(str).tolist())


                            @st.cache_resource
                            def process_chat_content(content):
                                try:
                                    # Use TextSplitter directly on the content
                                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
                                    texts = text_splitter.split_text(content)

                                    # Create embeddings
                                    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

                                    # Create vectorstore
                                    vectorstore = FAISS.from_texts(texts, embeddings)

                                    return vectorstore
                                except Exception as e:
                                    st.error(f"Error processing content: {str(e)}")
                                    return None


                            vectorstore = process_chat_content(chat_content)

                            if vectorstore:
                                # Success message with animation
                                st.markdown("""
                                    <div class="success-toast">
                                        <span>✓</span> Chat data ready!
                                    </div>
                                """, unsafe_allow_html=True)

                                retriever = vectorstore.as_retriever()
                                chain = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    chain_type='stuff',
                                    retriever=retriever,
                                    input_key='question'
                                )

                                # Improved message container with better scrolling
                                message_container = st.container(height=350)

                                # Add custom CSS for improved chat messages
                                st.markdown("""
                                    <style>
                                    .chat-message-container {
                                        display: flex;
                                        flex-direction: column;
                                        overflow-y: auto;
                                        height: 100%;
                                        padding-right: 10px;
                                        scroll-behavior: smooth;
                                    }

                                    /* Custom scrollbar */
                                    .chat-message-container::-webkit-scrollbar {
                                        width: 6px;
                                    }

                                    .chat-message-container::-webkit-scrollbar-track {
                                        background: rgba(255, 255, 255, 0.1);
                                        border-radius: 10px;
                                    }

                                    .chat-message-container::-webkit-scrollbar-thumb {
                                        background: rgba(255, 255, 255, 0.3);
                                        border-radius: 10px;
                                    }

                                    .chat-message-container::-webkit-scrollbar-thumb:hover {
                                        background: rgba(255, 255, 255, 0.5);
                                    }
                                    </style>
                                    """, unsafe_allow_html=True)

                                # Get user input from the fixed bottom position
                                input_placeholder = st.empty()
                                prompt = input_placeholder.chat_input('Ask a question about your WhatsApp chat')

                                # Display messages with improved styling
                                with message_container:
                                    st.markdown('<div class="chat-message-container">', unsafe_allow_html=True)

                                    # Show welcome message if this is the first interaction
                                    if len(st.session_state.messages) == 0:
                                        welcome_msg = "Welcome! I can help you analyze your WhatsApp chat. What would you like to know?"
                                        st.session_state.messages.append({'role': 'assistant', 'content': welcome_msg,
                                                                          'time': datetime.now().strftime("%H:%M")})

                                    for message in st.session_state.messages:
                                        msg_time = message.get('time', '')

                                        if message['role'] == 'user':
                                            st.markdown(f"""
                                                <div class="message user-message">
                                                    <div class="message-content">{message['content']}</div>
                                                    <div class="message-time">{msg_time}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"""
                                                <div class="message bot-message">
                                                    <div class="message-content">{message['content']}</div>
                                                    <div class="message-time">{msg_time}</div>
                                                </div>
                                                """, unsafe_allow_html=True)

                                    st.markdown('</div>', unsafe_allow_html=True)

                                if prompt:
                                    # Add timestamp to messages
                                    current_time = datetime.now().strftime("%H:%M")

                                    # Add user message to session state
                                    st.session_state.messages.append(
                                        {'role': 'user', 'content': prompt, 'time': current_time})

                                    # Show typing indicator
                                    with message_container:
                                        st.markdown("""
                                            <div class="typing-indicator">
                                                <span></span><span></span><span></span>
                                            </div>
                                            """, unsafe_allow_html=True)

                                    try:
                                        with st.spinner(''):
                                            # Generate response
                                            response = chain.run(prompt)
                                            # Add assistant response to session state
                                            st.session_state.messages.append({'role': 'assistant', 'content': response,
                                                                              'time': datetime.now().strftime("%H:%M")})
                                            # Rerun to update the UI with new messages
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Error generating response: {str(e)}")
                        except Exception as e:
                            st.error(f"Error initializing chatbot: {str(e)}")
                    else:
                        # Better empty state message
                        st.markdown("""
                            <div class="empty-state">
                                <div class="empty-icon">📁</div>
                                <h3>No Chat Data Available</h3>
                                <p>Please upload a WhatsApp chat export file to begin.</p>
                                <div class="upload-hint">Tap to upload</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Improved chatbox styling with glassmorphism effect
                chatbox_css = """
                    position: fixed;
                    bottom: 60px;
                    right: 20px;
                    width: 400px;
                    height: 500px;
                    background: rgba(15, 15, 20, 0.9);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 14px;
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
                    z-index: 999;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                    color: white;
                    animation: slideIn 0.3s ease-out;

                    /* Add header style */
                    &:before {
                        content: "";
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        height: 50px;
                        background: rgba(30, 30, 40, 0.7);
                        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 14px 14px 0 0;
                        z-index: -1;
                    }

                    @keyframes slideIn {
                        from { opacity: 0; transform: translateY(20px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                """
                chatbox_container.float(chatbox_css)

                # Add comprehensive styles for all UI elements
                st.markdown("""
                    <style>
                    /* Header styles */
                    .stColumns [data-testid="column"] {
                        padding: 15px 0 5px 0 !important;
                    }

                    /* Remove default button styling */
                    .stColumns [data-testid="column"] [data-testid="stButton"] {
                        text-align: center !important;
                    }

                    .stColumns [data-testid="column"] button {
                        background: transparent !important;
                        border: none !important;
                        color: rgba(255, 255, 255, 0.7) !important;
                        transition: color 0.2s !important;
                        padding: 2px 8px !important;
                        border-radius: 4px !important;
                        font-weight: normal !important;
                    }

                    .stColumns [data-testid="column"] button:hover {
                        color: white !important;
                        background: rgba(255, 255, 255, 0.1) !important;
                    }

                    /* Title styling */
                    .chat-title {
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        padding-left: 15px;
                    }

                    .chat-icon {
                        font-size: 18px;
                    }

                    .chat-name {
                        font-weight: 600;
                        font-size: 16px;
                        letter-spacing: 0.3px;
                    }

                    /* Message styles */
                    .message {
                        margin: 8px 0;
                        max-width: 85%;
                        display: flex;
                        flex-direction: column;
                        position: relative;
                        animation: messageIn 0.3s ease-out;
                    }

                    @keyframes messageIn {
                        from { opacity: 0; transform: translateY(10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }

                    .user-message {
                        align-self: flex-end;
                        margin-left: auto;
                    }

                    .bot-message {
                        align-self: flex-start;
                        margin-right: auto;
                    }

                    .message-content {
                        padding: 10px 14px;
                        border-radius: 16px;
                        font-size: 14px;
                        line-height: 1.4;
                    }

                    .user-message .message-content {
                        background: linear-gradient(135deg, #2b5d80 0%, #1e4060 100%);
                        border-bottom-right-radius: 4px;
                        color: white;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    }

                    .bot-message .message-content {
                        background: linear-gradient(135deg, #333333 0%, #222222 100%);
                        border-bottom-left-radius: 4px;
                        color: #f0f0f0;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    }

                    .message-time {
                        font-size: 10px;
                        color: rgba(255, 255, 255, 0.5);
                        margin-top: 4px;
                        align-self: flex-end;
                        padding: 0 5px;
                    }

                    /* Typing indicator */
                    .typing-indicator {
                        display: inline-flex;
                        align-items: center;
                        background: rgba(50, 50, 60, 0.7);
                        border-radius: 14px;
                        padding: 10px 15px;
                        margin: 8px 0;
                        animation: fadeIn 0.5s;
                    }

                    .typing-indicator span {
                        height: 8px;
                        width: 8px;
                        background: rgba(255, 255, 255, 0.7);
                        border-radius: 50%;
                        display: inline-block;
                        margin: 0 2px;
                        animation: typing 1.2s infinite;
                    }

                    .typing-indicator span:nth-child(2) {
                        animation-delay: 0.2s;
                    }

                    .typing-indicator span:nth-child(3) {
                        animation-delay: 0.4s;
                    }

                    @keyframes typing {
                        0%, 100% { transform: translateY(0); }
                        50% { transform: translateY(-5px); }
                    }

                    /* Success toast */
                    .success-toast {
                        position: absolute;
                        top: 50px;
                        left: 50%;
                        transform: translateX(-50%);
                        background: rgba(46, 125, 50, 0.9);
                        color: white;
                        padding: 8px 16px;
                        border-radius: 20px;
                        font-size: 13px;
                        animation: toastIn 0.5s, toastOut 0.5s 3s forwards;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }

                    .success-toast span {
                        display: inline-flex;
                        justify-content: center;
                        align-items: center;
                        width: 20px;
                        height: 20px;
                        background: rgba(255, 255, 255, 0.2);
                        border-radius: 50%;
                    }

                    @keyframes toastIn {
                        from { opacity: 0; transform: translate(-50%, -20px); }
                        to { opacity: 1; transform: translate(-50%, 0); }
                    }

                    @keyframes toastOut {
                        from { opacity: 1; transform: translate(-50%, 0); }
                        to { opacity: 0; transform: translate(-50%, -20px); }
                    }

                    /* Empty state */
                    .empty-state {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 280px;
                        text-align: center;
                        color: rgba(255, 255, 255, 0.7);
                        padding: 20px;
                    }

                    .empty-icon {
                        font-size: 40px;
                        margin-bottom: 15px;
                        opacity: 0.7;
                    }

                    .empty-state h3 {
                        margin: 0 0 8px 0;
                        font-weight: 500;
                    }

                    .empty-state p {
                        margin: 0 0 20px 0;
                        font-size: 14px;
                        opacity: 0.7;
                    }

                    .upload-hint {
                        background: rgba(255, 255, 255, 0.1);
                        padding: 8px 16px;
                        border-radius: 20px;
                        font-size: 12px;
                        animation: pulse 2s infinite;
                    }

                    /* Chat input styling */
                    .stChatInputContainer {
                        position: absolute !important;
                        bottom: 15px !important;
                        left: 15px !important;
                        right: 15px !important;
                        z-index: 1000 !important;
                        background-color: transparent !important;
                    }

                    .stChatInputContainer > div {
                        background-color: rgba(50, 50, 60, 0.7) !important;
                        border: 1px solid rgba(255, 255, 255, 0.1) !important;
                        border-radius: 20px !important;
                        padding: 2px 5px !important;
                    }

                    [data-testid="stChatInput"] {
                        color: white !important;
                        caret-color: white !important;
                        font-size: 14px !important;
                    }

                    /* Mobile responsiveness */
                    @media (max-width: 768px) {
                        .chatbox-container {
                            width: 90vw !important;
                            height: 70vh !important;
                            bottom: 20px !important;
                            right: 5vw !important;
                        }

                        .robot-label {
                            right: 5vw !important;
                        }

                        /* Adjust input position when keyboard appears on mobile */
                        @media (max-height: 400px) {
                            .stChatInputContainer {
                                bottom: 5px !important;
                            }
                        }
                    }
                    </style>
                    """, unsafe_allow_html=True)