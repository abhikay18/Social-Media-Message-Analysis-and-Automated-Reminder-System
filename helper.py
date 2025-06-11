from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression
import emoji
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import numpy as np

extract = URLExtract()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words, excluding media messages
    words = []
    for message in df['message']:
        # Skip counting words if the message is a media message
        if not (message.lower().strip() == '<media omitted>' or
                'file attached' in message.lower() or
                message.lower().strip() == 'null' or
                'this message was deleted' in message.lower()):
            words.extend(message.split())

    # fetch number of media messages (including '<Media omitted>')
    num_media_messages = df[
        (df['message'].str.contains('(file attached)', na=False, case=False)) |
        (df['message'].str.contains('<media omitted>', na=False, case=False))
    ].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df

def create_wordcloud(selected_user, df):
    try:
        f = open('stop_hinglish.txt', 'r', encoding='utf-8')
        stop_words = f.read()
    except:
        stop_words = ""

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    # Exclude media messages and deleted messages from wordcloud
    temp = temp[~temp['message'].str.contains('(file attached)|<media omitted>|this message was deleted|null',
                                            na=False, case=False)]

    def remove_stop_words(message):
        words = []
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
        return " ".join(words)

    if temp.empty:
        return None

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    try:
        f = open('stop_hinglish.txt', 'r', encoding='utf-8')
        stop_words = f.read()
    except:
        stop_words = ""

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    # Exclude media messages and deleted messages
    temp = temp[~temp['message'].str.contains('(file attached)|<media omitted>|this message was deleted|null',
                                            na=False, case=False)]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    word_counts = Counter(words).most_common(20)
    most_common_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Exclude media messages and deleted messages
    temp = df[~df['message'].str.contains('(file attached)|<media omitted>|this message was deleted|null',
                                        na=False, case=False)]

    emojis = []
    for message in temp['message']:
        try:
            # Updated emoji detection method
            emojis.extend([c for c in message if emoji.is_emoji(c)])
        except:
            continue

    if len(emojis) > 0:
        emoji_dict = Counter(emojis)
        emoji_df = pd.DataFrame(list(emoji_dict.items()), columns=['Emoji', 'Count'])
        # Sort by count in descending order
        emoji_df = emoji_df.sort_values(by='Count', ascending=False)
        return emoji_df
    return pd.DataFrame(columns=['Emoji', 'Count'])

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period',
                                 values='message', aggfunc='count').fillna(0)

    return user_heatmap


def forecast_message_trends(df, algorithm='Linear Regression'):
    """Enhanced version with existing data structure integration"""
    try:
        # Ensure only_date is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['only_date']):
            df['only_date'] = pd.to_datetime(df['only_date'], errors='coerce')

        # Use preprocessed date column from existing data
        message_counts = df.groupby('only_date').size().reset_index(name='count')
        message_counts = message_counts.set_index('only_date').asfreq('D').fillna(0).reset_index()

        if algorithm == 'Linear Regression':
            message_counts['date_numeric'] = (message_counts['only_date'] -
                                              message_counts['only_date'].min()).dt.days
            X = message_counts[['date_numeric']]
            y = message_counts['count']
            model = LinearRegression().fit(X, y)
            return message_counts, model

        elif algorithm == 'ARIMA':
            # Auto ARIMA implementation
            model = ARIMA(message_counts.set_index('only_date')['count'], order=(1, 1, 1)).fit()
            return message_counts, model

        elif algorithm == 'Prophet':
            prophet_df = message_counts.rename(columns={'only_date': 'ds', 'count': 'y'})
            model = Prophet().fit(prophet_df)
            return message_counts, model

        elif algorithm == 'Exponential Smoothing':
            # Prepare data for ETS
            ts_data = message_counts.set_index('only_date')['count']
            # Fit ETS model (Holt-Winters exponential smoothing)
            model = ExponentialSmoothing(
                ts_data,
                trend='add',
                seasonal='add',
                seasonal_periods=7  # Weekly seasonality
            ).fit()
            return message_counts, model


        elif algorithm == 'LSTM':

            # Prepare data for LSTM

            message_counts['date_numeric'] = (message_counts['only_date'] -

                                              message_counts['only_date'].min()).dt.days

            # Normalize data

            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()

            scaled_data = scaler.fit_transform(message_counts['count'].values.reshape(-1, 1))

            # Create sequences

            look_back = 7  # Use one week to predict

            X, y = [], []

            for i in range(len(scaled_data) - look_back):
                X.append(scaled_data[i:i + look_back, 0])

                y.append(scaled_data[i + look_back, 0])

            X = np.array(X).reshape(-1, look_back, 1)

            y = np.array(y)

            # Build LSTM model

            model = Sequential()

            model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))

            model.add(LSTM(50))

            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train model

            model.fit(X, y, epochs=50, batch_size=32, verbose=0)

            # Store additional data in a dictionary to be used for prediction

            model = {

                'lstm_model': model,

                'scaler': scaler,

                'look_back': look_back,

                'last_sequence': scaled_data[-look_back:].reshape(1, look_back, 1)

            }

            return message_counts, model


        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    except Exception as e:
        raise ValueError(f"Forecasting error: {str(e)}")


def analyze_sentiment(df, selected_user):
    """
    Analyze sentiment of messages for selected user
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Exclude media messages and deleted messages
    df = df[~df['message'].str.contains('(file attached)|(this message was deleted)|null',
                                        na=False, case=False)]

    if len(df) == 0:
        return pd.DataFrame()

    # Initialize NLTK sentiment analyzer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # Function to get sentiment scores
    def get_sentiment_scores(text):
        try:
            if not isinstance(text, str):
                return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

            scores = analyzer.polarity_scores(text)
            return scores
        except:
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

    # Apply sentiment analysis to each message
    df['sentiment_scores'] = df['message'].apply(get_sentiment_scores)

    # Extract individual sentiment components
    df['negative'] = df['sentiment_scores'].apply(lambda score: score['neg'])
    df['neutral'] = df['sentiment_scores'].apply(lambda score: score['neu'])
    df['positive'] = df['sentiment_scores'].apply(lambda score: score['pos'])
    df['compound'] = df['sentiment_scores'].apply(lambda score: score['compound'])

    # Categorize sentiment based on compound score
    def categorize_sentiment(compound):
        if compound > 0.05:
            return 'Positive'
        elif compound < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['sentiment_category'] = df['compound'].apply(categorize_sentiment)

    # Check if 'only_date' column exists for time series analysis
    if 'only_date' not in df.columns and 'date' in df.columns:
        df['only_date'] = df['date'].dt.date

    return df

def evaluate_forecasting_algorithms(df, test_size=30):
    """
    Compare forecasting algorithms on the same train/test split.
    Returns a DataFrame with performance metrics for each algorithm.
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['only_date']):
        df['only_date'] = pd.to_datetime(df['only_date'], errors='coerce')

    # Aggregate daily message counts
    message_counts = df.groupby('only_date').size().reset_index(name='count')
    message_counts = message_counts.set_index('only_date').asfreq('D').fillna(0).reset_index()

    # Train/test split: last `test_size` days as test
    train = message_counts[:-test_size]
    test = message_counts[-test_size:]

    algorithms = ["Linear Regression", "ARIMA", "Prophet", "Exponential Smoothing", "LSTM"]
    results = []

    for algo in algorithms:
        try:
            # Fit model on train set
            train_df = train.copy()
            test_df = test.copy()

            # Use your existing forecast_message_trends logic
            data_for_model = pd.concat([train_df, test_df], ignore_index=True)
            message_counts_model, model = forecast_message_trends(data_for_model, algorithm=algo)

            # Generate predictions for test period
            if algo == "Linear Regression":
                message_counts_model['date_numeric'] = (message_counts_model['only_date'] -
                                                        message_counts_model['only_date'].min()).dt.days
                last_numeric_day = message_counts_model['date_numeric'].iloc[len(train_df)-1]
                future_numeric_days = np.array([(last_numeric_day + i + 1) for i in range(test_size)]).reshape(-1, 1)
                preds = model.predict(future_numeric_days)
            elif algo == "ARIMA":
                preds = model.forecast(steps=test_size).values
            elif algo == "Prophet":
                prophet_df = message_counts_model.rename(columns={'only_date': 'ds', 'count': 'y'})
                model = Prophet().fit(prophet_df.iloc[:len(train_df)])
                future = model.make_future_dataframe(periods=test_size)
                forecast = model.predict(future)
                preds = forecast['yhat'][-test_size:].values
            elif algo == "Exponential Smoothing":
                preds = model.forecast(steps=test_size).values
            elif algo == "LSTM":
                lstm_model = model['lstm_model']
                scaler = model['scaler']
                look_back = model['look_back']
                last_sequence = model['last_sequence'].copy()
                preds_scaled = []
                current_sequence = last_sequence.copy()
                for _ in range(test_size):
                    next_pred = lstm_model.predict(current_sequence, verbose=0)[0][0]
                    preds_scaled.append(next_pred)
                    current_sequence = current_sequence.copy()
                    current_sequence[:, :-1, :] = current_sequence[:, 1:, :]
                    current_sequence[:, -1, 0] = next_pred
                preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
            else:
                preds = np.zeros(test_size)

            # Calculate metrics
            y_true = test_df['count'].values
            y_pred = preds[:test_size]
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # Avoid division by zero

            results.append({
                "Algorithm": algo,
                "MAE": round(mae, 2),
                "MSE": round(mse, 2),
                "RMSE": round(rmse, 2),
                "MAPE (%)": round(mape, 2)
            })
        except Exception as e:
            results.append({
                "Algorithm": algo,
                "MAE": "Error",
                "MSE": "Error",
                "RMSE": "Error",
                "MAPE (%)": "Error"
            })
            print(f"Error evaluating {algo}: {e}")

    return pd.DataFrame(results)