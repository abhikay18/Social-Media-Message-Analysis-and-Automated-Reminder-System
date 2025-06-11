import re
import pandas as pd


def preprocess(data, output_csv_path="processed_chat.csv"):
    """
    Preprocess WhatsApp chat data and save it to a CSV file.

    Args:
        data (str): Raw WhatsApp chat data as a string.
        output_csv_path (str): Path to save the processed CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Pattern to match the WhatsApp message format
    pattern = '\d{2}/\d{2}/\d{2},\s\d{1,2}:\d{2}\s(?:am|pm)\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    if len(messages) == 0 or len(dates) == 0:
        print("No messages or dates found. Check if the input format matches the WhatsApp format.")
        return pd.DataFrame()

    # Create a DataFrame with messages and dates
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Clean and convert dates to datetime format
    def clean_date(date_string):
        try:
            return pd.to_datetime(date_string.strip(' -'), format='%d/%m/%y, %I:%M %p')
        except Exception as e:
            print(f"Error parsing date: {date_string} - {e}")
            return None

    df['date'] = df['message_date'].apply(clean_date)
    df.drop(columns=['message_date'], inplace=True)

    # Split user and message content
    users = []
    messages = []

    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # If user is found
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:  # If it's a system notification
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Add additional columns for analysis
    # In preprocessor.py
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Add time period (e.g., "23-00", "00-01")
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    # Save the processed DataFrame to a CSV file
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Processed data saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

    print(f"Processed {len(df)} messages.")

    return df
