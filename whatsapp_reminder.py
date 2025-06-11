# File: whatsapp_reminder.py (updated)

from datetime import datetime
import pytz
from modernbert_context_analyzer import ModernBERTContextAnalyzer
from duckling_date_extractor import DucklingDateExtractor
from calendar_integration import GoogleCalendarIntegration


class WhatsAppReminderSystem:
    def __init__(self):
        """Initialize with ModernBERT for context understanding and Duckling for date extraction"""
        self.calendar_integration = GoogleCalendarIntegration()
        self.context_analyzer = ModernBERTContextAnalyzer()
        self.date_extractor = DucklingDateExtractor(duckling_url="http://localhost:8000/parse")
        self.timezone = pytz.timezone('Asia/Kolkata')

    def process_chat(self, df):
        """Process chat messages to extract future reminders using ModernBERT and Duckling"""
        reminders = []
        now = datetime.now(self.timezone)  # Current time in the correct timezone

        for _, row in df.iterrows():
            message = row['message']
            user = row['user']

            # Convert message date to timezone-aware datetime
            message_date = row['date'].to_pydatetime()
            if message_date.tzinfo is None:
                message_date = self.timezone.localize(message_date)

            # Skip system messages or empty messages
            if user == 'group_notification' or not message or message.strip() == '':
                continue

            # Use ModernBERT to analyze message context
            context_analysis = self.context_analyzer.analyze_message(message)

            # Only process messages that might contain future events
            if context_analysis.get('contains_future_event', False):
                # Use Duckling to extract dates with message_date as reference time
                extracted_dates = self.date_extractor.extract_dates(message, message_date)

                # Update the context analysis with extracted dates
                context_analysis['dates'] = extracted_dates

                # Process each extracted date
                for date_info in extracted_dates:
                    # Parse the date string
                    date_str = date_info.get('date')
                    time_str = date_info.get('time')

                    if date_str:
                        # Create datetime object
                        if time_str:
                            date_obj = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                        else:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d")

                        # Make datetime timezone-aware
                        if date_obj.tzinfo is None:
                            date_obj = self.timezone.localize(date_obj)

                        # Only add if the date is still in the future
                        if date_obj > now:
                            title = context_analysis.get('event_title') or self._generate_title(message,
                                                                                                date_info.get('text',
                                                                                                              ''))

                            reminders.append({
                                'title': title,
                                'start_time': date_obj,
                                'tag': context_analysis.get('context', 'Personal'),
                                'user': user,
                                'original_message': message,
                                'confidence': context_analysis.get('confidence', 80)
                            })

        # Sort reminders by date
        reminders.sort(key=lambda x: x['start_time'])
        return reminders

    def _generate_title(self, message, date_text):
        """Generate a meaningful title for the reminder using NLP techniques"""
        # Let's use our ModernBERT for better context understanding
        context_analysis = self.context_analyzer.analyze_message(message)

        # If ModernBERT already extracted an event title with high confidence, use it
        if context_analysis.get('event_title') and context_analysis.get('confidence', 0) > 85:
            return context_analysis.get('event_title')

        # Remove the date text from the message
        clean_message = message.replace(date_text, "").strip()

        # Remove common WhatsApp message patterns like emojis, URLs, etc.
        import re
        clean_message = re.sub(r'http\S+', '', clean_message)  # Remove URLs
        clean_message = re.sub(r'[^\w\s]', ' ', clean_message)  # Replace punctuation with space
        clean_message = re.sub(r'\s+', ' ', clean_message).strip()  # Normalize whitespace

        # Use NLP to identify the main action/event
        # For a simple approach, look for action verbs + objects
        words = clean_message.split()

        # Look for common reminder patterns
        reminder_indicators = ['meeting', 'call', 'appointment', 'reminder', 'event',
                               'deadline', 'submit', 'pay', 'attend', 'remember']

        # Try to find segments with reminder words
        for indicator in reminder_indicators:
            if indicator in clean_message.lower():
                # Find the sentence or phrase containing this indicator
                pattern = r'[^.!?]*\b' + indicator + r'\b[^.!?]*[.!?]?'
                matches = re.findall(pattern, clean_message, re.IGNORECASE)
                if matches:
                    return matches[0].strip().capitalize()

        # If no pattern matched, use intelligent truncation
        if len(words) > 5:
            # Try to find a natural break point
            break_points = ['.', ',', ';', 'and', 'but', 'or']
            for i, word in enumerate(words[:10]):  # Look in first 10 words
                if any(bp in word for bp in break_points) and i >= 3:  # At least 3 words
                    return " ".join(words[:i + 1]).capitalize()

            # Default to first 6-8 words based on message length
            title_length = min(8, max(6, len(words) // 3))
            return " ".join(words[:title_length]).capitalize()
        else:
            title = clean_message

        # If title is too short or empty after cleaning, use a generic title
        if len(title) < 5:
            # Try to use context information
            context = context_analysis.get('context', '')
            if context and context != 'Personal':
                return f"{context} reminder for {date_text}"
            else:
                return f"Reminder for {date_text}"

        return title.capitalize()

    def add_reminders_to_calendar(self, reminders):
        """Add reminders to Google Calendar"""
        return self.calendar_integration.add_reminders(reminders)
