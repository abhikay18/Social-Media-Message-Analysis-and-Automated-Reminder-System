# File: duckling_date_extractor.py

import requests
import json
from datetime import datetime
import pytz


class DucklingDateExtractor:

    def __init__(self, duckling_url="http://localhost:8000/parse"):
        """Initialize the Duckling date extractor"""
        self.duckling_url = duckling_url
        self.timezone = pytz.timezone('Asia/Kolkata')

    def extract_dates(self, text, reference_time=None):
        """Extract date and time information from text using Duckling"""
        if not text:
            return []

        # Set reference time to current time if not provided
        if reference_time is None:
            reference_time = datetime.now(self.timezone)

        # Format reference time for Duckling
        ref_time_str = reference_time.strftime("%Y-%m-%dT%H:%M:%S%z")

        # Prepare request data
        data = {
            'locale': 'en_US',
            'text': text,
            'dims': '["time"]',
            'reftime': ref_time_str
        }

        try:
            # Send request to Duckling service with timeout
            response = requests.post(self.duckling_url, data=data, timeout=5)
            response.raise_for_status()

            # Parse response
            results = response.json()

            # Extract and format date information
            dates = []
            for result in results:
                if result.get('dim') == 'time':
                    value = result.get('value', {})

                    # Handle different time value types
                    if 'values' in value:  # Interval or multiple values
                        for val in value['values']:
                            dates.append(self._format_date_result(val, result['body']))
                    else:  # Single value
                        dates.append(self._format_date_result(value, result['body']))

            return dates

        except Exception as e:
            print(f"Duckling date extraction error: {str(e)}")
            # Fall back to dateparser
            return self._fallback_date_extraction(text, reference_time)

    def _format_date_result(self, value, original_text):
        """Format Duckling date result into a standardized structure"""
        try:
            # Parse the ISO datetime string
            date_time = datetime.fromisoformat(value['value'].replace('Z', '+00:00'))

            # Convert to local timezone
            date_time = date_time.astimezone(self.timezone)

            # Extract date and time components
            date_str = date_time.strftime("%Y-%m-%d")
            time_str = date_time.strftime("%H:%M") if value.get('grain') in ['second', 'minute', 'hour'] else None

            return {
                "text": original_text,
                "date": date_str,
                "time": time_str,
                "grain": value.get('grain', 'day'),
                "value": value['value']
            }
        except Exception as e:
            print(f"Error formatting date result: {str(e)}")
            return {
                "text": original_text,
                "date": None,
                "time": None,
                "grain": "unknown",
                "value": None
            }

    def _fallback_date_extraction(self, text, reference_time):
        """Simple fallback method for date extraction when Duckling is unavailable"""
        try:
            from dateparser.search import search_dates

            # Try to find dates in the text
            found_dates = search_dates(text, languages=['en'], settings={
                'RELATIVE_BASE': reference_time,
                'PREFER_DATES_FROM': 'future'
            })

            if not found_dates:
                return []

            dates = []
            for date_text, date_obj in found_dates:
                # Make sure date is timezone aware
                if date_obj.tzinfo is None:
                    date_obj = self.timezone.localize(date_obj)

                # Format result similar to Duckling format
                dates.append({
                    "text": date_text,
                    "date": date_obj.strftime("%Y-%m-%d"),
                    "time": date_obj.strftime("%H:%M") if date_obj.hour != 0 or date_obj.minute != 0 else None,
                    "grain": "day",
                    "value": date_obj.isoformat()
                })

            return dates
        except Exception as e:
            print(f"Fallback date extraction error: {str(e)}")
            return []
