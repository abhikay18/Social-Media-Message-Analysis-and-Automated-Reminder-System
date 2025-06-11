# File: calendar_integration.py (updated)
from datetime import timedelta
from auth import GoogleAuth
import pytz


class GoogleCalendarIntegration:
    def __init__(self):
        self.auth = GoogleAuth()
        self.service = self.auth.get_calendar_service()
        self.timezone = pytz.timezone('Asia/Kolkata')

    def create_reminder(self, title, start_time, description=None, tag=None):
        """Create calendar event with timezone awareness"""
        if not self.service:
            return False, "Authentication failed"

        event = {
            'summary': title,
            'description': f"{tag} reminder\n{description}",
            'start': {
                'dateTime': start_time.astimezone(self.timezone).isoformat(),
                'timeZone': 'Asia/Kolkata'
            },
            'end': {
                'dateTime': (start_time + timedelta(hours=1)).isoformat(),
                'timeZone': 'Asia/Kolkata'
            },
            'reminders': {'useDefault': False, 'overrides': [{'method': 'popup', 'minutes': 30}]}
        }

        try:
            created_event = self.service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            return True, created_event.get('htmlLink')
        except Exception as e:
            return False, str(e)

    def add_reminders(self, reminders):
        """Batch add reminders to calendar"""
        results = []
        for reminder in reminders:
            success, message = self.create_reminder(
                title=reminder['title'],
                start_time=reminder['start_time'],
                description=reminder['original_message'],
                tag=reminder['tag']
            )
            results.append({
                'title': reminder['title'],
                'success': success,
                'message': message
            })
        return results
