import os
import pickle
import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import socket


class GoogleAuth:
    def __init__(self):
        # Combined scopes for both authentication and calendar access
        self.SCOPES = [
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile',
            'https://www.googleapis.com/auth/calendar',
            'openid'
        ]
        self.creds = None
        self.user_info = None

    def authenticate(self):
        """Authenticate with Google."""
        # Check if token.json exists
        if os.path.exists('token.json'):
            with open('token.json', 'rb') as token:
                self.creds = pickle.load(token)

        # If credentials don't exist or are invalid, get new ones
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                # Set socket options to allow address reuse
                socket.socket.default_socket_options = [(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)]

                # Use InstalledAppFlow with explicit redirect URI
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json',
                    self.SCOPES,
                    redirect_uri='http://localhost:8501'  # Explicitly match what's in credentials.json
                )

                # Try with the registered port first
                try:
                    self.creds = flow.run_local_server(port=8501)
                except OSError:
                    # If port is in use, try an alternative approach
                    st.warning("Port 8501 is in use. Consider switching to Desktop App credentials.")
                    try:
                        # Try with a random port as a fallback
                        flow = InstalledAppFlow.from_client_secrets_file(
                            'credentials.json',
                            self.SCOPES
                        )
                        self.creds = flow.run_local_server(port=0)
                        st.warning(
                            "Authentication succeeded but used a different port. You may need to update your redirect URIs in Google Cloud Console.")
                    except Exception as e:
                        st.error(f"Authentication error: {e}")
                        return False
                except Exception as e:
                    st.error(f"Authentication error: {e}")
                    return False

            # Save credentials for next run
            with open('token.json', 'wb') as token:
                pickle.dump(self.creds, token)

        # Get user info
        try:
            service = build('oauth2', 'v2', credentials=self.creds)
            self.user_info = service.userinfo().get().execute()
            return True
        except Exception as e:
            st.error(f"Error getting user info: {e}")
            return False

    def get_calendar_service(self):
        """Get the Google Calendar service."""
        if not self.creds:
            if not self.authenticate():
                return None

        return build('calendar', 'v3', credentials=self.creds)

    def get_user_info(self):
        """Get user information."""
        if not self.user_info:
            if not self.authenticate():
                return None

        return self.user_info

    def logout(self):
        """Log out the user."""
        if os.path.exists('token.json'):
            os.remove('token.json')
        self.creds = None
        self.user_info = None
        return True
