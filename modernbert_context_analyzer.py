# File: modernbert_context_analyzer.py

from transformers import AutoTokenizer, AutoModel
import torch
import json


class ModernBERTContextAnalyzer:
    def __init__(self):
        """Initialize the ModernBERT context analyzer"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "answerdotai/ModernBERT-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        # Define context classification prompt template
        self.context_prompt = """
        Analyze the following message and determine if it's related to work or personal matters.
        Also extract any event information that might indicate a future event.

        Message: {message}

        Provide a JSON response with context classification and event details.
        """

    def analyze_message(self, message):
        """Analyze message context using ModernBERT embeddings"""
        try:
            # Prepare input
            prompt = self.context_prompt.format(message=message)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use [CLS] token embedding

            # Classify context based on embeddings
            # This would typically use a fine-tuned classification head
            # For simplicity, we'll use a placeholder implementation
            work_score = self._compute_work_score(message, embeddings)

            # Determine context type based on score
            context_type = "Work" if work_score > 0.5 else "Personal"
            confidence = int(abs(work_score - 0.5) * 200)  # Convert to 0-100 scale

            # Extract event information
            # Note: This will be supplemented by Duckling for date extraction
            contains_event = self._detect_event_indicators(message)

            return {
                "context": context_type,
                "confidence": confidence,
                "contains_future_event": contains_event,
                "event_title": self._extract_event_title(message) if contains_event else "",
                "dates": []  # Dates will be filled by Duckling
            }

        except Exception as e:
            print(f"ModernBERT context analysis error: {str(e)}")
            return {
                "context": "Personal",  # Default to personal as safer option
                "confidence": 50,
                "contains_future_event": False,
                "event_title": "",
                "dates": []
            }

    def _compute_work_score(self, message, embeddings):
        """Compute work-related score based on message content and embeddings"""
        # This would typically use a fine-tuned classifier
        # For demonstration, we'll use a keyword-based approach
        work_keywords = ['meeting', 'deadline', 'project', 'report', 'client', 'presentation']
        personal_keywords = ['dinner', 'party', 'family', 'friends', 'movie', 'weekend']

        work_count = sum(1 for word in work_keywords if word.lower() in message.lower())
        personal_count = sum(1 for word in personal_keywords if word.lower() in message.lower())

        # Combine keyword count with embedding information
        # In a real implementation, you would use the embeddings with a trained classifier
        keyword_score = work_count / (work_count + personal_count + 0.1)  # Avoid division by zero

        return keyword_score

    def _detect_event_indicators(self, message):
        """Detect if message contains indicators of a future event"""
        event_indicators = ['remind', 'remember', 'don\'t forget', 'appointment', 'schedule', 'meeting', 'event']
        return any(indicator in message.lower() for indicator in event_indicators)

    def _extract_event_title(self, message):
        """Extract a potential event title from the message using more robust heuristics"""
        # Convert to lowercase for case-insensitive matching
        message_lower = message.lower()

        # Expanded list of event indicators with variations
        event_indicators = [
            'remind', 'reminder', 'remember', 'don\'t forget', 'appointment',
            'schedule', 'meeting', 'event', 'call', 'conference', 'due',
            'deadline', 'interview', 'reservation'
        ]

        # Check for indicators and extract relevant text
        for indicator in event_indicators:
            if indicator in message_lower:
                # Find the indicator position
                start_idx = message_lower.find(indicator)
                # Get text after the indicator
                after_indicator = message[start_idx + len(indicator):].strip()

                # Clean up the text - remove common connector words
                after_indicator = after_indicator.strip(' :.,-about')

                # Look for prepositions or transition words that often introduce the event title
                connector_words = [' to ', ' about ', ' for ', ' regarding ', ' re: ', ' that ', ' me ', ' us ']
                for connector in connector_words:
                    if connector.lower() in after_indicator.lower():
                        # Take text after the connector word
                        parts = after_indicator.split(connector.lower(), 1)
                        if len(parts) > 1 and parts[1].strip():
                            after_indicator = parts[1].strip()
                            break

                # Extract a reasonable title length (up to 10 words)
                words = after_indicator.split()
                # Stop at punctuation that might end a title
                title_words = []
                for word in words[:10]:  # Limit to first 10 words
                    title_words.append(word)
                    if word.endswith('.') or word.endswith('!') or word.endswith('?'):
                        break

                title = " ".join(title_words)

                # Clean up any trailing punctuation
                title = title.rstrip(' ,.!?:;')

                # If we have a reasonable title (at least 2 characters), return it
                if len(title) > 2:
                    return title

        # If no indicator was found or title extraction failed, use NLP-based approach
        # (For simplicity, we'll use improved fallback here)
        sentences = message.split('.')
        if sentences and len(sentences[0]) > 3:
            # Take the first sentence, but limit to 10 words
            words = sentences[0].split()
            return " ".join(words[:min(10, len(words))])

        # Last resort fallback
        words = message.split()
        return " ".join(words[:min(7, len(words))])
