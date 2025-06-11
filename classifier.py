import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib
import re
import emoji


class WhatsAppClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.max_length = 128

    def preprocess_message(self, message):
        """Enhanced preprocessing for WhatsApp messages"""
        message = str(message).lower()



        # Spam
        spam_patterns = [
            r"this message was deleted", r"null"
        ]
        if any(re.search(p, message) for p in spam_patterns):
            return "spam"

        # Media messages
        media_patterns = [
            r"<media omitted>"
        ]
        if any(re.search(p, message) for p in media_patterns):
            return "media_content"

        # Clean and normalize text
        message = re.sub(r'https?://\S+', '[URL]', message)
        message = re.sub(r'\b\d{10}\b', '[PHONE]', message)
        message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', message)
        message = emoji.demojize(message, delimiters=(" ", " "))
        message = re.sub(r'\s+', ' ', message).strip()

        return message

    def train(self, data_path, model_name="bert-base-uncased"):
        """Train the transformer model"""
        df = pd.read_csv(data_path)
        texts = df['message'].apply(self.preprocess_message).tolist()

        # Convert labels to int64 explicitly
        labels = self.label_encoder.fit_transform(df['category']).astype(np.int64)  # <-- FIX HERE

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_encoder.classes_)
        ).to(self.device)

        # Tokenize data
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Convert labels to LongTensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)  # <-- EXPLICIT CASTING

        # Create dataset with correct tensor types
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            labels_tensor  # <-- USE PROPERLY TYPED TENSOR
        )

        # Training configuration
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        # Training loop
        self.model.train()
        for epoch in range(3):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = [t.to(self.device) for t in batch]
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # Save model
        self.save("models/whatsapp_classifier")

    def predict(self, messages):
        """Make predictions"""
        if not self.model or not self.tokenizer:
            self.load("models/whatsapp_classifier")

        preprocessed = [self.preprocess_message(msg) for msg in messages]
        encodings = self.tokenizer(
            preprocessed,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(preds)

    def save(self, path):
        """Save model components"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        joblib.dump(self.label_encoder, f"{path}/label_encoder.pkl")

    def load(self, path):
        """Load model components"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.label_encoder = joblib.load(f"{path}/label_encoder.pkl")


# Example usage
if __name__ == "__main__":
    classifier = WhatsAppClassifier()
    classifier.train("labeled_data.csv")

    test_messages = [
        "Meeting tomorrow at 10am in conference room",
        "Hey, want to grab lunch?",
        "This message was deleted",
        "<media omitted>"
    ]

    print(classifier.predict(test_messages))
