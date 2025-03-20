import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Using device: {device}")

# ✅ Load dataset
print("📂 Loading dataset...")
df = pd.read_csv("classified_crime_data.csv")

# ✅ Ensure necessary columns exist
required_columns = ["crime_description", "predicted_category", "predicted_sub_category"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"❌ ERROR: Missing column in CSV: {col}")

# ✅ Handle missing values
df = df.dropna(subset=required_columns)
df["crime_description"] = df["crime_description"].astype(str)  # Convert to string

# ✅ Encode categories
category_encoder = LabelEncoder()
df["category_label"] = category_encoder.fit_transform(df["predicted_category"])

subcategory_encoder = LabelEncoder()
df["subcategory_label"] = subcategory_encoder.fit_transform(df["predicted_sub_category"])

# ✅ Save encoders
joblib.dump(category_encoder, "category_encoder.pkl")
joblib.dump(subcategory_encoder, "subcategory_encoder.pkl")

# ✅ Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ Create Dataset class
class ComplaintsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# ✅ Prepare training and validation data
train_texts, val_texts, train_labels, val_labels = train_test_split(df["crime_description"], df["category_label"], test_size=0.2, random_state=42)
train_dataset = ComplaintsDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = ComplaintsDataset(val_texts.tolist(), val_labels.tolist())

# ✅ Load BERT model (you can use a smaller model if necessary)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(category_encoder.classes_))
model.to(device)

# ✅ Training arguments with batch size reduced and gradient accumulation
training_args = TrainingArguments(
    output_dir="./bert_model",
    num_train_epochs=4,
    per_device_train_batch_size=1,  # Reduced batch size to 1
    per_device_eval_batch_size=1,   # Reduced batch size for evaluation
    gradient_accumulation_steps=16,  # Accumulate gradients over 16 steps to simulate larger batch size
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,  # Enables mixed precision training
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 🚀 Train model
print("🚀 Training BERT on", device, "...")
trainer.train()

# ✅ Save trained model
print("✅ Training Complete!")
model.save_pretrained("bert_category_model")
tokenizer.save_pretrained("bert_category_model")

# ✅ Load and classify new complaints
print("📂 Loading new complaints...")
df_new = pd.read_csv("new_complaints.csv")

# ✅ Tokenize new complaints
print("🔍 Tokenizing complaints...")
inputs = tokenizer(df_new["crime_description"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors to GPU

# ✅ Predict categories
print("🔍 Classifying complaints...")
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predicted_labels = torch.argmax(outputs.logits, dim=1).cpu().numpy()

# ✅ Convert back to category labels
df_new["predicted_category"] = category_encoder.inverse_transform(predicted_labels)

# ✅ Save classified complaints
df_new.to_csv("classified_new_complaints.csv", index=False)
print("✅ Classification Complete! Results saved to classified_new_complaints.csv")
