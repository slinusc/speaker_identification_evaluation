import os
import sys
import torch
import librosa
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, Wav2Vec2Config
import math
from datasets import load_metric
from datetime import datetime


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Define the custom dataset class using pandas
class LocalAudioDataset(Dataset):
    def __init__(self, csv_file, processor, subset):
        self.processor = processor
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['subset'] == subset]
        self.speaker_ids = {label: idx for idx, label in enumerate(self.data['label'].unique())}
        self.data['label'] = self.data['label'].map(self.speaker_ids)
        
        print(f"Loaded {len(self.speaker_ids)} speakers: {self.speaker_ids}")
        print(f"Total files in {subset}: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['label']
        
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            audio = librosa.to_mono(audio)
            audio = self._pad_or_truncate(audio, max_length=16000)
            input_values = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_values.squeeze(0)
            return {"input_values": input_values, "labels": label}
        except Exception as e:
            print(f"Error loading {file_path}: {e}", file=sys.stderr)
            return self.__getitem__((idx + 1) % len(self))

    def _pad_or_truncate(self, audio, max_length):
        if len(audio) < max_length:
            pad_size = max_length - len(audio)
            audio = np.pad(audio, (0, pad_size), 'constant', constant_values=(0, 0))
        else:
            audio = audio[:max_length]
        return audio


# Paths to dataset CSV file
csv_file = 'dataset_large.csv'
train_dataset = LocalAudioDataset(csv_file, processor, 'train')
validate_dataset = LocalAudioDataset(csv_file, processor, 'validate')
test_dataset = LocalAudioDataset(csv_file, processor, 'test')

num_speakers = len(train_dataset.speaker_ids)
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-960h", num_labels=num_speakers)
model = Wav2Vec2ForSequenceClassification(config)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

def validate_labels(dataset):
    for item in dataset:
        label = item['labels']
        if label >= num_speakers or label < 0:
            print(f"Invalid label {label} for item: {item}")
            raise ValueError(f"Invalid label {label} found in dataset.")
    print("All labels are valid.")

validate_labels(train_dataset)
validate_labels(validate_dataset)
validate_labels(test_dataset)

batch_size = 8
steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
logging_steps = steps_per_epoch // 5
eval_steps = steps_per_epoch // 5

accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

log_dir = "/home/rag/experimental_trial/results/training_logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_log_100_epochs_5_layer{timestamp}.csv")
with open(log_file, "w") as f:
    f.write("Timestamp,Step,Training Loss,Validation Loss,Accuracy\n")

class SaveMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(log_file, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                step = state.global_step
                training_loss = logs.get("loss", "")
                validation_loss = logs.get("eval_loss", "")
                accuracy = logs.get("eval_accuracy", "")
                f.write(f"{timestamp},{step},{training_loss},{validation_loss},{accuracy}\n")

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=100, early_stopping_threshold=0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        metric = kwargs.get("metrics", {}).get("eval_loss")
        if metric is None:
            return
        
        if self.best_metric is None or metric < self.best_metric - self.early_stopping_threshold:
            self.best_metric = metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.early_stopping_patience:
            print(f"Early stopping at step {state.global_step}")
            control.should_training_stop = True

training_args = TrainingArguments(
    output_dir="./results",
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    evaluation_strategy="steps",
    num_train_epochs=100,
    save_steps=logging_steps,
    eval_steps=eval_steps,
    logging_steps=logging_steps,
    learning_rate=5e-6,
    save_total_limit=2,
    no_cuda=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,  # lower eval_loss is better
    save_strategy="steps"  # or "epoch" if you prefer to save every epoch
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    callbacks=[SaveMetricsCallback(), EarlyStoppingCallback()] # Entfernte EarlyStoppingCallback
)

trainer.train()

metrics = trainer.evaluate(test_dataset)

print(f"Test set evaluation metrics: {metrics}")
print("Training and evaluation completed successfully!")

best_model_dir = "./results/best_model_100_epochs_5_layer"
os.makedirs(best_model_dir, exist_ok=True)

trainer.save_model(best_model_dir)
processor.save_pretrained(best_model_dir)

print(f"Best model saved to {best_model_dir}")