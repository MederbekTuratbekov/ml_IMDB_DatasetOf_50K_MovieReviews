import torch
import uvicorn
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from torchtext.data import get_tokenizer


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = torch.load("vocab_IMDB_DatasetOf_50K_MovieReviews.pth", map_location=device, weights_only=False)

if "<unk>" in vocab:
    vocab.set_default_index(vocab["<unk>"])
else:
    vocab.set_default_index(0)

model = SentimentModel(len(vocab)).to(device)
model.load_state_dict(torch.load("model_IMDB_DatasetOf_50K_MovieReviews.pth", map_location=device))
model.eval()

# Настройка FastAPI
app = FastAPI()


class TextIn(BaseModel):
    text: str


tokenizer = get_tokenizer("basic_english")


def preprocess(text: str):
    tokens = tokenizer(text)

    if not tokens:
        return torch.tensor([[vocab["<unk>"]]], dtype=torch.int64, device=device)

    ids = [vocab[token] for token in tokens]
    return torch.tensor([ids], dtype=torch.int64, device=device)

@app.post("/predict")
def predict(item: TextIn):
    x = preprocess(item.text)
    with torch.no_grad():
        pred = model(x)
        label = torch.argmax(pred, dim=1).item()
    return {"label": "это позитивный отзыв" if label == 1 else "это negative отзыв"}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
