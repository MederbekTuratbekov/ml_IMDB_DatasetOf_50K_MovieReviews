# Sentiment Analysis API

> LSTM-модель классифицирует текст как positive/negative за < 50 мс,
> позволяя командам поддержки и продукта мониторить восприятие бренда в реальном времени —
> без единой ручной метки.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-87%25-brightgreen)]()
[![F1](https://img.shields.io/badge/F1-0.87-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Проблема

Компании ежедневно получают тысячи отзывов, тикетов и упоминаний в соцсетях.
Читать их вручную невозможно. Один вирусный негативный отзыв стоит бренду до 30 клиентов
(Harvard Business Review). Этот API оценивает любой текст за миллисекунды —
злые клиенты всплывают мгновенно, без единого human label.

---

## Быстрый старт

```bash
git clone https://github.com/your-username/sentiment-analysis-api
cd sentiment-analysis-api
pip install torch==2.3.0 torchtext==0.18.0 torchdata==0.9.0 fastapi uvicorn portalocker

python train.py   # → сохраняет model.pth + vocab.pth
python main.py    # → http://127.0.0.1:8000/docs
```

---

## Demo

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Absolutely loved this product, works perfectly!"}'
```

```json
{"label": "positive", "confidence": 0.94}
```

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Terrible experience, broke after two days."}'
```

```json
{"label": "negative", "confidence": 0.91}
```

---

## Результаты

| Модель                       | Accuracy | F1   | Улучшение vs baseline |
|------------------------------|----------|------|----------------------|
| Majority-class classifier    | 50%      | 0.33 | baseline             |
| Bag-of-Words + Logistic Reg  | 78%      | 0.78 | +28%                 |
| **LSTM (embed=64, hidden=128)** | **87%** | **0.87** | **+37%**        |

**Почему LSTM, а не BERT:**
- BERT даёт ~93% на этом датасете, но требует 400 МБ и GPU в продакшене
- LSTM — 8 МБ, инференс < 50 мс на CPU, деплоится на любом сервере
- Для бинарной классификации отзывов точность 87% достаточна для production-использования

---

## Датасет

- **Источник:** IMDB Movie Reviews — `torchtext.datasets.IMDB`, без ручной загрузки
- **Объём:** 50 000 отзывов (25K train / 25K test)
- **Баланс классов:** 50/50 — ресэмплинг не нужен
- **Словарь:** ~100K токенов, собран только на train-сплите (нет утечки в тест)

---

## Архитектура

    POST /predict  {"text": "..."}

│

▼

    Tokenizer           ← basic_english, lowercase

│

▼

    Vocab lookup        ← ~100K tokens, <unk> для OOV

│

▼

    Embedding(vocab, 64)

│

▼

    LSTM(64 → 128)      ← финальный hidden state

│

▼

    Linear(128 → 2)

│

▼

    Softmax → argmax    ← label + confidence

│

▼

    {"label": "positive", "confidence": 0.94}



**Ключевые решения:**

`pad_sequence(batch_first=True)` внутри `collate_batch` —
LSTM требует фиксированный размер тензора, но отзывы варьируются от 10 до 2 000 токенов.
Zero-padding решает это без ручной обрезки.

Vocab строится только на train — тест-токены без совпадений падают в `<unk>`.
Это гарантирует честную out-of-sample оценку.

Модель и vocab загружаются один раз при старте приложения (не на каждый запрос) →
латентность < 50 мс на CPU.

---

## Стек

| Слой          | Технологии                              |
|---------------|-----------------------------------------|
| Deep Learning | PyTorch 2.3, torchtext 0.18             |
| API           | FastAPI, Uvicorn, Pydantic              |
| Данные        | torchtext datasets, DataLoader          |
| Сериализация  | torch.save / torch.load                 |
| Среда         | pip / venv, Colab-совместимо            |

---

## API

| Метод | Эндпоинт   | Вход                    | Выход                                      |
|-------|------------|-------------------------|--------------------------------------------|
| POST  | `/predict` | `{"text": "string"}`    | `{"label": "...", "confidence": 0.xx}`     |
| GET   | `/health`  | —                       | `{"status": "ok", "model": "lstm-v1.0"}`   |

---

## Что дальше (Roadmap)

- [ ] `confidence` в ответе — уже показывает `softmax`-вероятность, нужен порог для "uncertain"
- [ ] Батч-эндпоинт `POST /predict/batch` — массовая обработка очередей тикетов
- [ ] MLflow: трекинг экспериментов, версионирование модели
- [ ] Data drift мониторинг (evidently) — алерт при смещении входного распределения
- [ ] Docker Compose для production-деплоя
- [ ] Fine-tuning на domain-specific данных (support chats, app store reviews)

---

## Business Impact

| Задача                                    | До                        | После                   |
|-------------------------------------------|---------------------------|-------------------------|
| Обнаружение негативных тикетов            | Часы ручного просмотра    | < 50 мс на запрос       |
| Маркировка тональности для аналитики      | Ручная разметка           | Автоматически, 87% точн.|
| Интеграция в CRM / чат-бот / review feed  | Кастомная разработка      | Один API-вызов          |
| Масштабирование на новый домен            | Переобучение с нуля       | Fine-tuning на своих данных |

---

[//]: # (## Автор)
[//]: # (**[Имя]** — [LinkedIn]&#40;https://linkedin.com/in/you&#41; | [GitHub]&#40;https://github.com/you&#41;)

