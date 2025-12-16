import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# --------------------------------------------------
# Device
# --------------------------------------------------
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

# --------------------------------------------------
# Load RoBERTa ATE+Sentiment model
# --------------------------------------------------
ate_sent_pipeline = pipeline(
    task='ner',
    aggregation_strategy='simple',
    model='gauneg/roberta-base-absa-ate-sentiment'
)

# --------------------------------------------------
# Load DeBERTa ABSA model
# --------------------------------------------------
deberta_model_name = "yangheng/deberta-v3-base-absa-v1.1"
deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name).to(device)

# --------------------------------------------------
# Load and prepare data
# --------------------------------------------------
df = pd.read_csv("../web_scraping/asus_zenbook_reviews.csv")
df = df[df["review"].notna() & df["rating"].notna()].copy()

def map_rating_to_sentiment(rating):
    if rating in [1, 2]:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df["TrueSentiment"] = df["rating"].apply(map_rating_to_sentiment)

# --------------------------------------------------
# Run hybrid pipeline
# --------------------------------------------------
results = []
y_true = []
y_pred = []

for review, true_label in tqdm(zip(df["review"], df["TrueSentiment"]), total=len(df)):
    # RoBERTa: extract aspects and token-level sentiment
    try:
        output = ate_sent_pipeline(review)
    except Exception:
        output = []

    aspect_sentiments = []
    for item in output:
        aspect = item["word"].strip()
        sentiment = item["entity_group"].replace("B-", "")
        confidence = item["score"]
        aspect_sentiments.append((aspect, sentiment, confidence))

        results.append({
            "review": review,
            "true_sentiment": true_label,
            "aspect": aspect,
            "predicted_sentiment": sentiment,
            "confidence": confidence
        })

    # DeBERTa: review-level sentiment prediction
    inputs = deberta_tokenizer(
        review, "general",
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(device)

    with torch.inference_mode():
        outputs = deberta_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()
        pred_label = deberta_model.config.id2label[pred_id]

    y_true.append(true_label)
    y_pred.append(pred_label)

# --------------------------------------------------
# Save aspect-level results
# --------------------------------------------------
aspect_df = pd.DataFrame(results)
aspect_df.to_csv("asus_hybrid_aspect_sentiment.csv", index=False)
print("Saved aspect-level results to asus_hybrid_aspect_sentiment.csv")

# --------------------------------------------------
# Review-level Evaluation
# --------------------------------------------------
print("\n📊 Review-level Evaluation:")
print(classification_report(y_true, y_pred, digits=3))

acc = (sum([t == p for t, p in zip(y_true, y_pred)]) / len(y_true)) * 100
print(f"🔍 Accuracy: {acc:.2f}%")

cm = confusion_matrix(y_true, y_pred, labels=["Negative", "Neutral", "Positive"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Hybrid ABSA (RoBERTa + DeBERTa)")
plt.show()

# --------------------------------------------------
# Aspect Aggregation Summary
# --------------------------------------------------
print("\n Aggregating Aspect Sentiment...")

sentiment_map = {"pos": 1, "neg": -1, "neu": 0}
aspect_df["score"] = aspect_df["predicted_sentiment"].map(sentiment_map)

agg = aspect_df.groupby("aspect").agg(
    total_mentions=("aspect", "count"),
    positive=("predicted_sentiment", lambda x: (x == "pos").sum()),
    negative=("predicted_sentiment", lambda x: (x == "neg").sum()),
    net_score=("score", "sum")
).reset_index()

agg.to_csv("hybrid_aspect_summary.csv", index=False)
print(" Saved aspect sentiment summary to hybrid_aspect_summary.csv")

top_positive = agg.sort_values("net_score", ascending=False).head(5)
top_negative = agg.sort_values("net_score").head(5)

print("\n Top Positive Aspects:")
print(top_positive[["aspect", "positive", "negative", "net_score"]])

print("\n Top Negative Aspects:")
print(top_negative[["aspect", "positive", "negative", "net_score"]])

