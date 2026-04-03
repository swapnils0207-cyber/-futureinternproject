"""
Customer Support Ticket Classification System
=============================================
Tools: Python (NLTK / spaCy), Scikit-learn
Skills: Text preprocessing, NLP classification, priority logic, support analytics
"""

import re
import json
import random
import string
from collections import Counter

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET
# ─────────────────────────────────────────────

TICKETS = [
    # Billing
    ("I was charged twice for my subscription this month!", "billing", "high"),
    ("My invoice shows the wrong amount, please fix it.", "billing", "high"),
    ("Can I get a receipt for my last payment?", "billing", "low"),
    ("How do I update my credit card details?", "billing", "medium"),
    ("I want to cancel my subscription and get a refund.", "billing", "high"),
    ("What payment methods do you accept?", "billing", "low"),
    ("I haven't received my invoice for March.", "billing", "medium"),
    ("My payment failed even though my card is valid.", "billing", "high"),

    # Technical
    ("The app crashes every time I try to log in.", "technical", "high"),
    ("I cannot upload files larger than 10MB, getting an error.", "technical", "high"),
    ("The dashboard is loading very slowly today.", "technical", "medium"),
    ("Password reset email never arrives.", "technical", "high"),
    ("Two-factor authentication is broken, I can't sign in.", "technical", "high"),
    ("The export to PDF feature is not working.", "technical", "medium"),
    ("How do I integrate the API with my app?", "technical", "medium"),
    ("Getting a 500 error on the checkout page.", "technical", "high"),
    ("The mobile app is not syncing with the web version.", "technical", "medium"),
    ("Can you add dark mode to the app?", "technical", "low"),

    # Account
    ("I forgot my username, how do I recover it?", "account", "medium"),
    ("How do I delete my account permanently?", "account", "medium"),
    ("I want to change my email address.", "account", "medium"),
    ("My account was suspended without any notice!", "account", "high"),
    ("How do I add a team member to my account?", "account", "low"),
    ("Can I merge two accounts into one?", "account", "medium"),
    ("I cannot access my account, it says it's locked.", "account", "high"),
    ("How do I change my notification preferences?", "account", "low"),

    # General
    ("What are your support hours?", "general", "low"),
    ("Do you offer a free trial?", "general", "low"),
    ("I would like to leave a compliment for your support team.", "general", "low"),
    ("When will the new features be released?", "general", "low"),
    ("I'd like to speak to a manager.", "general", "medium"),
    ("Can I get a demo of your product?", "general", "low"),
]


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "not", "no", "so", "if", "as", "from", "by",
    "how", "what", "when", "where", "who", "which", "very", "just", "get",
    "also", "any", "all", "more", "about", "up", "out", "there",
}

def preprocess(text: str) -> list[str]:
    """Lowercase, remove punctuation, tokenize, remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION  (TF representation)
# ─────────────────────────────────────────────

def build_vocab(corpus: list[list[str]]) -> list[str]:
    all_words = [w for doc in corpus for w in doc]
    freq = Counter(all_words)
    # keep words that appear at least twice
    return sorted(w for w, c in freq.items() if c >= 1)

def vectorize(tokens: list[str], vocab: list[str]) -> list[int]:
    token_set = set(tokens)
    return [1 if w in token_set else 0 for w in vocab]


# ─────────────────────────────────────────────
# 4. NAIVE BAYES CLASSIFIER  (from scratch)
# ─────────────────────────────────────────────

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.classes = []

    def fit(self, X: list[list[int]], y: list[str], vocab: list[str]):
        self.classes = list(set(y))
        self.vocab = vocab
        n = len(y)
        class_counts = Counter(y)

        for cls in self.classes:
            # prior
            self.class_probs[cls] = class_counts[cls] / n
            # indices where class matches
            indices = [i for i, label in enumerate(y) if label == cls]
            # sum feature vectors
            word_counts = [sum(X[i][j] for i in indices) for j in range(len(vocab))]
            total = sum(word_counts) + len(vocab)  # Laplace smoothing
            self.word_probs[cls] = [(c + 1) / total for c in word_counts]

    def predict(self, x: list[int]) -> str:
        import math
        scores = {}
        for cls in self.classes:
            score = math.log(self.class_probs[cls])
            for j, val in enumerate(x):
                if val:
                    score += math.log(self.word_probs[cls][j])
            scores[cls] = score
        return max(scores, key=scores.get)

    def predict_proba(self, x: list[int]) -> dict[str, float]:
        import math
        scores = {}
        for cls in self.classes:
            score = math.log(self.class_probs[cls])
            for j, val in enumerate(x):
                if val:
                    score += math.log(self.word_probs[cls][j])
            scores[cls] = score
        # softmax
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: round(v / total, 3) for k, v in exp_scores.items()}


# ─────────────────────────────────────────────
# 5. PRIORITY LOGIC
# ─────────────────────────────────────────────

HIGH_PRIORITY_KEYWORDS = {
    "crash", "crashes", "error", "broken", "failed", "fail", "urgent", "suspended",
    "locked", "cannot", "charged", "twice", "refund", "500", "broken", "hack",
    "breach", "blocked", "critical", "immediately", "asap", "emergency"
}
MEDIUM_PRIORITY_KEYWORDS = {
    "slow", "delay", "wrong", "missing", "issue", "problem", "not working",
    "update", "change", "cancel", "invoice", "export", "sync", "integrate"
}

def assign_priority(tokens: list[str], predicted_category: str) -> str:
    token_set = set(tokens)
    if token_set & HIGH_PRIORITY_KEYWORDS:
        return "high"
    if token_set & MEDIUM_PRIORITY_KEYWORDS:
        return "medium"
    # category-based fallback
    if predicted_category in ("technical", "billing"):
        return "medium"
    return "low"


# ─────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────

texts, categories, priorities = zip(*TICKETS)
tokenized = [preprocess(t) for t in texts]
vocab = build_vocab(tokenized)
X = [vectorize(tok, vocab) for tok in tokenized]

# Split 80/20
random.seed(42)
indices = list(range(len(X)))
random.shuffle(indices)
split = int(0.8 * len(indices))
train_idx, test_idx = indices[:split], indices[split:]

X_train = [X[i] for i in train_idx]
y_train = [categories[i] for i in train_idx]
X_test  = [X[i] for i in test_idx]
y_test  = [categories[i] for i in test_idx]

clf = NaiveBayes()
clf.fit(X_train, y_train, vocab)


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    preds = [model.predict(x) for x in X_test]
    correct = sum(p == t for p, t in zip(preds, y_test))
    accuracy = correct / len(y_test)

    classes = sorted(set(y_test))
    report = {}
    for cls in classes:
        tp = sum(1 for p, t in zip(preds, y_test) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(preds, y_test) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(preds, y_test) if p != cls and t == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall    = tp / (tp + fn) if (tp + fn) else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        report[cls] = {"precision": round(precision, 2),
                       "recall": round(recall, 2),
                       "f1": round(f1, 2),
                       "support": sum(1 for t in y_test if t == cls)}
    return accuracy, report


# ─────────────────────────────────────────────
# 8. CLASSIFY NEW TICKET  (main API)
# ─────────────────────────────────────────────

def classify_ticket(text: str) -> dict:
    tokens = preprocess(text)
    vec    = vectorize(tokens, vocab)
    category = clf.predict(vec)
    confidence = clf.predict_proba(vec)
    priority = assign_priority(tokens, category)
    return {
        "text": text,
        "category": category,
        "priority": priority,
        "confidence": confidence,
        "tokens": tokens,
    }


# ─────────────────────────────────────────────
# 9. DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CUSTOMER SUPPORT TICKET CLASSIFICATION SYSTEM")
    print("=" * 60)

    accuracy, report = evaluate(clf, X_test, y_test)
    print(f"\n📊 Model Accuracy: {accuracy:.0%}\n")
    print(f"{'Category':<12} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>8}")
    print("-" * 48)
    for cls, m in report.items():
        print(f"{cls:<12} {m['precision']:>10.2f} {m['recall']:>8.2f} {m['f1']:>6.2f} {m['support']:>8}")

    print("\n" + "=" * 60)
    print("  SAMPLE PREDICTIONS")
    print("=" * 60)

    samples = [
        "My account was hacked and I can't log in!",
        "Can you tell me when the app will support dark mode?",
        "I was billed twice this month, this is unacceptable.",
        "The API integration keeps returning 500 errors.",
        "How do I update my billing address?",
    ]

    for s in samples:
        result = classify_ticket(s)
        print(f"\n🎫 Ticket  : {result['text']}")
        print(f"   Category: {result['category'].upper()}  |  Priority: {result['priority'].upper()}")
        top = max(result['confidence'], key=result['confidence'].get)
        print(f"   Confidence: {result['confidence'][top]:.0%} ({top})")
