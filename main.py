from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Tải dữ liệu cần thiết cho nltk (chỉ lần đầu)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI(title="Tóm tắt Giao thông TP.HCM 2025 - True Summarizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đọc CSV (giữ nguyên chức năng cũ)
try:
    df = pd.read_csv("data_extractive.csv")
    df = df.fillna("")
except:
    df = pd.DataFrame()

@app.get("/")
def home():
    return {"message": "API True Summarizer đang chạy!", "total_articles": len(df)}

# Giữ nguyên chức năng tìm kiếm cũ
@app.get("/articles")
def get_articles(q: str = Query(None), limit: int = Query(20)):
    result = df.copy()
    if q and not result.empty:
        mask = (
            result['title'].str.contains(q, case=False, na=False) |
            result['content'].str.contains(q, case=False, na=False)
        )
        result = result[mask]
    return result.head(limit).to_dict(orient="records")

# ================== HÀM TÓM TẮT THẬT – CHẠY MỌI BÀI BÁO MỚI ==================
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def textrank_summary(text: str, sentences_count: int = 5) -> str:
    try:
        parser = PlaintextParser.from_string(clean_text(text), Tokenizer("vietnamese"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join([str(sent) for sent in summary])
    except:
        # Fallback tiếng Anh nếu lỗi
        parser = PlaintextParser.from_string(clean_text(text), Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join([str(sent) for sent in summary])

def kmeans_summary(text: str, num_sentences: int = 5) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) <= num_sentences:
        return ". ".join(sentences)
    
    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform(sentences)
        kmeans = KMeans(n_clusters=num_sentences, random_state=42).fit(X)
        summary = []
        for i in range(num_sentences):
            cluster_sents = [sentences[j] for j in range(len(sentences)) if kmeans.labels_[j] == i]
            if cluster_sents:
                summary.append(max(cluster_sents, key=len))
        return ". ".join(summary) + "."
    except:
        return ". ".join(sentences[:num_sentences])

# ENDPOINT MỚI – TÓM TẮT BẤT KỲ BÀI BÁO NÀO
@app.post("/summarize")
async def summarize_text(text: str):
    if not text or len(text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Văn bản quá ngắn! Vui lòng dán ít nhất 100 ký tự.")
    
    textrank = textrank_summary(text, 5)
    kmeans = kmeans_summary(text, 5)
    
    return {
        "textrank_summary": textrank,
        "kmeans_summary": kmeans,
        "message": "Tóm tắt thành công bài báo mới!"
    }
