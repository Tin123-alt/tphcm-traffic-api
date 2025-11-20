from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI(title="Tóm tắt Giao thông TP.HCM 2025")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đọc file CSV
try:
    df = pd.read_csv("data_extractive.csv")
    df = df.fillna("")
except FileNotFoundError:
    df = pd.DataFrame()  # Nếu không có file, tạo DataFrame rỗng

@app.get("/")
def home():
    return {"message": "API Tóm tắt Giao thông TP.HCM 2025 đang chạy!", "total_articles": len(df)}

@app.get("/articles")
def get_articles(q: str = Query(None, description="Từ khóa tìm kiếm"), limit: int = Query(20, description="Số bài tối đa")):
    result = df.copy()
    if q and not result.empty:
        mask = (
            result['title'].str.contains(q, case=False, na=False) |
            result['summary_semantic_textrank'].str.contains(q, case=False, na=False) |
            result['summary_kmean'].str.contains(q, case=False, na=False)
        )
        result = result[mask]
    return result.head(limit).to_dict(orient="records")
