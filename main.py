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
    # ================== THÊM CHỨC NĂNG TÓM TẮT VĂN BẢN MỚI ==================
from fastapi import HTTPException

# Dùng lại 2 hàm bạn đã có sẵn trong code cũ
# (nếu tên hàm khác thì bạn đổi lại cho đúng nhé)
def textrank_summary(text: str, ratio: float = 0.3) -> str:
    # ← Copy nguyên hàm textrank_summary từ code cũ của bạn vào đây
    # (hàm có dùng networkx, nltk, v.v.)
    # Ví dụ: return summarize_with_textrank(text)
    pass  # ← bạn thay bằng hàm thật

def kmeans_summary(text: str, num_sentences: int = 4) -> str:
    # ← Copy nguyên hàm kmeans_summary từ code cũ vào đây
    pass  # ← bạn thay bằng hàm thật

@app.post("/summarize")
async def summarize_text(text: str):
    if not text or len(text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Văn bản quá ngắn, vui lòng dán ít nhất 100 ký tự!")
    
    try:
        textrank = textrank_summary(text)
        kmeans = kmeans_summary(text)
        return {
            "textrank_summary": textrank.strip(),
            "kmeans_summary": kmeans.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")
