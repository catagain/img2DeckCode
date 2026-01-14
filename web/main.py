import sys
import cv2
import numpy as np
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 1. 處理路徑，確保能匯入 src.recognizer
# 假設目錄結構：
# /project_root
#   /src/recognizer.py
#   /web/main.py
#   /web/index.html
#   /output/
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from src.recognizer import CardRecognizer

app = FastAPI(title="Yu-Gi-Oh Card Recognizer")

# 2. CORS 設定 (解決連線失敗的關鍵)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 掛載靜態檔案 (讓瀏覽器能存取 output 裡的 _full_grid.jpg)
output_path = project_root / "output"
output_path.mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_path)), name="output")

# 初始化辨識器
recognizer = CardRecognizer(confidence_threshold=90)

# 4. 首頁路由：直接提供 index.html 畫面
@app.get("/")
async def get_index():
    index_path = Path(__file__).parent / "index.html"
    if not index_path.exists():
        return HTMLResponse(content="<h1>index.html not found in web/ folder</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# 5. WebSocket 辨識核心
@app.websocket("/ws/recognize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # 接收圖片二進位數據
        bytes_data = await websocket.receive_bytes()
        
        # 將 bytes 轉為 OpenCV 格式
        nparr = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            await websocket.send_json({"type": "error", "message": "無法解析圖片"})
            return

        # 定義進度回報的回呼函數 (給 Recognizer 使用)
        async def send_progress(current, total):
            await websocket.send_json({
                "type": "progress",
                "current": current,
                "total": total,
                "percent": int((current / total) * 100)
            })

        # 呼叫非同步辨識邏輯
        # 這裡會回傳 JSON 結果，以及相對於 output 資料夾的圖片路徑
        final_json, relative_grid_path = await recognizer.process_image_async(
            image, 
            user_id="web_user", 
            progress_callback=send_progress
        )

        # 回傳最終結果給前端
        await websocket.send_json({
            "type": "final",
            "data": final_json,
            "image_url": f"/output/{relative_grid_path}" # 拼接成 URL
        })

    except WebSocketDisconnect:
        print("使用者中斷連線")
    except Exception as e:
        print(f"發生錯誤: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    # 也可以直接執行 python web/main.py
    uvicorn.run(app, host="127.0.0.1", port=8000)