from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import asyncio
import logging
import re
import httpx  # used to fecth drawing numbers (File name)
import uuid
from fpdf import FPDF
import requests
import tempfile
import json
import cloudinary
import cloudinary.uploader
#import google.generativeai as genai
from openai import OpenAI
import base64
from PIL import Image, ImageFilter  # used by sharpening step
from typing import List, Tuple
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
from googleapiclient.discovery import build
import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

print("🚀 Server has started and main.py is loaded")

# Configure CORS - explicitly allow your GitHub Pages d
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://veenu-wootz.github.io", "http://localhost:3000", "https://aniketsandhanwootz-wq.github.io"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

# Paths to pre-downloaded model files
PADDLE_HOME = os.path.expanduser("~/.paddleocr")
DET_MODEL_DIR = os.path.join(PADDLE_HOME, "whl/det/en/en_PP-OCRv3_det_infer")
REC_MODEL_DIR = os.path.join(PADDLE_HOME, "whl/rec/en/en_PP-OCRv3_rec_infer")
CLS_MODEL_DIR = os.path.join(PADDLE_HOME, "whl/cls/ch_ppocr_mobile_v2.0_cls_infer")

# Gemini API Configuration
#GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#if GEMINI_API_KEY:
#    genai.configure(api_key=GEMINI_API_KEY)
#    logger.info("✅ Gemini API configured")

#Open AI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- Google DocAI config (kept out of GitHub) ---
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
    _sa_path = "/tmp/gcp_sa.json"
    if not os.path.exists(_sa_path):
        with open(_sa_path, "w") as f:
            f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _sa_path

DOC_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DOC_LOCATION   = os.getenv("GCP_LOCATION", "us")
DOC_PROCESSOR  = os.getenv("DOC_PROCESSOR_ID")

# ---- Google Sheets config (service account) ----
SHEET_ID  = os.getenv("SHEET_ID")
SHEET_TAB = os.getenv("SHEET_TAB", "Logs")
_SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_sheets_service = None

def _get_sheets_service():
    """Create and cache a Sheets API client using the same SA file we wrote to /tmp."""
    global _sheets_service
    if _sheets_service is None:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not cred_path or not os.path.exists(cred_path):
            logger.warning("Sheets: no GOOGLE_APPLICATION_CREDENTIALS file found")
            return None
        creds = service_account.Credentials.from_service_account_file(
            cred_path, scopes=_SHEETS_SCOPES
        )
        # cache_discovery avoids filesystem writes on serverless
        _sheets_service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return _sheets_service

#else:
#    logger.warning("⚠️ GEMINI_API_KEY not found - vision OCR will be disabled")
# Check and log model paths on startup
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting OCR API with model paths:")
    logger.info(f"Detection model: {DET_MODEL_DIR} (exists: {os.path.exists(DET_MODEL_DIR)})")
    logger.info(f"Recognition model: {REC_MODEL_DIR} (exists: {os.path.exists(REC_MODEL_DIR)})")
    logger.info(f"Classification model: {CLS_MODEL_DIR} (exists: {os.path.exists(CLS_MODEL_DIR)})")
    logger.info(f"Sheets configured: {bool(SHEET_ID)}; DocAI configured: {bool(DOC_PROCESSOR)}")

# Code look for an “O” preceded by whitespace and followed by a digit and replaces it with “Ø”
def fix_diameter(text: str) -> str:
    # look for an “O” preceded by whitespace and followed by a digit,
    # and replace it with “Ø”
    return re.sub(r'(?<=\s)[O0](?=\d)', 'Ø', text)

# Initialize OCR model
def get_ocr_model():
    from paddleocr import PaddleOCR
    logger.info("Initializing PaddleOCR model with pre-downloaded model files")
    return PaddleOCR(
        use_angle_cls=True,
        lang='en',
        det_model_dir=DET_MODEL_DIR,
        rec_model_dir=REC_MODEL_DIR,
        cls_model_dir=CLS_MODEL_DIR,
        use_gpu=False
    )
# ====== OUTER-BORDER COMPLETION (no internal lines touched) ======
# ====== OUTER-BORDER COMPLETION (Advanced v2) ======
def _binarize(gray):
    """Convert grayscale to binary (text/lines = white)"""
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return 255 - thr

def _find_internal_horizontals(inv, x_left, x_right, min_len_frac=0.35):
    """
    Detect internal horizontal lines within corridor using Hough transform.
    Uses two-pass approach: strict then relaxed detection.
    """
    H, W = inv.shape
    roi = inv[:, x_left:x_right+1]
    edges = cv2.Canny(roi, 30, 100, apertureSize=3)

    ys = []
    # Two-pass detection: strict then relaxed
    for thr, minFrac, maxGap in [(100, 0.50, 15), (70, min_len_frac, 25)]:
        min_len = max(12, int((x_right - x_left + 1) * minFrac))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=thr,
                                minLineLength=min_len, maxLineGap=maxGap)
        if lines is None:
            continue
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y2 - y1) <= 2:  # Horizontal line
                ys.append(int((y1 + y2) // 2))
    
    if not ys:
        return []
    
    # Merge nearby lines
    ys.sort()
    merged = []
    for y in ys:
        if not merged or abs(y - merged[-1]) > 6:
            merged.append(y)
    return merged

def _estimate_corridor_from_text(inv, Y, pad_lr=5):
    """
    Estimate left/right corridor boundaries using percentile of text distribution.
    More robust than simple min/max approach.
    """
    H, W = inv.shape
    if not Y:
        return pad_lr, W - 1 - pad_lr
    
    lefts, rights = [], []
    for y in Y:
        y1 = max(0, y - 12)
        y2 = min(H - 1, y + 12)
        band = inv[y1:y2, :]
        cols = (band > 0).sum(axis=0)
        nz = np.where(cols > 0)[0]
        if nz.size:
            lefts.append(nz.min())
            rights.append(nz.max())
    
    if not lefts or not rights:
        return pad_lr, W - 1 - pad_lr
    
    # Use 5th/95th percentile instead of min/max for robustness
    x_left  = int(np.percentile(lefts,  5)) - pad_lr
    x_right = int(np.percentile(rights, 95)) + pad_lr
    return max(0, x_left), min(W - 1, x_right)

def _median_row_gap(Y):
    """Calculate median gap between consecutive horizontal lines"""
    if len(Y) < 2:
        return None
    gaps = np.diff(sorted(Y))
    return float(np.median(gaps))

def _snap_to_text_edge(inv, x_left, x_right, search='up', pad=4):
    """
    Find y-coordinate just outside first/last ink row within corridor.
    
    Args:
        inv: Binary inverted image
        x_left, x_right: Corridor boundaries
        search: 'up' for top edge, 'down' for bottom edge
        pad: Extra padding pixels
    """
    H = inv.shape[0]
    corr = inv[:, x_left:x_right+1]
    ink = (corr > 0).sum(axis=1)
    
    # Threshold: require ~2% of corridor width to have ink
    thr = max(3, int(0.02 * (x_right - x_left + 1)))
    nz = np.where(ink > thr)[0]
    
    if nz.size == 0:
        return 0 if search == 'up' else H - 1
    
    if search == 'up':
        return max(0, nz[0] - pad)
    else:
        return min(H - 1, nz[-1] + pad)

def _band_coverage(inv, x_left, x_right, y1, y2):
    """
    Calculate ink coverage ratio in a band.
    Used to validate that new borders contain actual content.
    """
    if y2 <= y1:
        return 0.0
    band = inv[y1:y2, x_left:x_right+1]
    total_pixels = band.size
    ink_pixels = (band > 0).sum()
    return float(ink_pixels) / float(total_pixels + 1e-6)

# ====== TOP & BOTTOM BORDER ONLY ======
def add_top_bottom_borders(img_bgr, line_thickness=2):
    """Add only top and bottom horizontal borders"""
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - binary
    
    # Find vertical projection to detect text boundaries
    vertical_proj = np.sum(inv, axis=1)
    threshold = np.max(vertical_proj) * 0.02
    text_rows = np.where(vertical_proj > threshold)[0]
    
    if len(text_rows) > 0:
        y_top = max(0, text_rows[0] - 4)
        y_bottom = min(H - 1, text_rows[-1] + 4)
    else:
        y_top = 0
        y_bottom = H - 1
    
    # Draw only top and bottom lines
    out = gray.copy()
    cv2.line(out, (0, y_top), (W-1, y_top), 0, thickness=line_thickness)
    cv2.line(out, (0, y_bottom), (W-1, y_bottom), 0, thickness=line_thickness)
    
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

# ====== ROW DETECTOR (Hough) ======
def detect_cell_borders(img_bgr):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=int(W * 0.5), maxLineGap=7)

    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:
                horizontal_lines.append((y1 + y2) // 2)

    horizontal_lines = sorted(set(horizontal_lines))
    merged_lines = []
    for y in horizontal_lines:
        if not merged_lines or abs(y - merged_lines[-1]) > 10:
            merged_lines.append(y)

    # ---- Add image top and bottom as borders ----
    # top
    if not merged_lines or abs(merged_lines[0] - 0) > 10:
        merged_lines = [0] + merged_lines
    else:
        merged_lines[0] = 0
    # bottom
    if not merged_lines or abs((H - 1) - merged_lines[-1]) > 10:
        merged_lines = merged_lines + [H - 1]
    else:
        merged_lines[-1] = H - 1
    # --------------------------------------------

    bands = []
    for i in range(len(merged_lines) - 1):
        y1, y2 = merged_lines[i], merged_lines[i + 1]
        if y2 - y1 > 15:
            bands.append((y1, y2))

    return bands

# ====== SPACING / UPSCALE / SHARPEN (used only for Quantity) ======
def add_vertical_spacing(img_bgr, cell_boundaries, spacing_px=25):
    H, W = img_bgr.shape[:2]
    new_h = H + len(cell_boundaries)*spacing_px
    out = np.ones((new_h, W, 3), np.uint8) * 255
    y = 0
    new_bounds = []
    for (y1,y2) in cell_boundaries:
        h = y2-y1
        out[y:y+h, :] = img_bgr[y1:y2, :]
        new_bounds.append((y, y+h))
        y += h + spacing_px
    return out, new_bounds

def upscale_image(img, scale_factor=4):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale_factor), int(h*scale_factor)), interpolation=cv2.INTER_CUBIC)

def sharpen_image(img_bgr, strength=1.5):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    sharp = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=int(strength*100), threshold=3))
    return cv2.cvtColor(np.array(sharp), cv2.COLOR_RGB2BGR)

def _docai_lines(image_bgr):
    """Run Google Document AI and return [{'text', 'x', 'y', 'confidence'}] in image coords."""
    if not (DOC_PROJECT_ID and DOC_LOCATION and DOC_PROCESSOR):
        logger.warning("DocAI env not set; skipping")
        return []
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return []
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(DOC_PROJECT_ID, DOC_LOCATION, DOC_PROCESSOR)
    raw_document = documentai.RawDocument(content=buf.tobytes(), mime_type="image/png")
    result = client.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw_document))

    H, W = image_bgr.shape[:2]  # ← ADD WIDTH
    out = []
    for page in result.document.pages:
        for line in getattr(page, "lines", []):
            text = ""
            for seg in line.layout.text_anchor.text_segments:
                s = int(seg.start_index) if seg.start_index else 0
                e = int(seg.end_index) if seg.end_index else 0
                text += result.document.text[s:e]
            if not text.strip():
                continue
            v = line.layout.bounding_poly.normalized_vertices
            x = ((v[0].x + v[2].x)/2.0) * W  # ← ADD X COORDINATE
            y = ((v[0].y + v[2].y)/2.0) * H
            # ← ADD PHI SYMBOL FIX HERE
            cleaned_text = fix_diameter(text.strip())
            out.append({"text": cleaned_text, "x": x, "y": y, "confidence": float(line.layout.confidence or 0.9)})
    return out

def _map_to_bands(lines, bands, scale=1.0):
    """Map DocAI lines to (y1,y2) bands. If the image was upscaled, pass scale (e.g., 4)."""
    cells = []
    for (y1, y2) in bands:
        lo, hi = y1*scale, y2*scale
        here = [l for l in lines if lo <= l["y"] <= hi]
        if here:
            # ← SORT BY Y FIRST (top to bottom), THEN X (left to right)
            here.sort(key=lambda x: (x["y"], x["x"]))
            txt = " ".join(l["text"] for l in here)
            conf = min(l["confidence"] for l in here)
            cells.append({"text": txt, "confidence": conf})
        else:
            cells.append({"text": "", "confidence": 0.0})
    return cells

def docai_extract_column(img_bgr, column_name: str):
    """
    Column-specific processing:
    - PartNumber: Raw DocAI only (no borders, no hough)
    - Quantity: Borders → Hough → Spacing+Upscale+Sharpen → DocAI
    - Others (Desc/Material): Borders → Hough → DocAI
    """
    column_lower = (column_name or "").lower()
    
    # PartNumber: NO preprocessing, just raw DocAI
    if column_lower == "partnumber":
        lines = _docai_lines(img_bgr)  # ← Direct DocAI on raw image
        if not lines:
            return []  # triggers fallback (Paddle only)
        # Since there's no cell detection, return as single-cell rows
        return [{"text": l["text"], "confidence": l["confidence"]} for l in lines]
    
    # For all other columns: apply border completion + hough
    #img_done = complete_outer_borders_only(img_bgr)
    img_done = add_top_bottom_borders(img_bgr)
    bands = detect_cell_borders(img_done)
    if not bands:
        return []  # triggers fallback chain

    # Quantity: special processing
    if column_lower == "quantity":
        spaced, spaced_bands = add_vertical_spacing(img_done, bands, spacing_px=25)
        up = upscale_image(spaced, 4)
        fin = sharpen_image(up, 1.5)
        lines = _docai_lines(fin)
        return _map_to_bands(lines, spaced_bands, scale=4.0)
    
    # Description/Material: borders + hough only (no scaling)
    else:
        lines = _docai_lines(img_done)
        return _map_to_bands(lines, bands, scale=1.0)

def log_backend_choice(run_id: str, column: str, winner: str):
    """Try to log to Google Sheets; if not configured, fall back to webhook; else just log."""
    row = [
        datetime.datetime.utcnow().isoformat(),
        run_id,
        column,
        1 if winner == "docai" else 0,
        1 if winner == "openai" else 0,
        1 if winner == "paddle" else 0,
    ]

    # 1) Prefer Google Sheets API if SHEET_ID is set
    try:
        if SHEET_ID:
            svc = _get_sheets_service()
            if svc:
                body = {"values": [row]}
                svc.spreadsheets().values().append(
                    spreadsheetId=SHEET_ID,
                    range=f"{SHEET_TAB}!A:F",
                    valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS",
                    body=body,
                ).execute()
                logger.info("[LOG] wrote row to Google Sheets")
                return
    except Exception as e:
        logger.warning(f"[LOG] Sheets write failed: {e}")

    # 2) Optional fallback to webhook (keep your existing env if you like)
    url = os.getenv("SHEET_LOG_WEBHOOK_URL")
    if url:
        try:
            httpx.post(
                url,
                json={
                    "run": run_id,
                    "column": column,
                    "google_doc_ai": row[3],
                    "openai": row[4],
                    "paddleocr": row[5],
                },
                timeout=5.0,
            )
            logger.info("[LOG] wrote row via webhook fallback")
            return
        except Exception as e:
            logger.warning(f"[LOG] webhook failed: {e}")

    # 3) Else: just log to stdout
    logger.info(f"[LOG] (no sheets/webhook) run={run_id} column={column} winner={winner}")

# Process image with OCR
def simple_cells(img_rgb):
    """
    Run PaddleOCR on an RGB image and return one cell per detected box,
    sorted by its vertical (y) center.
    """
    ocr_model = get_ocr_model()
    raw = ocr_model.ocr(img_rgb, cls=True)[0]

    print(f"🔍 simple_cells: OCR detected {len(raw) if raw else 0} items")
    
    if not raw:
        print("❌ OCR returned no results at all")
        return []

    cells = []
    for i, (box, (text, confidence)) in enumerate(raw):
        print(f"  Item {i}: '{text}' (conf: {confidence:.2f})")
        raw_text = text.strip()  #new line added
        if not raw_text:  #new line added
            print(f"    ⚠️ Skipped: empty text")
            continue      #new line added
        cleaned = fix_diameter(raw_text)   #new line added       
        # if not text.strip():
        #     continue
        # compute the vertical midpoint for sorting
        y_center = int((box[0][1] + box[2][1]) / 2)
        cells.append({
            "y_center": y_center,
            "text": cleaned,    #text.strip()
            "confidence": confidence
        })

    print(f"📊 simple_cells: Returning {len(cells)} valid cells")
    # stable sort top→bottom
    cells.sort(key=lambda c: c["y_center"])
    # drop the y_center before returning
    return [{"text": c["text"], "confidence": c["confidence"]} for c in cells]


def advanced_cells_with_rectangles(img):
    # 1) resize+decode as before...
    #    (make sure `img` here is your OpenCV BGR image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    h, w = img.shape[:2]

    # 2) horizontal strokes
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w//80), 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)

    # 3) vertical strokes
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, h//80)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel)

    # —— now CLOSE each so broken segments reconnect —— 
    horiz = cv2.morphologyEx(horiz,
                             cv2.MORPH_CLOSE,
                             np.ones((3,3), np.uint8),
                             iterations=1)
    vert  = cv2.morphologyEx(vert,
                             cv2.MORPH_CLOSE,
                             np.ones((3,3), np.uint8),
                             iterations=1)

    # 4) AND them to get only true grid‐lines
    grid = cv2.bitwise_and(horiz, vert)

    # —— final closing so tiny gaps don’t break a cell in two ——
    grid = cv2.morphologyEx(grid,
                             cv2.MORPH_CLOSE,
                             np.ones((5,5), np.uint8),
                             iterations=1)
    

    # # 5) optional: dilate so borders join cleanly into rectangles
    # grid = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=1)

    # 6) find all contours on that grid
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        # throw away anything too small to be a cell
        if rw < w//20 or rh < h//30:
            continue
        rects.append((x, y, rw, rh))

    # === DEBUG VISUALIZATION START ===
    # draw all the rects you’ve kept onto a copy of the image
    debug_img = img.copy()
    for (x, y, rw, rh) in rects:
        cv2.rectangle(debug_img, (x, y), (x+rw, y+rh), (0,255,0), 2)
    # save to a temp file so you can inspect it
    debug_path = f"/tmp/cell_debug_{uuid.uuid4().hex[:8]}.png"
    cv2.imwrite(debug_path, debug_img)
    print("🛠️  Debug cell map written to:", debug_path)
    # === DEBUG VISUALIZATION END ===    

    
    
    # if we found **no** real rectangles, fall back
    if not rects:
        print("⚠️  No rectangles detected, falling back to simple_cells")
        return simple_cells(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 7) sort the rectangles top→bottom, left→right
    rects.sort(key=lambda r: (r[1], r[0]))

    # 8) do one OCR pass and map each snippet into its containing rect
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raw = get_ocr_model().ocr(rgb, cls=True)[0]
    cells = []
    for box, (text, conf) in raw:
        raw_text = text.strip()
        # if not text.strip(): 
        #     continue
        if not raw_text:
            continue
        cleaned = fix_diameter(raw_text)
        mx = int((box[0][0] + box[2][0]) / 2)
        my = int((box[0][1] + box[2][1]) / 2)
        # find which rectangle contains this midpoint
        for idx, (x, y, rw, rh) in enumerate(rects):
            if x <= mx < x+rw and y <= my < y+rh:
                cells.append((idx, mx, my, cleaned, conf))
                break

    # 9) for each rect‐index, collect its bits, sort by x (then y), glue text
    out = []
    for i, (x, y, rw, rh) in enumerate(rects):
        bucket = [(mx, my, t, c) for (idx, mx, my, t, c) in cells if idx == i]
        if not bucket:
            out.append({"text": "", "confidence": 0})
            continue
        # reading order inside a cell: left→right, top→bottom
        bucket.sort(key=lambda e: (e[1], e[0]))
        joined = " ".join(e[2] for e in bucket)
        conf   = min(e[3] for e in bucket)
        out.append({"text": joined, "confidence": conf})

    return out


def advanced_cells(img):
    # 1) Single OCR pass (RGB)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raw = get_ocr_model().ocr(rgb, cls=True)[0]

    # 2) Estimate a “typical” line‐height and set merge_thresh = max(median_h, 20px)
    heights = [abs(box[2][1] - box[0][1]) for box, (txt, _) in raw if txt.strip()]
    if heights:
        median_h = sorted(heights)[len(heights)//2]
        merge_thresh = max(median_h, 20)
    else:
        merge_thresh = 20

    # 3) Binarize & invert for horizontal‐line detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # 4) Extract horizontal strokes
    h, w = img.shape[:2]
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w//80), 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern)

    # 5) Hough‐detect those strokes (even short ones)
    lines = cv2.HoughLinesP(
        horiz,
        rho=1, theta=np.pi/180,
        threshold=30,
        minLineLength=w//40,
        maxLineGap=5
    )

    # 6) Cluster all y‐coordinates of detected lines into row_bounds
    ys = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            ys += [y1, y2]
    ys.sort()

    clusters = []
    for y in ys:
        if not clusters or abs(y - clusters[-1][0]) > merge_thresh:
            clusters.append([y])
        else:
            clusters[-1].append(y)
    row_bounds = [int(sum(c)/len(c)) for c in clusters]

    # 7) Fallback: if we found no interior lines, drop back to your old simple_cells
    if lines is None or len(row_bounds) < 2:
        return simple_cells(rgb)

    # 8) Bucket the same OCR boxes into those horizontal bands
    cells = []
    for box, (txt, conf) in raw:
        if not txt.strip():
            continue
        x = int((box[0][0] + box[2][0]) / 2)
        y = int((box[0][1] + box[2][1]) / 2)
        cells.append({"x": x, "y": y, "text": txt.strip(), "conf": conf})

    rows = []
    # head (above the first grid line)
    head = [c for c in cells if c["y"] < row_bounds[0]]
    if head:
        head.sort(key=lambda c: (c["y"], c["x"]))
        rows.append({
            "text": " ".join(c["text"] for c in head),
            "confidence": min(c["conf"] for c in head)
        })

    # middle bands
    for top, bot in zip(row_bounds, row_bounds[1:]):
        band = [c for c in cells if top <= c["y"] < bot]
        if not band:
            continue
        band.sort(key=lambda c: (c["y"], c["x"]))
        rows.append({
            "text": " ".join(c["text"] for c in band),
            "confidence": min(c["conf"] for c in band)
        })

    # tail (below the last grid line)
    tail = [c for c in cells if c["y"] >= row_bounds[-1]]
    if tail:
        tail.sort(key=lambda c: (c["y"], c["x"]))
        rows.append({
            "text": " ".join(c["text"] for c in tail),
            "confidence": min(c["conf"] for c in tail)
        })

    return rows
'''
def gemini_extract_column(image_bytes, column_name):
    """
    Extract a single column from image using Gemini Vision.
    Returns list of dicts: [{text: str, confidence: float}]
    """
    try:
        if not GEMINI_API_KEY:
            raise Exception("Gemini API key not configured")
        
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Column-specific prompts
        prompts = {
            "PartNumber": "Extract valid text from this table column. Return whole cell content per line, nothing else. If a cell has multiple lines, combine them into one line.",
            "Quantity": "Extract valid text from this table column. Return whole cell content per line, nothing else. If a cell has multiple lines, combine them into one line.",
            "Description": "Extract valid text from this table column. Return whole cell content per line, nothing else. If a cell has multiple lines, combine them into one line.",
            "Material": "Extract valid text from this table column. Return whole cell content per line, nothing else. If a cell has multiple lines, combine them into one line."
        }
        
        prompt = prompts.get(column_name, prompts["Description"])
        
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content([
            prompt + "\n\nIMPORTANT: Each table row should produce exactly ONE line in your output, even if the cell text spans multiple lines in the image.",
            {
                'mime_type': 'image/jpeg',
                'data': image_b64
            }
        ])
        
        # Parse response
        text = response.text.strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Return in same format as PaddleOCR
        result = []
        for line in lines:
            result.append({
                "text": line,
                "confidence": 0.95  # Gemini doesn't provide confidence, use high default
            })
        
        logger.info(f"✅ Gemini extracted {len(result)} items for {column_name}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Gemini extraction failed: {e}")
        raise
'''

def openai_extract_column(image_bytes, column_name):
    """
    Extract column using OpenAI GPT-4o Vision
    """
    try:
        if not openai_client:
            raise Exception("OpenAI API key not configured")
        
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a specialized OCR extraction tool. Extract only visible text from images. Never provide explanations, commentary, or conversational responses. Output only the extracted data."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this table column image and extract cell contents.
.
Cell identification: Count ONLY horizontal border lines to identify cells. Ignore vertical spacing/whitespace.
Multi-line handling: If text wraps onto multiple lines within one cell, join with single space.
Whitespace handling: Empty vertical space within bordered cells is NOT a separate cell - skip it completely.
Independence: Each cell value is independent. Never reference or copy from other cells.
Format: Return one line per cell with text, top to bottom order. Skip lines for cells with only whitespace.
Empty cells: If a bordered cell has no text at all, skip it (do not output blank line).

Critical: A cell is defined by horizontal borders, NOT by vertical spacing. Large vertical gaps within one bordered area are still ONE cell.
IMPORTANT: If consecutive cells contain identical or similar text, output each occurrence separately. Do NOT merge or deduplicate cells with same content. Each bordered cell must appear in output regardless of similarity to adjacent cells.
Warning : Return in same order, don't change any order
Output the cell values only, nothing else."""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        }
    ],
    max_tokens=500,
    temperature=0
)
        
        text = response.choices[0].message.content.strip()
        
        # Strip markdown formatting
        if text.startswith('```'):
            lines_split = text.split('\n')
            text = '\n'.join(lines_split[1:-1]) if len(lines_split) > 2 else text
            text = text.replace('```', '').strip()
        
        # Detect conversational responses
        lower_text = text.lower()
        reject_phrases = [
            "i cannot", "i can't", "sorry", "unable to",
            "appears to", "seems to", "the image", "this is",
            "how can i", "what would", "please provide"
        ]
        
        if any(phrase in lower_text for phrase in reject_phrases):
            logger.warning(f"GPT-4o returned conversational text: {text[:80]}")
            raise Exception("Conversational response detected")
        
        lines = [line.strip() for line in text.split('\n')]
        
        if not lines:
            raise Exception("Empty extraction")
        
        result = [{"text": line, "confidence": 0.50} for line in lines]
        
        logger.info(f"GPT-4o extracted {len(result)} items")
        return result
        
    except Exception as e:
        logger.error(f"GPT-4o extraction failed: {e}")
        raise
# Working fine except multiline text extraction
# def advanced_cells(img):

#     # 1) Run one OCR pass to get raw boxes
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     raw = get_ocr_model().ocr(rgb, cls=True)[0]

#     # 2) Estimate a typical line-height from the OCR boxes
#     heights = [abs(box[2][1] - box[0][1]) for box,(_,_) in raw]
#     if heights:
#         median_h = sorted(heights)[len(heights)//2]
#         # half a line-height but never below 5px
#         merge_thresh = max(5, median_h // 2)
#     else:
#         # fallback if OCR saw nothing
#         merge_thresh = 5

#     # 3) Binarize & invert for line-detect
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

#     # 4) Extract horizontal strokes
#     h, w = img.shape[:2]
#     kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w//80), 1))
#     horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern)

#     # 5) Hough for even short lines
#     lines = cv2.HoughLinesP(
#         horiz, rho=1, theta=np.pi/180,
#         threshold=30, minLineLength=w//40, maxLineGap=5
#     )

#     # 6) Cluster all y’s into row_bounds using the dynamic threshold
#     ys = []
#     if lines is not None:
#         for x1,y1,x2,y2 in lines[:,0]:
#             ys += [y1, y2]
#     ys.sort()

#     clusters = []
#     for y in ys:
#         if not clusters or abs(y - clusters[-1][0]) > merge_thresh:
#             clusters.append([y])
#         else:
#             clusters[-1].append(y)
#     row_bounds = [int(sum(c)/len(c)) for c in clusters]

#     # 7) FALLBACK if no lines or only one cluster
#     if lines is None or len(row_bounds) < 2:
#         return simple_cells(rgb)

#     # 8) Now bucket the same raw OCR into those bands
#     cells = []
#     for box,(t,c) in raw:
#         if not t.strip(): continue
#         x = int((box[0][0]+box[2][0]) / 2)
#         y = int((box[0][1]+box[2][1]) / 2)
#         cells.append({"x": x, "y": y, "text": t.strip(), "conf": c})

#     rows = []
#     # head (above first line)
#     head = [c for c in cells if c["y"] < row_bounds[0]]
#     if head:
#         head.sort(key=lambda c:(c["y"],c["x"]))
#         rows.append({
#             "text": " ".join(c["text"] for c in head),
#             "confidence": min(c["conf"] for c in head)
#         })

#     # middle bands
#     for top, bot in zip(row_bounds, row_bounds[1:]):
#         band = [c for c in cells if top <= c["y"] < bot]
#         if not band: continue
#         band.sort(key=lambda c:(c["y"],c["x"]))
#         rows.append({
#             "text": " ".join(c["text"] for c in band),
#             "confidence": min(c["conf"] for c in band)
#         })

#     # tail (below last line)
#     tail = [c for c in cells if c["y"] >= row_bounds[-1]]
#     if tail:
#         tail.sort(key=lambda c:(c["y"],c["x"]))
#         rows.append({
#             "text": " ".join(c["text"] for c in tail),
#             "confidence": min(c["conf"] for c in tail)
#         })

#     return rows

# def process_image(image_bytes):
#     import cv2, numpy as np

#     # —————————————————————————————
#     # 1) Decode & resize exactly as before
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     max_dim = 800
#     h, w = img.shape[:2]
#     if max(h, w) > max_dim:
#         scale = max_dim / max(h, w)
#         img = cv2.resize(img, (int(w*scale), int(h*scale)))
    
#     # 2) Prepare a gray/binary image for line detection
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
#     # 3) Extract long horizontal strokes (the table grid‐lines)
#     horiz_kernel = cv2.getStructuringElement(
#         cv2.MORPH_RECT, (max(20, img.shape[1]//30), 1)
#     )
#     horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)
    
#     # 4) Hough‐detect those strokes
#     lines = cv2.HoughLinesP(
#         horiz, rho=1, theta=np.pi/180, threshold=100,
#         minLineLength=img.shape[1]//2, maxLineGap=20
#     )
#     ys = []
#     if lines is not None:
#         for x1, y1, x2, y2 in lines[:,0]:
#             ys += [y1, y2]
#     ys.sort()
    
#     # 5) Cluster y’s that are within 5px of each other → true row boundaries
#     clusters = []
#     for y in ys:
#         if not clusters or abs(y - clusters[-1][0]) > 5:
#             clusters.append([y])
#         else:
#             clusters[-1].append(y)
#     row_bounds = sorted(int(sum(c)/len(c)) for c in clusters)
    
#     # —————————————————————————————
#     # 6) Finally run your OCR on the (RGB) image
#     image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = get_ocr_model().ocr(image_rgb, cls=True)[0]
    
#     # 7) Compute each snippet’s center
#     cells = []
#     for box, (text, conf) in results:
#         if not text.strip(): 
#             continue
#         x_c = int((box[0][0] + box[2][0]) / 2)
#         y_c = int((box[0][1] + box[2][1]) / 2)
#         cells.append({"x": x_c, "y": y_c, "text": text.strip(), "conf": conf})
    
#     # 8) Group into bands between each consecutive pair of row_bounds
#     rows = []
#     for i in range(len(row_bounds) - 1):
#         top, bot = row_bounds[i], row_bounds[i+1]
#         band = [c for c in cells if top <= c["y"] < bot]
#         if not band:
#             continue
#         band.sort(key=lambda c: c["x"])
#         merged_text = " ".join(c["text"] for c in band)
#         merged_conf = min(c["conf"] for c in band)
#         rows.append({"text": merged_text, "confidence": merged_conf})
    
#     return rows

    
# def process_image(image_bytes):
#     logger.info(f"Processing image of size {len(image_bytes)} bytes")
    
#     # Convert to OpenCV format
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     # Resize image for faster processing
#     max_dim = 800
#     height, width = img.shape[:2]
#     if max(height, width) > max_dim:
#         scale = max_dim / max(height, width)
#         img = cv2.resize(img, (int(width * scale), int(height * scale)))
#         logger.info(f"Resized image to {img.shape[1]}x{img.shape[0]}")
    
#     # Convert to RGB for OCR
#     image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Process with OCR
#     logger.info("Starting OCR processing")
#     ocr_model = get_ocr_model()
#     results = ocr_model.ocr(image_rgb, cls=True)
#     cells = []
    
#     if results and len(results) > 0 and len(results[0]) > 0:
#         for box, (text, confidence) in results[0]:
#             if text.strip():
#                 y_center = int((box[0][1] + box[2][1]) / 2)
#                 cells.append({"y_center": y_center, "text": text.strip(), "confidence": confidence})
#         cells.sort(key=lambda c: c["y_center"])
    
#     logger.info(f"OCR completed, found {len(cells)} text elements")
#     return cells

# Health check endpoint
@app.get("/health")
async def health_check():
    model_paths_exist = all([
        os.path.exists(DET_MODEL_DIR),
        os.path.exists(REC_MODEL_DIR),
        os.path.exists(CLS_MODEL_DIR)
    ])
    return {
        "status": "ok", 
        "model_files_exist": model_paths_exist,
        "paddle_home": PADDLE_HOME
    }

# Test endpoint
@app.get("/test")
async def test_endpoint():
    return {"status": "ok", "message": "API is working"}

# Handle HEAD requests
@app.head("/")
async def head_endpoint():
    return JSONResponse(content={"status": "ok"})

# Main OCR endpoint
@app.post("/")
async def ocr_endpoint(request: Request):
    try:
        logger.info(f"Received OCR request with Content-Type: {request.headers.get('content-type')}")

        # 1) Get form + fields
        form = await request.form()
        logger.info(f"Received form data with keys: {list(form.keys())}")

        if "image" not in form:
            logger.warning("Missing image parameter in request")
            return JSONResponse(status_code=400, content={"error": "Missing image parameter"})
        if "mode" not in form:
            logger.warning("Missing mode parameter in request")
            return JSONResponse(status_code=400, content={"error": "Missing mode parameter"})
        # ← NEW: optional “column” tag
        column_id = form.get("column", None)

        image_file = form["image"]
        mode       = form["mode"]
        logger.info(f"Processing request in mode: {mode}, image filename: {image_file.filename}, column: {column_id}")

        # 2) Read the bytes
        image_bytes = await image_file.read()

        # 3) Decode once to a CV2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 4) Dispatch to the right OCR routine under a worker thread
        def do_ocr():
            # quick mode: just raw text join
            if mode == "quick":
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # reuse simple_cells to get list of dicts
                cells = simple_cells(rgb)
                extracted_text = "\n".join(c["text"] for c in cells)
                return {"mode": mode, "extracted_text": extracted_text, "cells": cells}

            # table mode: choose by column tag
            elif mode == "table":
                run_id = form.get("run") or uuid.uuid4().hex[:8]
                column_lower = (column_id or "").lower()
            
                # 1) Try Google Document AI (with special treatment for Quantity/PartNumber)
                try:
                    table_cells = docai_extract_column(img, column_id or "")
                    
                    # ====== NEW: QUANTITY SINGLE-CELL FALLBACK ======
                    if column_lower == "quantity" and len(table_cells) == 1:
                        logger.warning(f"⚠️ Quantity column: Only 1 cell detected by DocAI, falling back to OpenAI")
                        raise Exception("Single cell detected - triggering OpenAI fallback")
                    # ====== END QUANTITY FALLBACK ======
                    
                    if any(c.get("text") for c in table_cells):
                        log_backend_choice(run_id, column_id or "", "docai")
                        return {"mode": "table", "column": column_id, "table": table_cells, "engine": "docai"}
                except Exception as e:
                    logger.warning(f"DocAI failed: {e}")
            
                # 2) Fallback logic depends on column type
                # For PartNumber: skip OpenAI, go straight to Paddle
                if column_lower == "partnumber":
                    logger.info(f"PartNumber column: Skipping OpenAI, using Paddle fallback")
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    paddle_cells = simple_cells(rgb)
                    log_backend_choice(run_id, column_id or "", "paddle")
                    return {"mode": "table", "column": column_id, "table": paddle_cells, "engine": "paddle"}
            
                # For other columns: try OpenAI then Paddle
                try:
                    result = openai_extract_column(image_bytes, column_id or "")
                    if result and any(r.get("text") for r in result):
                        log_backend_choice(run_id, column_id or "", "openai")
                        return {"mode": "table", "column": column_id, "table": result, "engine": "openai"}
                except Exception as e:
                    logger.warning(f"OpenAI fallback failed: {e}")
            
                # 3) Last fallback: PaddleOCR
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                paddle_cells = simple_cells(rgb)
                log_backend_choice(run_id, column_id or "", "paddle")
                return {"mode": "table", "column": column_id, "table": paddle_cells, "engine": "paddle"}
                
                # # quantity gets the old per‐line logic
                # if column_id == "quantity":
                #     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     table_cells = simple_cells(rgb)
                # # everything else uses the fancy line‐based
                # else:
                #     table_cells = advanced_cells_with_rectangles(img)
                # return {"mode": mode, "table": table_cells}

            else:
                raise ValueError(f"Invalid mode provided: {mode}")

        # 5) Run OCR with timeout protection
        try:
            result = await asyncio.to_thread(do_ocr)
            return result

        except asyncio.TimeoutError:
            logger.error("OCR processing timed out")
            return JSONResponse(
                status_code=504,
                content={"error": "Processing timed out. Try with a smaller image."}
            )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Error processing request: {e}\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Server error: {str(e)}"}
        )

# Extracts Drawing number from File Name
def extract_drawing_number(url: str):
    if not url:
        return ""
    match = re.search(r"/([^/]+)\.pdf$", url, re.IGNORECASE)
    return match.group(1) if match else ""

# Updates the lastItem Number in glide Drawing table
async def update_last_ocr_bom_item_direct(row_id: str, new_last_item: int):
    """Update lastOcrBomItem using direct rowID"""
    try:
        print(f"🔄 Updating lastOcrBomItem to {new_last_item} for rowID: {row_id}")
        
        update_body = {
            "appID": GLIDE_APP_ID,
            "mutations": [{
                "kind": "set-columns-in-row",
                "tableName": GLIDE_TABLE,
                "columnValues": {"WddPP": new_last_item},
                "rowID": row_id
            }]
        }
        
        async with httpx.AsyncClient() as client:
            update_res = await client.post(
                "https://api.glideapp.io/api/function/mutateTables",
                headers={"Authorization": f"Bearer {GLIDE_API_KEY}", "Content-Type": "application/json"},
                json=update_body
            )
        
        update_res.raise_for_status()
        result = update_res.json()
        print(f"✅ Successfully updated lastOcrBomItem to {new_last_item}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating lastOcrBomItem: {e}")
        return False

# API code to fetch drawing numbers (File name)
GLIDE_API_KEY = os.getenv("GLIDE_API_KEY")
GLIDE_APP_ID = os.getenv("GLIDE_APP_ID")
GLIDE_TABLE = os.getenv("GLIDE_TABLE")
# ZAPIER_WEBHOOK_URL = "YOUR_ACTUAL_ZAPIER_WEBHOOK_URL_HERE"

#Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


@app.post("/fetch-drawings")
async def fetch_drawings(request: Request):
    print("🎯 /fetch-drawings endpoint hit")

    try:
        payload = await request.json()
        project = payload.get("project")
        part_number = payload.get("part")

        print("📦 Incoming fetch-drawings request:", project, part_number)
        
        if not project or not part_number:
            return {"error": "Missing project or part"}

        # ✅ Correct API format
        body = {
            "appID": GLIDE_APP_ID,
            "queries": [
                {
                    "tableName": GLIDE_TABLE,
                    "utc": True
                }
            ]
        }
        
        print("📤 Sending request to Glide:", body)

        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.glideapp.io/api/function/queryTables",
                headers={
                    "Authorization": f"Bearer {GLIDE_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=body
            )

        res.raise_for_status()
        
        try:
            full_data = res.json()
            print("✅ Full response from Glide:", full_data)
            # ✅ Extract and filter rows manually
            all_rows = full_data[0]["rows"]
            filtered = [
                row for row in all_rows
                if row.get("VQlMl") == project and row.get("nlHAO") == part_number
            ]

            print(f"✅ Filtered rows: {len(filtered)} match")
            # ✅ Add extracted drawingNumber from drawing link
            filtered_trimmed = [
                {
                    "project": row.get("VQlMl"),
                    "partNumber": row.get("nlHAO"),
                    "partName": row.get("Name"),
                    "drawingLink": row.get("9iB5E"),
                    "drawingNumber": extract_drawing_number(row.get("9iB5E"))
                }
                for row in filtered
            ]
        
            return {"rows": filtered_trimmed}
        
        except Exception:
            error_text = await res.aread()
            print("❌ Non-JSON response from Glide:", error_text.decode())
            raise


    except Exception as e:
        import traceback
        print("❌ Exception occurred:")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


# Add these endpoints to your existing FastAPI backend (main.py)
@app.post("/add-child-parts")
async def add_child_parts(request: Request):
    """Add child parts data to Glide Child Parts table"""
    print("🎯 /add-child-parts endpoint hit")
    
    try:
        payload = await request.json()
        rows_data = payload.get("rows", [])
        project = payload.get("project")
        parent_drawing_number = payload.get("parentDrawingNumber")
        part_number = payload.get("partNumber")  # Overall Part Number from URL
        rowID = payload.get("rowID")
        maxItemNumber = payload.get("maxItemNumber")
        
        print(f"📦 Child Parts Request: project={project}, parent={parent_drawing_number}, part={part_number}")
        print(f"📊 Rows to add: {len(rows_data)}, rowID: {rowID}, maxItemNumber: {maxItemNumber}")
        
        if not project or not parent_drawing_number or not part_number:
            return JSONResponse(
                status_code=400, 
                content={"error": "Missing required parameters: project, parentDrawingNumber, or partNumber"}
            )
        
        if not rows_data:
            return JSONResponse(
                status_code=400,
                content={"error": "No rows data provided"}
            )
        
        # Build mutations for Child Parts table
        mutations = []
        for row in rows_data:         
            mutation = {
                "kind": "add-row-to-table",
                "tableName": "native-table-3HZdeQgfDL37ac2rc3kF",  # Child Parts table
                "columnValues": {
                    "remote\u001dPart number": part_number,  # Overall Part Number from URL
                    "remote\u001dParent drawing number": parent_drawing_number,
                    "remote\u001dDrawing number": row.get("drawingNumber", ""),  # Select Drawing Number
                    "remote\u001dQuantity": str(row.get("quantity", "")),
                    "remote\u001dProject Name": project,
                    "qkM5k": row.get("description", ""),
                    "JIGhW": row.get("material", ""),
                    "remote\u001dItem #": row.get("itemNumber"),
                    "Inzp9":row.get("ocrWarning", "")
                    # Note: Item # is not being sent as per your requirement
                }
            }
            mutations.append(mutation)
        
        if not mutations:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid rows to add (missing required fields)"}
            )
        
        # Prepare Glide API request
        glide_body = {
            "appID": GLIDE_APP_ID,
            "mutations": mutations
        }
        
        print(f"📤 Sending {len(mutations)} child parts to Glide...")
        
        # Send to Glide API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.glideapp.io/api/function/mutateTables",
                headers={
                    "Authorization": f"Bearer {GLIDE_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=glide_body
            )
        
        response.raise_for_status()
        result = response.json()
        
        print("✅ Child Parts added successfully:", result)

        if rowID and maxItemNumber:
            update_success = await update_last_ocr_bom_item_direct(rowID, maxItemNumber)
            print(f"🎯 Drawing table update result: {update_success}") 
        return {
            "success": True, 
            "message": f"Successfully added {len(mutations)} child parts",
            "glide_response": result
        }
        
    except httpx.HTTPStatusError as e:
        error_text = await e.response.aread()
        print(f"❌ Glide API Error: {e.response.status_code} - {error_text.decode()}")
        return JSONResponse(
            status_code=e.response.status_code,
            content={"error": f"Glide API error: {error_text.decode()}"}
        )
    except Exception as e:
        import traceback
        print("❌ Exception in add_child_parts:")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Server error: {str(e)}"}
        )


@app.post("/add-bo-parts")
async def add_bo_parts(request: Request):
    """Add BO (Bought Out) parts data to Glide BO Parts table"""
    print("🎯 /add-bo-parts endpoint hit")
    
    try:
        payload = await request.json()
        rows_data = payload.get("rows", [])
        project = payload.get("project")
        parent_drawing_number = payload.get("parentDrawingNumber")
        part_number = payload.get("partNumber")  # Overall Part Number from URL

        rowID = payload.get("rowID")
        maxItemNumber = payload.get("maxItemNumber")
        
        print(f"📦 BO Parts Request: project={project}, parent={parent_drawing_number}, part={part_number}")
        print(f"📊 Rows to add: {len(rows_data)}")
        
        if not project or not parent_drawing_number or not part_number:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required parameters: project, parentDrawingNumber, or partNumber"}
            )
        
        if not rows_data:
            return JSONResponse(
                status_code=400,
                content={"error": "No rows data provided"}
            )
        
        # Build mutations for BO Parts table
        mutations = []
        for row in rows_data:       
            mutation = {
                "kind": "add-row-to-table",
                "tableName": "native-table-l2JX33tUJwUKYmNz7ZEs",  # BO Parts table
                "columnValues": {
                    "remote\u001dProject name": project,
                    "remote\u001dOverall Part number": part_number,  # Overall Part Number from URL
                    "remote\u001dParent drawing": parent_drawing_number,
                    "remote\u001dBoughout Part number": row.get("boughtoutPartNumber", ""),  # Drawing Number or Frontend Part Number
                    "remote\u001dDescription": row.get("description", ""),
                    "remote\u001dMOC": row.get("material", ""),  # Material goes to MOC field
                    "remote\u001dQuantity": str(row.get("quantity", "")),
                    "JPBNt": row.get("ocrWarning", ""),
                    "wRubP": row.get("itemNumber")
                    # Note: cbN8e (Last updated at), remote\u001dItem number, and 8Kjom (Boughtout rate) are not being sent
                }
            }
            mutations.append(mutation)
        
        if not mutations:
            return JSONResponse(
                status_code=400,
                content={"error": "No valid rows to add (missing required fields)"}
            )
        
        # Prepare Glide API request
        glide_body = {
            "appID": GLIDE_APP_ID,
            "mutations": mutations
        }
        
        print(f"📤 Sending {len(mutations)} BO parts to Glide...")
        
        # Send to Glide API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.glideapp.io/api/function/mutateTables",
                headers={
                    "Authorization": f"Bearer {GLIDE_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=glide_body
            )
        
        response.raise_for_status()
        result = response.json()
        
        print("✅ BO Parts added successfully:", result)

        if rowID and maxItemNumber:
            update_success = await update_last_ocr_bom_item_direct(rowID, maxItemNumber)
            print(f"🎯 Drawing table update result: {update_success}") 

        return {
            "success": True,
            "message": f"Successfully added {len(mutations)} BO parts", 
            "glide_response": result
        }
        
    except httpx.HTTPStatusError as e:
        error_text = await e.response.aread()
        print(f"❌ Glide API Error: {e.response.status_code} - {error_text.decode()}")
        return JSONResponse(
            status_code=e.response.status_code,
            content={"error": f"Glide API error: {error_text.decode()}"}
        )
    except Exception as e:
        import traceback
        print("❌ Exception in add_bo_parts:")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Server error: {str(e)}"}
        )
# Childpart & BO Data Post Ends here

# @app.post("/generate-missing-childpart-pdf")
# async def generate_missing_pdf(request: Request):
#     # Get project and part number from query params
#     url_params = dict(request.query_params)
#     project = url_params.get("project")
#     part_number = url_params.get("part")

#     # Get matchedPart and zapierWebhookUrl from payload
#     payload = await request.json()
#     matched_part = payload.get("matchedPart") or part_number

#     if not project or not part_number:
#         return JSONResponse(
#             status_code=400,
#             content={"error": "Missing required parameters: project, part (in query), or zapierWebhookUrl (in body)"}
#         )

#     # 1. Create PDF file
#     pdf = FPDF(orientation='L', unit='mm', format='A4')
#     pdf.add_page()
#     pdf.set_font("Arial", size=24)
#     pdf.cell(0, 60, txt="Missing Child Part Drawing", ln=True, align='C')
#     pdf.set_font("Arial", size=18)
#     pdf.cell(0, 10, txt=f"Part Number: {matched_part}", ln=True, align='C')

#     # Use a temp file for PDF
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         filename = tmp.name
#         pdf.output(filename)

#     # 2. Send to Zapier
#     with open(filename, 'rb') as f:
#         files = {'file': (f"Missing_ChildPart_{matched_part}.pdf", f, 'application/pdf')}
#         data = {
#             'project': project,
#             'partNumber': part_number
#         }
#         response = requests.post(ZAPIER_WEBHOOK_URL, data=data, files=files)

#     os.remove(filename)

#     return {"status": "PDF sent", "zapier_status_code": response.status_code}


# --- Generate Missing Child Part PDF and upload directly to Glide Drawing Table ---

# @app.post("/generate-missing-childpart-pdf")
# async def generate_missing_childpart_pdf(request: Request):
#     """Generate Missing Child Part PDF and upload directly to Glide Drawing Table"""
#     try:
#         # Get project and part number from query params (like your old code)
#         url_params = dict(request.query_params)
#         project = url_params.get("project")
#         part_number = url_params.get("part")

#         payload = await request.json()
#         matched_part = payload.get("matchedPart")

#         if not matched_part or not project or not part_number:
#             return JSONResponse(
#                 status_code=400,
#                 content={"error": "Missing required fields: matchedPart, project, partNumber"}
#             )

#         # 1️⃣ Generate PDF in-memory
#         pdf = FPDF(orientation="L", unit="mm", format="A4")
#         pdf.add_page()
#         pdf.set_font("Arial", size=24)
#         pdf.cell(0, 20, txt="Missing Child Part Drawing", ln=True, align="C")
#         pdf.set_font("Arial", size=18)
#         pdf.cell(0, 15, txt=f"Part Number: {matched_part}", ln=True, align="C")

#         pdf_file_path = f"/tmp/{matched_part}.pdf"
#         pdf.output(pdf_file_path)

#         # 2️⃣ Prepare Glide mutation (Drawing table)
#         mutations = [
#             {
#                 "kind": "add-row-to-table",
#                 "tableName": GLIDE_TABLE,  # same table as fetch-drawings
#                 "columnValues": {
#                     "VQlMl": project,        # Project column
#                     "nlHAO": part_number,     # Part number column
#                     "9iB5E": "file"
#                 }
#             }
#         ]

#         url = "https://api.glideapp.io/api/function/mutateTables"
#         headers = {"Authorization": f"Bearer {GLIDE_API_KEY}"}

#         # 3️⃣ Send multipart request to Glide
#         with open(pdf_file_path, "rb") as file:
#             form_data = {
#                 "appID": (None, GLIDE_APP_ID),
#                 "mutations": (None, json.dumps(mutations)),
#                 "file": (f"{matched_part}.pdf", file, "application/pdf")
#             }

#             async with httpx.AsyncClient() as client:
#                 response = await client.post(url, headers=headers, files=form_data)
#                 response.raise_for_status()
#                 result = response.json()

#         os.remove(pdf_file_path)
        
#         return {
#             "success": True,
#             "message": "PDF uploaded directly to Glide Drawing Table",
#             "glide_response": result
#         }

#     except Exception as e:
#         import traceback
#         print("❌ Error in generate_missing_childpart_pdf:")
#         print(traceback.format_exc())
#         return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/generate-missing-childpart-pdf")
async def generate_missing_childpart_pdf(request: Request):
    """Generate Missing Child Part PDF, upload to Cloudinary, and save URL in Glide Drawing Table"""
    try:
        # Get project and part number from query params (like your old code)
        url_params = dict(request.query_params)
        project = url_params.get("project")
        part_number = url_params.get("part")

        payload = await request.json()
        matched_part = payload.get("matchedPart")
        matched_part = matched_part.replace(".", "") if matched_part else None

        if not matched_part or not project or not part_number:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required fields: matchedPart, project, partNumber"}
            )

        # 1️⃣ Generate PDF in-memory
        # pdf = FPDF(orientation="L", unit="mm", format="A4")
        # pdf.add_page()
        # pdf.set_font("Arial", size=24)
        # pdf.cell(0, 20, txt="Missing Child Part Drawing", ln=True, align="C")
        # pdf.set_font("Arial", size=18)
        # pdf.cell(0, 15, txt=f"Part Number: {matched_part}", ln=True, align="C")

        # pdf_file_path = f"/tmp/{matched_part}.pdf"
        # pdf.output(pdf_file_path)

        # 1️⃣ Generate PDF in-memory
        pdf = FPDF(orientation="L", unit="mm", format="A4")
        pdf.add_page()
        
        # Move down from top to center the content vertically
        pdf.ln(60)  # Move down 80mm from top (roughly center for landscape A4)
        
        # Title - larger font
        pdf.set_font("Arial", size=32)  # Increased from 24
        pdf.cell(0, 15, txt="Missing Drawing", ln=True, align="C")
        
        # Add some space between title and part number
        pdf.ln(5)  # 10mm gap
        
        # Part number - larger font  
        pdf.set_font("Arial", size=24)  # Increased from 18
        pdf.cell(0, 12, txt=f"Part Number: {matched_part}", ln=True, align="C")
        
        pdf_file_path = f"/tmp/{matched_part}.pdf"
        pdf.output(pdf_file_path)


        # 2️⃣ Upload PDF to Cloudinary
        try:
            with open(pdf_file_path, "rb") as file:
                upload_result = cloudinary.uploader.upload(
                    file,
                    resource_type="raw",               # Required for PDF
                    public_id=f"missing-pdfs/{matched_part}",  # File name will be included
                    overwrite=True
                )
                pdf_url = upload_result["secure_url"]
                print(f"✅ PDF uploaded to Cloudinary: {pdf_url}")
                
        except Exception as cloudinary_error:
            print(f"❌ Cloudinary upload failed: {cloudinary_error}")
            # Clean up temp file on Cloudinary error
            if os.path.exists(pdf_file_path):
                os.remove(pdf_file_path)
            raise Exception(f"Failed to upload PDF to Cloudinary: {str(cloudinary_error)}")

        # Clean up temp file after successful upload
        if os.path.exists(pdf_file_path):
            os.remove(pdf_file_path)

        # 3️⃣ Prepare Glide mutation (Drawing table) with Cloudinary URL
        mutations = [
            {
                "kind": "add-row-to-table",
                "tableName": GLIDE_TABLE,  # same table as fetch-drawings
                "columnValues": {
                    "VQlMl": project,         # Project column
                    "nlHAO": part_number,     # Part number column
                    "9iB5E": pdf_url          # Assuming 9iB5E is the drawing link column
                }
            }
        ]

        url = "https://api.glideapp.io/api/function/mutateTables"
        headers = {
            "Authorization": f"Bearer {GLIDE_API_KEY}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json={"appID": GLIDE_APP_ID, "mutations": mutations}
            )
            response.raise_for_status()
            result = response.json()

        return {
            "success": True,
            "message": "PDF uploaded to Cloudinary and URL saved in Glide Drawing Table",
            "pdf_url": pdf_url,
            "glide_response": result
        }

    except Exception as e:
        import traceback
        print("❌ Error in generate_missing_childpart_pdf:")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ocr-vision")
async def ocr_vision_endpoint(request: Request):
    """
    OpenAI Vision OCR endpoint with PaddleOCR fallback
    """
    try:
        form = await request.form()
        
        if "image" not in form:
            return JSONResponse(status_code=400, content={"error": "Missing image"})
        if "column" not in form:
            return JSONResponse(status_code=400, content={"error": "Missing column parameter"})
        
        image_file = form["image"]
        column_name = form["column"]
        
        logger.info(f"🔍 Vision OCR request for column: {column_name}")
        
        # Read image bytes
        image_bytes = await image_file.read()
        
        # Try OpenAI first
        try:
            def do_vision_ocr():
                return openai_extract_column(image_bytes, column_name)
            
            result = await asyncio.to_thread(do_vision_ocr)
            
            # Validate response
            if not result or len(result) == 0:
                logger.warning("⚠️ OpenAI returned empty result, falling back to PaddleOCR")
                raise Exception("Empty result from OpenAI")
            
            # Check for conversational response
            first_text = result[0]["text"].lower()
            conversational_indicators = ["how can i", "i can help", "please provide", "?", "sorry", "i cannot"]
            if any(indicator in first_text for indicator in conversational_indicators):
                logger.warning(f"⚠️ OpenAI returned conversational response: {first_text[:50]}, falling back to PaddleOCR")
                raise Exception("Conversational response detected")
            
            logger.info(f"✅ OpenAI extracted {len(result)} items for {column_name}")
            return {
                "mode": "vision",
                "column": column_name,
                "table": result
            }
            
        except Exception as openai_error:
            logger.warning(f"⚠️ OpenAI failed: {openai_error}, attempting PaddleOCR fallback")
            
            # Fallback to PaddleOCR
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            def do_paddle_ocr():
                return simple_cells(rgb)
            
            paddle_result = await asyncio.to_thread(do_paddle_ocr)
            
            logger.info(f"✅ PaddleOCR fallback extracted {len(paddle_result)} items for {column_name}")
            return {
                "mode": "paddleocr_fallback",
                "column": column_name,
                "table": paddle_result
            }
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Both OpenAI and PaddleOCR failed: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": f"OCR failed: {str(e)}"}
        )


@app.get("/debug")
async def debug():
    print("✅ /debug route hit", flush=True)
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
