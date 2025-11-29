from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Request
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import stripe
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

import cv2
import numpy as np
import io
import os
import uuid
import datetime as dt

from pydantic import BaseModel

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Float,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

import requests
from typing import Optional, List

# =================== FASTAPI APP ===================

app = FastAPI(title="FaceSwap AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_analyser = None
face_swapper = None

# tạo thư mục lưu ảnh nếu chưa có (cho lịch sử ảnh)
if not os.path.exists("saved"):
    os.makedirs("saved", exist_ok=True)

# mount static để xem ảnh đã lưu
app.mount("/saved", StaticFiles(directory="saved"), name="saved")

# =================== CREDIT / BILLING SYSTEM ===================

DATABASE_URL = "sqlite:///./faceswap.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

CREDIT_COST_PER_SWAP = 10  # mỗi lần swap trừ 10 điểm


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    credits = Column(Integer, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


# ====== LỊCH SỬ ẢNH SWAP ======

class SwapHistory(Base):
    __tablename__ = "swap_history"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    image_path = Column(String)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


# =================== STRIPE CONFIG ===================

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


# =================== CREDIT PACKAGES & ORDERS ===================

CREDIT_PACKAGES = {
    "pack_50": {
        "name": "Gói 50 điểm",
        "credits": 50,
        "amount": 50000,
    },
    "pack_200": {
        "name": "Gói 200 điểm",
        "credits": 200,
        "amount": 180000,
    },
    "pack_1000": {
        "name": "Gói 1000 điểm",
        "credits": 1000,
        "amount": 750000,
    },
}


class CreditOrder(Base):
    __tablename__ = "credit_orders"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    package_id = Column(String)
    package_name = Column(String)
    credits = Column(Integer)
    amount = Column(Integer)
    currency = Column(String, default="vnd")
    provider = Column(String, default="stripe")
    external_id = Column(String, nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=dt.datetime.utcnow)


# =================== DB DEPENDENCY ===================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =================== SCHEMAS ===================

class GuestCreateResponse(BaseModel):
    user_id: str
    credits: int


class CreditsResponse(BaseModel):
    credits: int


class CheckoutSessionCreate(BaseModel):
    package_id: str


class CheckoutSessionResponse(BaseModel):
    checkout_url: str


class FirebaseVerifyBody(BaseModel):
    id_token: str


# =================== STARTUP: LOAD MODELS + CREATE DB ===================

@app.on_event("startup")
async def load_models():
    global face_analyser, face_swapper
    try:
        # tạo bảng DB
        Base.metadata.create_all(bind=engine)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "models", "inswapper_128.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # ================================
        # CHẠY TRÊN RENDER (CPU MODE)
        # ================================
        providers = ["CPUExecutionProvider"]

        print("🔁 Loading FaceAnalysis model (CPU)…")
        face_analyser = FaceAnalysis(name="buffalo_l", providers=providers)
        face_analyser.prepare(ctx_id=-1, det_size=(640, 640))  # CPU MODE

        print("🔁 Loading FaceSwapper model (CPU)…")
        face_swapper = get_model(model_path, providers=providers)

        print("✅ AI Models loaded successfully (Render CPU Mode)!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")


# =================== AUTH / CREDITS API ===================

@app.post("/auth/guest", response_model=GuestCreateResponse)
def create_guest_user(db: Session = Depends(get_db)):
    user_id = str(uuid.uuid4())
    user = User(id=user_id, credits=5)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"user_id": user.id, "credits": user.credits}


@app.get("/credits", response_model=CreditsResponse)
def get_credits(
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    user = db.get(User, x_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"credits": user.credits}


@app.post("/credits/add-test", response_model=CreditsResponse)
def add_test_credits(
    amount: int = 10,
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    user = db.get(User, x_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.credits += amount
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"credits": user.credits}


# =================== PROFILE API ===================

@app.get("/profile")
def get_profile(
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    user = db.get(User, x_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "user_id": user.id,
        "credits": user.credits,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


# =================== STRIPE CHECKOUT (REAL) ===================

@app.post("/credits/checkout/stripe", response_model=CheckoutSessionResponse)
def create_stripe_checkout_session(
    payload: CheckoutSessionCreate,
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe chưa được cấu hình trên server")

    user = db.get(User, x_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    package = CREDIT_PACKAGES.get(payload.package_id)
    if not package:
        raise HTTPException(status_code=400, detail="Gói điểm không tồn tại")

    order_id = str(uuid.uuid4())

    order = CreditOrder(
        id=order_id,
        user_id=user.id,
        package_id=payload.package_id,
        package_name=package["name"],
        credits=package["credits"],
        amount=package["amount"],
        currency="vnd",
        provider="stripe",
        status="pending",
    )
    db.add(order)
    db.commit()

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "vnd",
                        "product_data": {"name": package["name"]},
                        "unit_amount": package["amount"],
                    },
                    "quantity": 1,
                }
            ],
            metadata={
                "order_id": order.id,
                "user_id": user.id,
            },
            success_url=f"{FRONTEND_URL}/?payment_success=1",
            cancel_url=f"{FRONTEND_URL}/?payment_cancel=1",
        )
    except Exception as e:
        order.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    order.external_id = session.id
    db.commit()

    return {"checkout_url": session.url}


# =================== STRIPE WEBHOOK (REAL) ===================

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured")

    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event["type"] == "checkout.session.completed":
        data = event["data"]["object"]
        meta = data.get("metadata", {})

        order_id = meta.get("order_id")

        order = db.get(CreditOrder, order_id)
        if order and order.status != "paid":
            user = db.get(User, order.user_id)
            if user:
                user.credits += order.credits
                db.add(user)

            order.status = "paid"
            db.add(order)
            db.commit()

    return {"received": True}


# =================== PAYMENT HISTORY ===================

@app.get("/payment/history")
def payment_history(
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    orders = (
        db.query(CreditOrder)
        .filter(CreditOrder.user_id == x_user_id)
        .order_by(CreditOrder.created_at.desc())
        .all()
    )
    return [
        {
            "id": o.id,
            "package_id": o.package_id,
            "package_name": o.package_name,
            "credits": o.credits,
            "amount": o.amount,
            "currency": o.currency,
            "status": o.status,
            "created_at": o.created_at.isoformat() if o.created_at else None,
        }
        for o in orders
    ]


# =================== OAUTH GOOGLE / FACEBOOK ===================

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI",
    "http://localhost:8000/oauth/google/callback",
)

FACEBOOK_CLIENT_ID = os.getenv("FACEBOOK_CLIENT_ID", "")
FACEBOOK_CLIENT_SECRET = os.getenv("FACEBOOK_CLIENT_SECRET", "")
FACEBOOK_REDIRECT_URI = os.getenv(
    "FACEBOOK_REDIRECT_URI",
    "http://localhost:8000/oauth/facebook/callback",
)


@app.get("/oauth/google/login")
def oauth_google_login():
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth chưa được cấu hình")
    url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&response_type=code"
        "&scope=openid%20email%20profile"
    )
    return RedirectResponse(url)


@app.get("/oauth/google/callback")
def oauth_google_callback(code: str):
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth chưa được cấu hình")

    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    r = requests.post(token_url, data=data)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Google login failed")

    token_data = r.json()
    id_token = token_data.get("id_token")
    access_token = token_data.get("access_token")

    return {
        "id_token": id_token,
        "access_token": access_token,
    }


@app.get("/oauth/facebook/login")
def oauth_facebook_login():
    if not FACEBOOK_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Facebook OAuth chưa được cấu hình")
    url = (
        "https://www.facebook.com/v20.0/dialog/oauth"
        f"?client_id={FACEBOOK_CLIENT_ID}"
        f"&redirect_uri={FACEBOOK_REDIRECT_URI}"
        "&scope=email,public_profile"
    )
    return RedirectResponse(url)


@app.get("/oauth/facebook/callback")
def oauth_facebook_callback(code: str):
    if not FACEBOOK_CLIENT_ID or not FACEBOOK_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Facebook OAuth chưa được cấu hình")

    token_url = (
        "https://graph.facebook.com/v20.0/oauth_access_token"
        f"?client_id={FACEBOOK_CLIENT_ID}"
        f"&redirect_uri={FACEBOOK_REDIRECT_URI}"
        f"&client_secret={FACEBOOK_CLIENT_SECRET}"
        f"&code={code}"
    )
    r = requests.get(token_url)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Facebook login failed")

    data = r.json()
    access_token = data.get("access_token")

    return {"access_token": access_token}


# =================== FIREBASE AUTH VERIFY ===================

@app.post("/auth/firebase/verify")
def firebase_verify(body: FirebaseVerifyBody):
    verify_url = "https://oauth2.googleapis.com/tokeninfo"
    resp = requests.get(verify_url, params={"id_token": body.id_token})

    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Invalid Firebase token")

    info = resp.json()
    return {
        "user_id": info.get("sub"),
        "email": info.get("email"),
    }


# =================== FACE SWAP API ===================

@app.post("/faceswap")
async def faceswap(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    if face_analyser is None or face_swapper is None:
        raise HTTPException(status_code=503, detail="AI models not ready")

    user = db.get(User, x_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.credits < CREDIT_COST_PER_SWAP:
        raise HTTPException(
            status_code=402,
            detail="Không đủ điểm tín dụng, vui lòng nạp thêm để sử dụng tính năng.",
        )

    # trừ điểm
    user.credits -= CREDIT_COST_PER_SWAP
    db.add(user)
    db.commit()
    db.refresh(user)

    try:
        source_contents = await source_image.read()
        target_contents = await target_image.read()

        source_np = np.frombuffer(source_contents, np.uint8)
        target_np = np.frombuffer(target_contents, np.uint8)

        source_img = cv2.imdecode(source_np, cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(target_np, cv2.IMREAD_COLOR)

        if source_img is None or target_img is None:
            raise HTTPException(status_code=400, detail="Invalid images")

        source_faces = face_analyser.get(source_img)
        target_faces = face_analyser.get(target_img)

        if not source_faces or not target_faces:
            raise HTTPException(status_code=400, detail="No faces detected")

        source_face = source_faces[0]
        target_face = target_faces[0]

        result_img = face_swapper.get(
            target_img,
            target_face,
            source_face,
            paste_back=True,
        )

        ok, buffer = cv2.imencode(".jpg", result_img)
        if not ok:
            raise HTTPException(
                status_code=500, detail="Encode result image failed"
            )

        file_name = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join("saved", file_name)
        with open(save_path, "wb") as f:
            f.write(buffer.tobytes())

        history = SwapHistory(
            id=str(uuid.uuid4()),
            user_id=user.id,
            image_path=file_name,
        )
        db.add(history)
        db.commit()

        io_buffer = io.BytesIO(buffer.tobytes())
        resp = StreamingResponse(io_buffer, media_type="image/jpeg")
        resp.headers["X-Credits-Remaining"] = str(user.credits)
        return resp

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Face swap error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =================== LỊCH SỬ ẢNH SWAP API ===================

@app.get("/swap/history")
def swap_history(
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(SwapHistory)
        .filter(SwapHistory.user_id == x_user_id)
        .order_by(SwapHistory.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "url": f"/saved/{r.image_path}",
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# =================== GLOBAL ERROR HANDLER (CHO ĐỠ 500 CÂM) ===================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("🔥 Unhandled error:", repr(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# =================== HEALTHCHECK ===================

@app.get("/")
async def root():
    return {"message": "🚀 FaceSwap AI Backend Ready!", "status": "OK"}