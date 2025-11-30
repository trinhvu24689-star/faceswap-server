import os
import uuid
import datetime as dt
import io
import random

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Header,
    Depends,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse

from pydantic import BaseModel
from typing import List

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Date,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

import requests

# =================== FASTAPI APP ===================

app = FastAPI(title="FaceSwap AI Backend (Light Mode)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# thư mục lưu ảnh lịch sử
if not os.path.exists("saved"):
    os.makedirs("saved", exist_ok=True)

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


class SwapHistory(Base):
    __tablename__ = "swap_history"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    image_path = Column(String)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class FreeCreditLog(Base):
    __tablename__ = "free_credit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    claimed_date = Column(Date, index=True, nullable=False)  # ngày nhận free
    amount = Column(Integer, nullable=False)                 # số Bông Tuyết free
    created_at = Column(DateTime, default=dt.datetime.utcnow)

# =================== STRIPE CONFIG (OPTIONAL) ===================

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

if STRIPE_SECRET_KEY:
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
else:
    stripe = None


# ====== CREDIT PACKAGES (THÊM ĐỦ BACKEND CHO PACKS FRONTEND) ======
CREDIT_PACKAGES = {
    # mấy gói cũ (giữ nguyên không đụng tới)
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

    # các gói mới khớp với backendId ở frontend
    "pack_36": {
        "name": "Gói 36❄️",
        "credits": 36,
        "amount": 26000,
    },
    "pack_70": {
        "name": "Gói 70❄️",
        "credits": 70,
        "amount": 52000,
    },
    "pack_150": {
        "name": "Gói 150❄️",
        "credits": 150,
        "amount": 125000,
    },
    "pack_200": {  # giữ id pack_200 vừa cũ vừa mới, amount theo shop
        "name": "Gói 200❄️",
        "credits": 200,
        "amount": 185000,
    },
    "pack_400": {
        "name": "Gói 400❄️",
        "credits": 400,
        "amount": 230000,
    },
    "pack_550": {
        "name": "Gói 550❄️",
        "credits": 550,
        "amount": 375000,
    },
    "pack_750": {
        "name": "Gói 750❄️",
        "credits": 750,
        "amount": 510000,
    },
    "pack_999": {
        "name": "Gói 999❄️",
        "credits": 999,
        "amount": 760000,
    },
    "pack_1500": {
        "name": "Gói 1.500❄️",
        "credits": 1500,
        "amount": 1050000,
    },
    "pack_2600": {
        "name": "Gói 2.600❄️",
        "credits": 2600,
        "amount": 1500000,
    },
    "pack_4000": {
        "name": "Gói 4.000❄️",
        "credits": 4000,
        "amount": 2400000,
    },
    "pack_7600": {
        "name": "Gói 7.600❄️",
        "credits": 7600,
        "amount": 3600000,
    },
    "pack_10000": {
        "name": "Gói 10.000❄️",
        "credits": 10000,
        "amount": 5000000,
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


# =================== STARTUP ===================

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    print("✅ Database ready (LIGHT MODE). No AI models loaded.")


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


@app.post("/credits/free/daily")
def claim_daily_free(
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    today = dt.date.today()

    # lấy / tạo user (chỉ dùng field có trong model, không gắn field lạ)
    user = db.get(User, x_user_id)
    if not user:
        user = User(id=x_user_id, credits=0)
        db.add(user)
        db.commit()
        db.refresh(user)

    # kiểm tra FreeCreditLog xem hôm nay đã nhận chưa
    existing = (
        db.query(FreeCreditLog)
        .filter(
            FreeCreditLog.user_id == x_user_id,
            FreeCreditLog.claimed_date == today,
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Hôm nay bạn đã nhận Bông Tuyết miễn phí rồi, quay lại vào ngày mai nha 💖",
        )

    # random số free hôm nay
    added = random.randint(3, 15)

    # cộng credits vào user
    user.credits += added
    db.add(user)

    # log lại
    log = FreeCreditLog(
        user_id=x_user_id,
        claimed_date=today,
        amount=added,
    )
    db.add(log)
    db.commit()
    db.refresh(user)

    return {
        "added": added,
        "message": f"Hôm nay bạn nhận được {added}❄️ Bông Tuyết miễn phí ✨ (không sử dụng sẽ mất khi sang ngày mới)",
    }

# =================== STRIPE CHECKOUT (nếu có cấu hình) ===================

@app.post("/credits/checkout/stripe", response_model=CheckoutSessionResponse)
def create_stripe_checkout_session(
    payload: CheckoutSessionCreate,
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    if not stripe or not STRIPE_SECRET_KEY:
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
        import stripe as stripe_lib
        session = stripe_lib.checkout.Session.create(
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


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    if not stripe or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured")

    import stripe as stripe_lib

    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe_lib.Webhook.construct_event(
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


# =================== FIREBASE AUTH VERIFY (OPTION) ===================

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


# =================== FACE SWAP (LIGHT) ===================

@app.post("/faceswap")
async def faceswap_light(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    """
    Bản LIGHT:
    - Không dùng insightface.
    - Trừ 10 credits.
    - Lưu history.
    - Trả lại chính ảnh target.
    """

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

    # đọc ảnh target và lưu lại
    target_bytes = await target_image.read()

    file_name = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join("saved", file_name)
    with open(save_path, "wb") as f:
        f.write(target_bytes)

    history = SwapHistory(
        id=str(uuid.uuid4()),
        user_id=user.id,
        image_path=file_name,
    )
    db.add(history)
    db.commit()

    io_buffer = io.BytesIO(target_bytes)
    resp = StreamingResponse(
        io_buffer,
        media_type=target_image.content_type or "image/jpeg",
    )
    resp.headers["X-Credits-Remaining"] = str(user.credits)
    return resp


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


# =================== GLOBAL ERROR HANDLER ===================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("🔥 Unhandled error:", repr(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error (light mode)"},
    )


# =================== HEALTHCHECK ===================

@app.get("/")
async def root():
    return {"message": "🚀 FaceSwap AI Backend Ready! (light mode)", "status": "OK"}