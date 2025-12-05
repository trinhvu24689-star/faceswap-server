import os
import uuid
import datetime as dt
import io
import random
import requests
import insightface
import numpy as np
import cv2

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Header,
    Depends,
    Request,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse

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

from db_engine import SessionLocal as SwapSessionLocal, SwapHistoryModel, init_swap_db
from rate_limit import check_rate_limit
from routers.system_router import router as system_router

# ❌ BỎ DÒNG NÀY (NGUYÊN NHÂN 504)
# from insightface.app import FaceAnalysis


# =================== FASTAPI APP ===================

app = FastAPI(title="FaceSwap AI Backend", version="1.0.0")

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


# ✅ ===================== LAZY LOAD MODEL (DÁN TẠI ĐÂY) =====================

face_analyser = None
swapper = None
model_loaded = False

def load_model_once():
    global face_analyser, swapper, model_loaded
    if model_loaded:
        return

    print(">>> LOADING FACE MODEL ...")

    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    face_analyser = FaceAnalysis(name="buffalo_l", root="./models")
    face_analyser.prepare(ctx_id=0)

    swapper = get_model("./models/inswapper_128.onnx", download=False)

    model_loaded = True
    print(">>> MODEL LOADED SUCCESSFULLY")


# =================== DATABASE ===================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./faceswap.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

CREDIT_COST_PER_SWAP = 10
VIDEO_CREDITS_PER_30S = 15


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
    claimed_date = Column(Date, index=True, nullable=False)
    amount = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


def calculate_video_credits(duration_seconds: int) -> int:
    if duration_seconds <= 0:
        return 0
    blocks = (duration_seconds + 29) // 30
    return blocks * VIDEO_CREDITS_PER_30S


def charge_credits_for_video(db: Session, user_id: str, duration_seconds: int):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")

    cost = calculate_video_credits(duration_seconds)

    if cost > 0:
        if user.credits < cost:
            raise HTTPException(
                402,
                "Không đủ điểm tín dụng, vui lòng nạp thêm để sử dụng tính năng.",
            )

        user.credits -= cost
        db.add(user)
        db.commit()
        db.refresh(user)

    return {
        "credits_charged": cost,
        "credits_left": user.credits,
    }


# =================== STRIPE CONFIG ===================

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

stripe = None
if STRIPE_SECRET_KEY:
    import stripe as stripe_lib

    stripe_lib.api_key = STRIPE_SECRET_KEY
    stripe = stripe_lib


CREDIT_PACKAGES = {
    "pack_50": {"name": "Gói 50 điểm", "credits": 50, "amount": 50000},
    "pack_200": {"name": "Gói 200 điểm", "credits": 200, "amount": 180000},
    "pack_1000": {"name": "Gói 1000 điểm", "credits": 1000, "amount": 750000},
    "pack_36": {"name": "Gói 36❄️", "credits": 36, "amount": 26000},
    "pack_70": {"name": "Gói 70❄️", "credits": 70, "amount": 52000},
    "pack_150": {"name": "Gói 150❄️", "credits": 150, "amount": 125000},
    "pack_200": {"name": "Gói 200❄️", "credits": 200, "amount": 185000},
    "pack_400": {"name": "Gói 400❄️", "credits": 400, "amount": 230000},
    "pack_550": {"name": "Gói 550❄️", "credits": 550, "amount": 375000},
    "pack_750": {"name": "Gói 750❄️", "credits": 750, "amount": 510000},
    "pack_999": {"name": "Gói 999❄️", "credits": 999, "amount": 760000},
    "pack_1500": {"name": "Gói 1.500❄️", "credits": 1500, "amount": 1050000},
    "pack_2600": {"name": "Gói 2.600❄️", "credits": 2600, "amount": 1500000},
    "pack_4000": {"name": "Gói 4.000❄️", "credits": 4000, "amount": 2400000},
    "pack_7600": {"name": "Gói 7.600❄️", "credits": 7600, "amount": 3600000},
    "pack_10000": {"name": "Gói 10.000❄️", "credits": 10000, "amount": 5000000},
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


# =================== DB DEP ===================


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


# =================== STARTUP + MIDDLEWARE ===================

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    init_swap_db()
    start_cleanup_thread()
    start_keep_alive()


@app.middleware("http")
async def activity_middleware(request: Request, call_next):
    mark_activity()
    return await call_next(request)


# =================== AUTH / CREDITS API ===================

@app.post("/auth/guest", response_model=GuestCreateResponse)
def create_guest_user(db: Session = Depends(get_db)):
    user = User(id=str(uuid.uuid4()), credits=5)
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
        raise HTTPException(404, "User not found")
    return {"credits": user.credits}


@app.post("/credits/add-test", response_model=CreditsResponse)
def add_test_credits(
    amount: int = 10,
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    user = db.get(User, x_user_id)
    if not user:
        raise HTTPException(404, "User not found")

    user.credits += amount
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"credits": user.credits}


@app.post("/credits/free/daily")
def claim_daily_free(
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    today = dt.date.today()

    user = db.get(User, x_user_id)
    if not user:
        user = User(id=x_user_id, credits=0)
        db.add(user)
        db.commit()
        db.refresh(user)

    existed = (
        db.query(FreeCreditLog)
        .filter(
            FreeCreditLog.user_id == x_user_id,
            FreeCreditLog.claimed_date == today,
        )
        .first()
    )
    if existed:
        raise HTTPException(
            400,
            "Hôm nay bạn đã nhận Bông Tuyết miễn phí rồi, quay lại vào ngày mai nha 💖",
        )

    added = random.randint(3, 15)

    user.credits += added
    db.add(user)

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


@app.post("/credits/video")
def deduct_video_credits(
    duration_seconds: int = Form(...),
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    r = charge_credits_for_video(db, x_user_id, duration_seconds)
    return {
        "duration_seconds": duration_seconds,
        "credits_charged": r["credits_charged"],
        "credits_left": r["credits_left"],
    }


# =================== STRIPE CHECKOUT ===================

@app.post("/credits/checkout/stripe", response_model=CheckoutSessionResponse)
def create_stripe_checkout_session(
    payload: CheckoutSessionCreate,
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    if not stripe:
        raise HTTPException(500, "Stripe chưa được cấu hình trên server")

    user = db.get(User, x_user_id)
    if not user:
        raise HTTPException(404, "User not found")

    package = CREDIT_PACKAGES.get(payload.package_id)
    if not package:
        raise HTTPException(400, "Gói điểm không tồn tại")

    order = CreditOrder(
        id=str(uuid.uuid4()),
        user_id=user.id,
        package_id=payload.package_id,
        package_name=package["name"],
        credits=package["credits"],
        amount=package["amount"],
    )
    db.add(order)
    db.commit()

    sess = stripe.checkout.Session.create(
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

    order.external_id = sess.id
    db.commit()

    return {"checkout_url": sess.url}


@app.post("/stripe/webhook")
async def stripe_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    if not stripe or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(500, "Stripe webhook secret not configured")

    payload = await request.body()
    sig = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except Exception:
        raise HTTPException(400, "Invalid signature")

    if event["type"] == "checkout.session.completed":
        data = event["data"]["object"]
        meta = data.get("metadata", {})

        order = db.get(CreditOrder, meta.get("order_id"))
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
    rows = (
        db.query(CreditOrder)
        .filter(CreditOrder.user_id == x_user_id)
        .order_by(CreditOrder.created_at.desc())
        .all()
    )

    return [
        {
            "id": r.id,
            "package_id": r.package_id,
            "package_name": r.package_name,
            "credits": r.credits,
            "amount": r.amount,
            "currency": r.currency,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# =================== FIREBASE VERIFY ===================

@app.post("/auth/firebase/verify")
def firebase_verify(body: FirebaseVerifyBody):
    resp = requests.get(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"id_token": body.id_token},
    )

    if resp.status_code != 200:
        raise HTTPException(400, "Invalid Firebase token")

    info = resp.json()
    return {
        "user_id": info.get("sub"),
        "email": info.get("email"),
    }


# =================== FULL AI MODEL (XOÁ LIGHT MODE) ===================

MODEL_ROOT = "./models"

face_app = FaceAnalysis(name="buffalo_l", root=MODEL_ROOT)
face_app.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model("inswapper_128", root=MODEL_ROOT)
swapper.prepare(ctx_id=0)

print("✅ Full AI FaceSwap Model Loaded!")


# =================== FACESWAP APIs ===================

@app.post("/faceswap")
async def faceswap_light(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    if not check_rate_limit(x_user_id):
        raise HTTPException(
            429,
            "Bạn thao tác quá nhanh, vui lòng thử lại sau 😭",
        )

    u = db.get(User, x_user_id)
    if not u:
        raise HTTPException(404, "User not found")

    if u.credits < CREDIT_COST_PER_SWAP:
        raise HTTPException(
            402,
            "Không đủ điểm tín dụng, vui lòng nạp thêm để sử dụng tính năng.",
        )

    u.credits -= CREDIT_COST_PER_SWAP
    db.add(u)
    db.commit()
    db.refresh(u)

    target_bytes = await target_image.read()

    file_name = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join("saved", file_name)
    with open(save_path, "wb") as f:
        f.write(target_bytes)

    history = SwapHistory(
        id=str(uuid.uuid4()),
        user_id=u.id,
        image_path=file_name,
    )
    db.add(history)
    db.commit()

    resp = StreamingResponse(
        io.BytesIO(target_bytes),
        media_type=target_image.content_type or "image/jpeg",
    )
    resp.headers["X-Credits-Remaining"] = str(u.credits)
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


@app.post("/faceswap/full")
async def faceswap_full(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    if not check_rate_limit(x_user_id):
        raise HTTPException(
            429,
            "Bạn thao tác quá nhanh, vui lòng thử lại sau 😭",
        )

    u = db.get(User, x_user_id)
    if not u:
        raise HTTPException(404, "User not found")

    if u.credits < CREDIT_COST_PER_SWAP:
        raise HTTPException(
            402,
            "Không đủ điểm tín dụng, vui lòng nạp thêm để sử dụng tính năng.",
        )

    u.credits -= CREDIT_COST_PER_SWAP
    db.add(u)
    db.commit()
    db.refresh(u)

    src = cv2.imdecode(
        np.frombuffer(await source_image.read(), np.uint8),
        cv2.IMREAD_COLOR,
    )
    tgt = cv2.imdecode(
        np.frombuffer(await target_image.read(), np.uint8),
        cv2.IMREAD_COLOR,
    )

    src_faces = face_app.get(src)
    tgt_faces = face_app.get(tgt)

    if not src_faces:
        raise HTTPException(400, "Không tìm thấy khuôn mặt trong ảnh gốc")

    if not tgt_faces:
        raise HTTPException(400, "Không tìm thấy khuôn mặt trong ảnh target")

    result = swapper.get(
        tgt,
        tgt_faces[0],
        src_faces[0],
        paste_back=True,
    )

    ok, out_img = cv2.imencode(".jpg", result)
    if not ok:
        raise HTTPException(500, "Không encode được ảnh kết quả")

    out_bytes = out_img.tobytes()

    file_name = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join("saved", file_name)
    with open(save_path, "wb") as f:
        f.write(out_bytes)

    history = SwapHistory(
        id=str(uuid.uuid4()),
        user_id=u.id,
        image_path=file_name,
    )
    db.add(history)
    db.commit()

    resp = StreamingResponse(
        io.BytesIO(out_bytes),
        media_type="image/jpeg",
    )
    resp.headers["X-Credits-Remaining"] = str(u.credits)
    return resp


# =================== GLOBAL ERROR HANDLER ===================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("🔥 Unhandled error:", repr(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# =================== ROUTERS + HEALTH ===================

app.include_router(video_router)
app.include_router(system_router)


@app.get("/ping")
def ping():
    return {"status": "alive"}


@app.get("/")
async def root():
    return {
        "message": "🚀 FaceSwap AI Backend Ready!",
        "status": "OK",
    }