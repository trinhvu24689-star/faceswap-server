# routers/credits.py
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session
from datetime import date
import random

from database import get_db
from models import User, FreeCreditLog

router = APIRouter()

@router.post("/credits/free/daily")
def claim_daily_free(
    x_user_id: str = Header(..., alias="x-user-id"),
    db: Session = Depends(get_db),
):
    # 1) create user if not exist
    user = db.query(User).filter(User.id == x_user_id).first()
    if not user:
        user = User(id=x_user_id, credits=0)
        db.add(user)
        db.commit()
        db.refresh(user)

    today = date.today()

    # 2) check if already claimed today
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
            status_code=400,
            detail="Hôm nay bạn đã nhận Bông Tuyết miễn phí rồi, quay lại vào ngày mai nha 💖",
        )

    # 3) random free amount (3–15)
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

    return {
        "added": added,
        "message": f"Đã tặng cho bạn {added}❄️ Bông Tuyết miễn phí hôm nay ✨",
    }