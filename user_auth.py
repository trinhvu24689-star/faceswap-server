from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import sqlite3, os, uuid, random
from datetime import datetime

router = APIRouter()

DB_PATH = "users.db"
AVATAR_DIR = "avatars"
os.makedirs(AVATAR_DIR, exist_ok=True)

# ================= DB INIT =================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  email TEXT,
  password TEXT,
  vip TEXT DEFAULT 'FREE',
  avatar TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS otps (
  email TEXT,
  otp TEXT,
  created_at TEXT
)
""")

conn.commit()

# ================= MODELS =================
class RegisterModel(BaseModel):
  user_id: str
  email: str
  password: str

class LoginModel(BaseModel):
  email: str
  password: str

class OTPModel(BaseModel):
  email: str

class VerifyOTPModel(BaseModel):
  email: str
  otp: str

class DeleteModel(BaseModel):
  user_id: str


# ================= REGISTER =================
@router.post("/register")
def register(data: RegisterModel):
  cur.execute("SELECT * FROM users WHERE email=?", (data.email,))
  if cur.fetchone():
    raise HTTPException(400, "Email đã tồn tại")

  cur.execute(
    "INSERT INTO users (user_id, email, password, vip, avatar) VALUES (?, ?, ?, ?, ?)",
    (data.user_id, data.email, data.password, "FREE", "")
  )
  conn.commit()
  return {"status": "ok"}


# ================= LOGIN =================
@router.post("/login")
def login(data: LoginModel):
  cur.execute(
    "SELECT user_id FROM users WHERE email=? AND password=?",
    (data.email, data.password)
  )
  row = cur.fetchone()
  if not row:
    raise HTTPException(401, "Sai tài khoản")

  return {"user_id": row[0]}


# ================= SEND OTP =================
@router.post("/send-otp")
def send_otp(data: OTPModel):
  otp = str(random.randint(100000, 999999))
  cur.execute("DELETE FROM otps WHERE email=?", (data.email,))
  cur.execute(
    "INSERT INTO otps (email, otp, created_at) VALUES (?, ?, ?)",
    (data.email, otp, datetime.now().isoformat())
  )
  conn.commit()

  print("OTP:", otp)  # bé xem trực tiếp trong log fly.dev
  return {"sent": True}


# ================= VERIFY OTP =================
@router.post("/verify-otp")
def verify_otp(data: VerifyOTPModel):
  cur.execute(
    "SELECT otp FROM otps WHERE email=? ORDER BY created_at DESC LIMIT 1",
    (data.email,)
  )
  row = cur.fetchone()
  if not row or row[0] != data.otp:
    raise HTTPException(401, "OTP sai")

  return {"verified": True}


# ================= UPLOAD AVATAR =================
@router.post("/upload-avatar")
async def upload_avatar(user_id: str, avatar: UploadFile = File(...)):
  filename = f"{user_id}_{uuid.uuid4().hex}.png"
  path = os.path.join(AVATAR_DIR, filename)

  with open(path, "wb") as f:
    f.write(await avatar.read())

  cur.execute("UPDATE users SET avatar=? WHERE user_id=?", (f"/avatars/{filename}", user_id))
  conn.commit()

  return {"url": f"/avatars/{filename}"}


# ================= DELETE ACCOUNT =================
@router.post("/delete-account")
def delete_account(data: DeleteModel):
  cur.execute("DELETE FROM users WHERE user_id=?", (data.user_id,))
  conn.commit()
  return {"deleted": True}


# ================= PROFILE =================
@router.get("/me")
def get_me(user_id: str):
  cur.execute("SELECT user_id, vip, avatar FROM users WHERE user_id=?", (user_id,))
  row = cur.fetchone()
  if not row:
    return {}

  return {
    "user_id": row[0],
    "vip": row[1],
    "avatar": row[2] or f"/avatars/default.png"
  }