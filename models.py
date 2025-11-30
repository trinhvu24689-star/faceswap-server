# models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, Date, DateTime
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    credits = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class FreeCreditLog(Base):
    __tablename__ = "free_credit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    claimed_date = Column(Date, index=True, nullable=False)
    amount = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)