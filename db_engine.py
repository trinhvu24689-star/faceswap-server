from sqlalchemy import create_engine, Column, Integer, String, DateTime, Date
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime as dt
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./swap.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SwapHistoryModel(Base):
    __tablename__ = "swap_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    file_name = Column(String)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


def init_swap_db():
    Base.metadata.create_all(bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    credits = Column(Integer, default=0)

class CreditOrder(Base):
    __tablename__ = "credit_orders"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    package_id = Column(String)
    package_name = Column(String)
    credits = Column(Integer)
    amount = Column(Integer)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=dt.datetime.utcnow)

class FreeCreditLog(Base):
    __tablename__ = "free_credit_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    claimed_date = Column(Date)
    amount = Column(Integer)