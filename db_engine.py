from sqlalchemy import create_engine, Column, Integer, String, DateTime
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