from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime

DATABASE_URL = "sqlite:///./swap.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class SwapHistoryModel(Base):
    __tablename__ = "swap_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    image_path = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_swap_db():
    Base.metadata.create_all(bind=engine)