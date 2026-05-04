from sqlalchemy import Column, Integer, String, Text, JSON
from .db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(128), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=True)
    preferences = Column(JSON, nullable=True)

class Trip(Base):
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    destination = Column(String(256), nullable=False)
    details = Column(JSON, nullable=False)

class TripLog(Base):
    __tablename__ = "trip_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    event = Column(String(256), nullable=False)
    metadata = Column(JSON, nullable=True)
