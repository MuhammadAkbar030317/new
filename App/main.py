# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.app import predict_cancellation

app = FastAPI(title="Hotel Cancellation API")


class BookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: float
    company: float
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    city: str


@app.get("/")
def health():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: BookingInput):
    return predict_cancellation(data.dict())
