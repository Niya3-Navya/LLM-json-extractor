"""Pydantic schemas: the strict contract the LLM output must satisfy."""
import re
from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    # extra="forbid": unexpected keys from the LLM are a validation error.
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------- Invoice
class LineItem(_Strict):
    description: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    amount: Optional[float] = Field(default=None, ge=0, description="quantity x unit_price")


class Invoice(_Strict):
    invoice_number: str = Field(min_length=1)
    vendor_name: str = Field(min_length=1)
    vendor_address: Optional[str] = None
    customer_name: Optional[str] = None
    invoice_date: date
    due_date: Optional[date] = None
    currency: Optional[str] = Field(default=None, description="ISO 4217 code, e.g. USD")
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Optional[float] = Field(default=None, ge=0)
    discount: Optional[float] = Field(default=None, ge=0, description="positive amount deducted")
    shipping: Optional[float] = Field(default=None, ge=0)
    tax: Optional[float] = Field(default=None, ge=0)
    total: float = Field(ge=0)

    @field_validator("invoice_number", mode="before")
    @classmethod
    def _number_to_str(cls, v):
        return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v

    @field_validator("currency")
    @classmethod
    def _currency_code(cls, v):
        if v is None:
            return v
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", v):
            raise ValueError("currency must be a 3-letter ISO code like USD")
        return v


def invoice_warnings(inv: Invoice, tol: float = 0.01) -> list[str]:
    """Soft arithmetic checks. Returned as warnings, they do not fail validation
    (real invoices have rounding, extra fees...)."""
    warnings = []
    amounts = []
    for i, item in enumerate(inv.line_items, 1):
        expected = round(item.quantity * item.unit_price, 2)
        if item.amount is not None:
            amounts.append(item.amount)
            if abs(expected - item.amount) > tol:
                warnings.append(f"line {i}: quantity x unit_price = {expected}, but amount = {item.amount}")
    if inv.subtotal is not None and len(amounts) == len(inv.line_items):
        if abs(sum(amounts) - inv.subtotal) > tol:
            warnings.append(f"sum of line amounts = {round(sum(amounts), 2)}, but subtotal = {inv.subtotal}")
    if inv.subtotal is not None:
        expected_total = round(
            inv.subtotal - (inv.discount or 0) + (inv.shipping or 0) + (inv.tax or 0), 2
        )
        if abs(expected_total - inv.total) > tol:
            warnings.append(
                f"subtotal - discount + shipping + tax = {expected_total}, but total = {inv.total}"
            )
    return warnings


# ------------------------------------------------------------------ Resume
_MONTH = r"^\d{4}(-(0[1-9]|1[0-2]))?$"  # YYYY or YYYY-MM


class Experience(_Strict):
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start_date: Optional[str] = Field(default=None, description="YYYY-MM or YYYY")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM, YYYY or 'Present'")
    description: Optional[str] = None

    @field_validator("start_date", "end_date")
    @classmethod
    def _date_format(cls, v):
        if v is None or v == "Present":
            return v
        if not re.fullmatch(_MONTH, v):
            raise ValueError("date must be YYYY-MM, YYYY, or 'Present'")
        return v


class Education(_Strict):
    institution: str = Field(min_length=1)
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    graduation_year: Optional[int] = Field(default=None, ge=1950, le=2100)


class Resume(_Strict):
    full_name: str = Field(min_length=1)
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    @field_validator("email")
    @classmethod
    def _email_format(cls, v):
        if v is None:
            return v
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("email is not a valid address")
        return v


# Registry used by the CLI: document type -> schema
SCHEMAS = {"invoice": Invoice, "resume": Resume}
