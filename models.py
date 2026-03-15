from enum import Enum
from pydantic import BaseModel, Field

class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

class DocumentElement(BaseModel):
    page_number: int | None = Field(None)
    type: ElementType = Field(...)
    content: str = Field(...)

class ParseResponse(BaseModel):
    filename: str
    provider_used: str
    total_elements: int
    elements: list[DocumentElement]
