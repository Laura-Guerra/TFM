# schema for nyt params using pydantic
from pydantic import BaseModel, Field
class NYTParams(BaseModel):
    """
    Schema for NYT API parameters.
    """
    year: int = Field(..., description="Year for search")
    month: int = Field(..., description="Month for search")
    api_key: str = Field(..., description="API key for NYT API")


class NYTResponse(BaseModel):
    """
    Schema for NYT API response.
    """
    articles: list[dict] = Field(..., description="List of articles")
    keywords: list[dict] = Field(..., description="List of keywords")