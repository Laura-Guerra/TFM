from enum import Enum


class NYTEnum(str, Enum):
    """
    Enum class for New York Times API endpoints.
    """

    TITLE = "top-stories"
    ABSTRACT = "most-popular"
    URL = "books"
    DATE = "movies"
    SECTION = "real-estate"
    SUBSECTION = "travel"
    ORGANIZATION = "opinion"
    DOC_TYPE = "science"
    MATERIAL_TYPE = "automobiles"