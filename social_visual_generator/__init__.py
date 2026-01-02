"""
Social Visual Generator Core Package

A core library for generating social media infographics and carousel posts with captions using AI.
This package provides the core functionality that can be used in various applications.
"""

__version__ = "1.0.0"

# Import core functionality
from .agent import (
    SocialMediaContentGeneratorAgent,
    create_agent,
    generate_single_informational_image,
    generate_single_informational_image_from_text,
    generate_infographic_from_prompt,
    generate_infographic_with_reference_image,
    generate_carousel_from_prompt,
    generate_carousel_from_text,
    scrape_article_content,
    text_to_article_content,
    sanitize_filename,
)

__all__ = [
    "SocialMediaContentGeneratorAgent",
    "create_agent",
    "generate_single_informational_image",
    "generate_single_informational_image_from_text",
    "generate_infographic_from_prompt",
    "generate_infographic_with_reference_image",
    "generate_carousel_from_prompt",
    "generate_carousel_from_text",
    "scrape_article_content",
    "text_to_article_content",
    "sanitize_filename",
]
