# Social Visual Generator Core

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Core library for generating social media infographics and carousel posts with captions using AI. This package provides the core functionality that can be integrated into your applications.

## Features

- **Infographic Generation**: Create beautiful infographics from text prompts or article URLs
- **Carousel Posts with Captions**: Generate multi-slide carousel posts with AI-generated captions
- **Multiple Content Types**: 
  - Infographics (list type or summary type)
  - Carousel posts from article URLs or text prompts
  - Single informational images
- **Text-Based Generation**: Generate content directly from article text (no URL scraping needed)
- **Image Orientation**: Choose between square (1080x1080) or vertical (1080x1350) orientations
- **Image Generation**: Supports multiple providers (OpenRouter/Gemini, OpenAI DALL-E/GPT-Image)
- **Reference Image Support**: Match existing design styles using reference images
- **Web Scraping**: Automatic article content extraction using newspaper4k
- **Design Customization**: Customize fonts, backgrounds, color schemes, and orientation
- **Background Processing**: Non-blocking image generation using background threads
- **Multi-Platform Captions**: Generate platform-specific captions for Facebook, Instagram, LinkedIn, Pinterest, and Reddit

## Installation

### From GitHub Packages

```bash
pip install social-visual-generator-core @ git+https://github.com/robort-gabriel/social-visual-generator-core.git
```

### From Source

```bash
git clone https://github.com/robort-gabriel/social-visual-generator-core.git
cd social-visual-generator-core
pip install -e .
```

## Quick Start

### Generate Carousel from URL

```python
from social_visual_generator import create_agent

# Create agent instance
agent = create_agent(
    openai_api_key="your-openai-key",
    openrouter_api_key="your-openrouter-key"
)

# Generate carousel from URL
result = await agent.process(
    url="https://example.com/article",
    max_slides=10,
    username="@yourhandle",
    tagline="daily tips",
    orientation="square"  # or "vertical"
)
```

### Generate Carousel from Text

```python
from social_visual_generator import generate_carousel_from_text

# Generate carousel directly from article text
result = await generate_carousel_from_text(
    article_text="Your article content here...",
    max_slides=5,
    username="@yourhandle",
    tagline="daily tips",
    orientation="square",  # or "vertical"
    enable_captions=True
)
```

### Generate Infographic from Prompt

```python
from social_visual_generator import generate_infographic_from_prompt

# Generate infographic from a text prompt
result = await generate_infographic_from_prompt(
    user_prompt="top 10 AI tools",
    username="@yourhandle",
    tagline="daily tips",
    orientation="square",
    enable_captions=True
)
```

## Key Features

### Image Orientation

Choose between **square** (1080x1080) or **vertical** (1080x1350) orientations:

```python
# Square orientation (default)
result = await generate_carousel_from_prompt(
    user_prompt="top 10 AI tools",
    orientation="square"
)

# Vertical orientation (more vertical space)
result = await generate_carousel_from_prompt(
    user_prompt="top 10 AI tools",
    orientation="vertical"
)
```

### Text-Based Generation

Generate content directly from article text without URL scraping:

```python
from social_visual_generator import generate_carousel_from_text

result = await generate_carousel_from_text(
    article_text="Your article content here...",
    max_slides=5,
    orientation="square"
)
```

### Background Image Generation

All image generation runs in **background threads** to prevent blocking:

- ✅ Non-blocking: Won't block your FastAPI event loop
- ✅ Concurrent: Multiple images generate in parallel (for carousels)
- ✅ Built-in: Uses Python's native `asyncio` - no external dependencies

### Caption Generation

Generate platform-specific captions for social media:

```python
from social_visual_generator import generate_carousel_from_prompt
from social_visual_generator.agent import INSTAGRAM, LINKEDIN

result = await generate_carousel_from_prompt(
    user_prompt="top 10 AI tools",
    enable_captions=True,
    enabled_platforms=[INSTAGRAM, LINKEDIN]
)

# Access captions
print(result["captions"]["instagram"])
print(result["captions"]["linkedin"])
```

**Supported Platforms:**
- Facebook Group & Page
- Instagram
- LinkedIn
- Pinterest
- Reddit

### Design Customization

Customize the appearance of your content:

```python
result = await generate_carousel_from_text(
    article_text="...",
    font_name="Arial",  # Custom font
    background_info="dark navy gradient",  # Background style
    color_schema="navy background with white and cyan accent text",  # Color scheme
    extra_instructions="Use minimalist design with bold typography"  # Additional instructions
)
```

## Available Functions

| Function | Description | Orientation | Captions | Background Threads |
|----------|-------------|-------------|----------|-------------------|
| `generate_carousel_from_text()` | Generate carousel from article text | ✅ | ✅ | ✅ (concurrent) |
| `generate_carousel_from_prompt()` | Generate carousel from text prompt | ✅ | ✅ | ✅ (concurrent) |
| `generate_single_informational_image_from_text()` | Single image from article text | ✅ | ✅ | ✅ |
| `generate_infographic_from_prompt()` | Generate infographic from prompt | ✅ | ✅ | ✅ |
| `generate_infographic_with_reference_image()` | Infographic with reference design | ✅ | ✅ | ✅ |
| `create_agent()` | Create agent for URL/text processing | ✅ | ✅ | ✅ (concurrent) |

## Requirements

- Python 3.11+
- OpenAI API key (for content generation)
- OpenRouter API key (for image generation, optional if using OpenAI)

## Documentation

Full documentation available at: [INSTALL_AND_USE_CORE.md](INSTALL_AND_USE_CORE.md)

For detailed usage examples and API reference, see the [Installation and Usage Guide](INSTALL_AND_USE_CORE.md).

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Support

For issues and questions, please open an issue on GitHub.
