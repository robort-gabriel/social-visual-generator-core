# Install and Use Core Package in Your FastAPI App

Simple guide to add `social-visual-generator-core` to your existing FastAPI application.

## Installation

### 1. Add to `requirements.txt`

```txt
social-visual-generator-core @ git+https://github.com/robort-gabriel/social-visual-generator-core.git
```

### 2. Install

```bash
pip install -r requirements.txt
```

## Usage

### Import Functions

```python
from social_visual_generator import (
    create_agent,
    generate_infographic_from_prompt,
    generate_carousel_from_prompt,
    generate_carousel_from_text,
    generate_single_informational_image,
    generate_single_informational_image_from_text,
    generate_infographic_with_reference_image,
)
from social_visual_generator.agent import (
    ALL_PLATFORMS,
    FACEBOOK_GROUP,
    FACEBOOK_PAGE,
    INSTAGRAM,
    LINKEDIN,
    PINTEREST,
    REDDIT,
)
```

### Example: Generate Infographic from Prompt

```python
from fastapi import FastAPI
from pydantic import BaseModel
from social_visual_generator import generate_infographic_from_prompt

app = FastAPI()

# Your API keys (from your config/database/etc.)
OPENAI_API_KEY = "your-openai-key"
OPENROUTER_API_KEY = "your-openrouter-key"  # Optional

class InfographicRequest(BaseModel):
    prompt: str
    username: str = "@test"
    tagline: str = "test tagline"
    enable_captions: bool = False
    enabled_platforms: list[str] = None  # Optional: ["instagram", "linkedin", ...]

@app.post("/api/v1/generate-infographic")
async def create_infographic(request: InfographicRequest):
    result = await generate_infographic_from_prompt(
        user_prompt=request.prompt,
        openai_api_key=OPENAI_API_KEY,
        openrouter_api_key=OPENROUTER_API_KEY,
        username=request.username,
        tagline=request.tagline,
        enable_captions=request.enable_captions,
        enabled_platforms=request.enabled_platforms or ALL_PLATFORMS if request.enable_captions else None,
    )
    return result
```

### Example: Generate Carousel from URL

```python
from social_visual_generator import create_agent

# Your API keys (from your config/database/etc.)
OPENAI_API_KEY = "your-openai-key"
OPENROUTER_API_KEY = "your-openrouter-key"  # Optional

@app.post("/api/v1/generate-carousel")
async def generate_carousel(
    url: str, 
    max_slides: int = 5,
    enable_captions: bool = False,
    enabled_platforms: list[str] = None
):
    agent = create_agent(
        openai_api_key=OPENAI_API_KEY,
        openrouter_api_key=OPENROUTER_API_KEY,
    )
    result = await agent.process(
        url=url,
        max_slides=max_slides,
        username="@test",
        tagline="test tagline",
        enable_captions=enable_captions,
        enabled_platforms=enabled_platforms or ALL_PLATFORMS if enable_captions else None,
    )
    return result
```

### Example: Generate Carousel from Text (using agent)

```python
from social_visual_generator import create_agent

# Your API keys (from your config/database/etc.)
OPENAI_API_KEY = "your-openai-key"
OPENROUTER_API_KEY = "your-openrouter-key"  # Optional

@app.post("/api/v1/generate-carousel-from-text-agent")
async def generate_carousel_from_text_agent(
    article_text: str, 
    max_slides: int = 5,
    title: str = None,
    enable_captions: bool = False,
    enabled_platforms: list[str] = None
):
    agent = create_agent(
        openai_api_key=OPENAI_API_KEY,
        openrouter_api_key=OPENROUTER_API_KEY,
    )
    result = await agent.process_from_text(
        article_text=article_text,
        max_slides=max_slides,
        title=title,  # Optional: override extracted title
        username="@test",
        tagline="test tagline",
        enable_captions=enable_captions,
        enabled_platforms=enabled_platforms or ALL_PLATFORMS if enable_captions else None,
    )
    return result
```

### Example: Generate Carousel from Prompt

```python
from social_visual_generator import generate_carousel_from_prompt

# Your API keys (from your config/database/etc.)
OPENAI_API_KEY = "your-openai-key"
OPENROUTER_API_KEY = "your-openrouter-key"  # Optional

@app.post("/api/v1/generate-carousel-from-prompt")
async def create_carousel(
    prompt: str, 
    max_slides: int = 5,
    enable_captions: bool = False,
    enabled_platforms: list[str] = None
):
    result = await generate_carousel_from_prompt(
        user_prompt=prompt,
        max_slides=max_slides,
        openai_api_key=OPENAI_API_KEY,
        openrouter_api_key=OPENROUTER_API_KEY,
        username="@test",
        tagline="test tagline",
        enable_captions=enable_captions,
        enabled_platforms=enabled_platforms or ALL_PLATFORMS if enable_captions else None,
    )
    return result
```

### Example: Generate Carousel from Article Text

```python
from social_visual_generator import generate_carousel_from_text

# Your API keys (from your config/database/etc.)
OPENAI_API_KEY = "your-openai-key"
OPENROUTER_API_KEY = "your-openrouter-key"  # Optional

@app.post("/api/v1/generate-carousel-from-text")
async def create_carousel_from_text(
    article_text: str, 
    max_slides: int = 5,
    title: str = None,
    enable_captions: bool = False,
    enabled_platforms: list[str] = None
):
    result = await generate_carousel_from_text(
        article_text=article_text,
        max_slides=max_slides,
        title=title,  # Optional: override extracted title
        openai_api_key=OPENAI_API_KEY,
        openrouter_api_key=OPENROUTER_API_KEY,
        username="@test",
        tagline="test tagline",
        enable_captions=enable_captions,
        enabled_platforms=enabled_platforms or ALL_PLATFORMS if enable_captions else None,
    )
    return result
```

### Example: Generate Single Image from Article Text

```python
from social_visual_generator import generate_single_informational_image_from_text

# Your API keys (from your config/database/etc.)
OPENAI_API_KEY = "your-openai-key"
OPENROUTER_API_KEY = "your-openrouter-key"  # Optional

@app.post("/api/v1/generate-single-image-from-text")
async def create_single_image_from_text(
    article_text: str,
    title: str = None,
    enable_captions: bool = False,
    enabled_platforms: list[str] = None
):
    result = await generate_single_informational_image_from_text(
        article_text=article_text,
        title=title,  # Optional: override extracted title
        openai_api_key=OPENAI_API_KEY,
        openrouter_api_key=OPENROUTER_API_KEY,
        username="@test",
        tagline="test tagline",
        enable_captions=enable_captions,
        enabled_platforms=enabled_platforms or ALL_PLATFORMS if enable_captions else None,
    )
    return result
```

### Example: Generate with Caption Generation

All generation functions support caption generation for social media platforms:

```python
from social_visual_generator import generate_carousel_from_prompt
from social_visual_generator.agent import ALL_PLATFORMS, INSTAGRAM, LINKEDIN

# Generate carousel with captions for Instagram and LinkedIn
result = await generate_carousel_from_prompt(
    user_prompt="top 10 AI tools",
    max_slides=5,
    openai_api_key=OPENAI_API_KEY,
    enable_captions=True,
    enabled_platforms=[INSTAGRAM, LINKEDIN],  # Or use ALL_PLATFORMS for all
)

# Captions are returned in the result and saved to captions.json
print(result["captions"])
# {
#     "instagram": "🎨 Check out these top 10 AI tools...",
#     "linkedin": "Discover the top 10 AI tools that are transforming..."
# }
```

**Supported Platforms:**
- `FACEBOOK_GROUP` - Facebook Groups
- `FACEBOOK_PAGE` - Facebook Pages
- `INSTAGRAM` - Instagram posts
- `LINKEDIN` - LinkedIn posts
- `PINTEREST` - Pinterest pins
- `REDDIT` - Reddit posts
- `ALL_PLATFORMS` - All platforms (default when enabled)

## Performance Features

### Background Image Generation

All image generation runs in **background threads** to prevent blocking your FastAPI event loop. This means:

✅ **Non-blocking**: Image generation won't block other API requests  
✅ **Concurrent execution**: Multiple images generate in parallel (for carousels)  
✅ **Built-in**: Uses Python's native `asyncio` - no external dependencies needed  
✅ **Production-ready**: Handles errors gracefully and maintains slide order

**How it works:**
- Single images: Generated in a background thread using `asyncio.to_thread()`
- Carousel images: All slides generate concurrently using `asyncio.gather()`
- No configuration needed: Works automatically with all generation functions

**Example:**
```python
# This won't block your FastAPI event loop
# Multiple images generate concurrently in background threads
result = await generate_carousel_from_prompt(
    user_prompt="top 10 AI tools",
    max_slides=10,  # All 10 images generate concurrently!
    openai_api_key=OPENAI_API_KEY,
)
```

**Benefits:**
- Your FastAPI app can handle other requests while images are generating
- Carousel generation is faster (parallel instead of sequential)
- Better resource utilization
- No need for external task queues (Dramatiq/Celery) for basic use cases

## API Keys

**Pass API keys directly in code** (recommended for FastAPI apps):

```python
# In your FastAPI app
OPENAI_API_KEY = "your-openai-key"  # From your config/database
OPENROUTER_API_KEY = "your-openrouter-key"  # Optional

# Pass to functions
result = await generate_infographic_from_prompt(
    user_prompt="...",
    openai_api_key=OPENAI_API_KEY,
    openrouter_api_key=OPENROUTER_API_KEY,
)
```

**Or use environment variables** (fallback):

If you don't provide API keys, the package will fall back to environment variables:
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

**Note**: API keys are required. Either provide them in code or set environment variables.

## Available Functions

All functions accept `openai_api_key` and `openrouter_api_key` parameters. All generation functions also support caption generation and run image generation in background threads:

| Function | Description | Caption Support | Background Threads |
|----------|-------------|-----------------|-------------------|
| `generate_infographic_from_prompt(prompt, ..., enable_captions=False, enabled_platforms=None)` | Generate single infographic from text prompt | ✅ Yes | ✅ Yes |
| `generate_carousel_from_prompt(prompt, max_slides, ..., enable_captions=False, enabled_platforms=None)` | Generate carousel slides from text prompt | ✅ Yes | ✅ Yes (concurrent) |
| `generate_carousel_from_text(article_text, max_slides, ..., enable_captions=False, enabled_platforms=None)` | Generate carousel slides from article text | ✅ Yes | ✅ Yes (concurrent) |
| `generate_single_informational_image(url, ..., enable_captions=False, enabled_platforms=None)` | Generate single image from article URL | ✅ Yes | ✅ Yes |
| `generate_single_informational_image_from_text(article_text, ..., enable_captions=False, enabled_platforms=None)` | Generate single image from article text | ✅ Yes | ✅ Yes |
| `generate_infographic_with_reference_image(prompt, image_bytes, ..., enable_captions=False, enabled_platforms=None)` | Generate infographic using reference image | ✅ Yes | ✅ Yes |
| `create_agent(openai_api_key, openrouter_api_key)` | Create agent for processing URLs/text | ✅ Yes (via `agent.process()` or `agent.process_from_text()`) | ✅ Yes (concurrent) |

**Caption Generation Parameters:**
- `enable_captions` (bool, default: False) - Enable caption generation
- `enabled_platforms` (list[str], optional) - List of platforms to generate captions for. Defaults to all platforms when `enable_captions=True`

## Response Format

All functions return a dictionary with:

```python
{
    "status": "success",
    "type": "infographic" | "carousel" | "list" | "summary",  # For infographics
    "title": "Generated title",
    "image_path": "/path/to/image.png",  # For single images
    "slides": [...],  # For carousels
    "total_slides": 5,  # For carousels
    "captions": {  # Only if enable_captions=True
        "facebook_group": "Caption text for Facebook Group...",
        "facebook_page": "Caption text for Facebook Page...",
        "instagram": "Caption text for Instagram...",
        "linkedin": "Caption text for LinkedIn...",
        "pinterest": "Caption text for Pinterest...",
        "reddit": "Caption text for Reddit..."
    }
}
```

**Caption Output:**
- Captions are returned in the `captions` dictionary (only when `enable_captions=True`)
- Captions are also automatically saved to `captions.json` in the output folder
- The JSON file includes metadata: `generated_at` timestamp and list of platforms

## Caption Generation Details

### How It Works

1. **Single LLM Call**: All platform captions are generated in a single API call (efficient!)
2. **Platform-Specific**: Each platform gets a tailored caption following best practices:
   - **Instagram**: Emoji-friendly, hashtag-ready, visual-first
   - **LinkedIn**: Professional, value-focused, industry-relevant
   - **Facebook**: Conversational (Groups) or professional (Pages)
   - **Pinterest**: Keyword-rich, action-oriented
   - **Reddit**: Community-focused, discussion-provoking

3. **Automatic Saving**: Captions are saved to `captions.json` in the output folder alongside images

### Example: Using Captions

```python
result = await generate_infographic_from_prompt(
    user_prompt="top 5 programming languages",
    enable_captions=True,
    enabled_platforms=[INSTAGRAM, LINKEDIN],
)

# Access captions from result
instagram_caption = result["captions"]["instagram"]
linkedin_caption = result["captions"]["linkedin"]

# Or read from JSON file
import json
with open("output/top-5-programming-languages/captions.json") as f:
    captions_data = json.load(f)
    print(captions_data["captions"]["instagram"])
```

## Quick Test

```bash
python -c "from social_visual_generator import create_agent; print('✅ Installed!')"
```

That's it! Your FastAPI app can now use the core package with:
- ✅ **Background image generation** (non-blocking, concurrent)
- ✅ **Caption generation** (multi-platform support)
- ✅ **Production-ready** performance

🎉
