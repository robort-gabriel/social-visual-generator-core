# Testing the Core Package

## Quick Test

### Step 1: Navigate to Core Repo

```bash
cd social-visual-generator-core
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install in Development Mode

```bash
pip install -e .
pip install -r requirements.txt
```

### Step 4: Test Import

```bash
python -c "from social_visual_generator import SocialMediaContentGeneratorAgent; print('✅ Import works!')"
```

## Comprehensive Test (All Functions)

### Run All Tests

```bash
# Make sure .env file exists with API keys (OPENAI_API_KEY, OPENROUTER_API_KEY)
python test_all_functions.py
```

This will test:
1. ✅ Import functionality
2. ✅ Filename sanitization
3. ✅ Agent creation
4. ✅ Web scraping
5. ✅ Infographic from prompt
6. ✅ Carousel from prompt
7. ✅ Single informational image
8. ✅ Agent process method
9. ✅ Infographic with reference image (requires image file)

## Individual Function Tests

### Test Infographic from Prompt

```python
import asyncio
from social_visual_generator import generate_infographic_from_prompt

async def test():
    result = await generate_infographic_from_prompt(
        user_prompt="top 5 programming languages",
        username="@test",
        tagline="test"
    )
    print(f"✅ Generated! Type: {result.get('type')}")
    print(f"   Image: {result.get('image_path')}")

asyncio.run(test())
```

### Test Carousel from Prompt

```python
import asyncio
from social_visual_generator import generate_carousel_from_prompt

async def test():
    result = await generate_carousel_from_prompt(
        user_prompt="top 5 AI tools",
        max_slides=5,
        username="@test",
        tagline="test"
    )
    print(f"✅ Generated {result.get('total_slides')} slides!")

asyncio.run(test())
```

### Test Agent Process (Carousel from URL)

```python
import asyncio
from social_visual_generator import create_agent

async def test():
    agent = create_agent()
    result = await agent.process(
        url="https://example.com/article",
        max_slides=5,
        username="@test",
        tagline="test"
    )
    print(f"✅ Generated {result.get('total_slides')} slides!")

asyncio.run(test())
```

### Test Single Image

```python
import asyncio
from social_visual_generator import generate_single_informational_image

async def test():
    result = await generate_single_informational_image(
        url="https://example.com/article",
        username="@test",
        tagline="test"
    )
    print(f"✅ Generated! Image: {result.get('image_path')}")

asyncio.run(test())
```

### Test Infographic with Reference

```python
import asyncio
from social_visual_generator import generate_infographic_with_reference_image

async def test():
    # Read reference image
    with open("reference.png", "rb") as f:
        image_bytes = f.read()
    
    result = await generate_infographic_with_reference_image(
        user_prompt="top 5 tools",
        reference_image_bytes=image_bytes,
        username="@test",
        tagline="test",
        image_provider="openai",
        image_model="gpt-image-1.5"
    )
    print(f"✅ Generated! Image: {result.get('image_path')}")

asyncio.run(test())
```

## Expected Output Location

Files will save to: `./output/` (in the directory where you run the script)

## Verify Output

After running, check:
```bash
ls -la output/
```

You should see folders with generated images.

## Test Requirements

- **API Keys**: Set `OPENAI_API_KEY` and `OPENROUTER_API_KEY` in `.env` file
- **Internet**: Required for web scraping tests
- **Time**: Full test suite takes ~2-5 minutes (image generation is slow)



