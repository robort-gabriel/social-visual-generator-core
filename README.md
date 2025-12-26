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
- **Image Generation**: Supports multiple providers (OpenRouter/Gemini, OpenAI DALL-E/GPT-Image)
- **Reference Image Support**: Match existing design styles using reference images
- **Web Scraping**: Automatic article content extraction using Playwright
- **Design Customization**: Customize fonts, backgrounds, and color schemes

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

```python
from social_visual_generator import SocialMediaContentGeneratorAgent

# Create agent instance
agent = SocialMediaContentGeneratorAgent()

# Generate carousel from URL
result = await agent.process(
    url="https://example.com/article",
    max_slides=10,
    username="@yourhandle",
    tagline="daily tips"
)
```

## Requirements

- Python 3.11+
- OpenAI API key (for content generation)
- OpenRouter API key (for image generation, optional if using OpenAI)

## Documentation

Full documentation available at: [GitHub Repository](https://github.com/robort-gabriel/social-visual-generator-core)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Support

For issues and questions, please open an issue on GitHub.
# social-visual-generator-core
# social-visual-generator-core
# social-visual-generator-core
