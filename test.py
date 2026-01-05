"""
Interactive test script for all core package functions.

Run this script and choose which function to test.
"""

import asyncio
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from social_visual_generator import (
    generate_infographic_from_prompt,
    generate_carousel_from_prompt,
    generate_carousel_from_text,
    generate_single_informational_image,
    generate_single_informational_image_from_text,
    generate_infographic_with_reference_image,
    create_agent,
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


async def test_infographic_from_prompt():
    """Test: Generate Infographic from Prompt"""
    print("\n" + "=" * 60)
    print("🧪 Generate Infographic from Prompt")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file")
        return

    prompt = (
        input("\nEnter prompt (e.g., 'top 5 programming languages'): ").strip()
        or "top 5 programming languages"
    )
    username = input("Username (default: @test): ").strip() or "@test"
    tagline = input("Tagline (default: test tagline): ").strip() or "test tagline"
    orientation = (
        input("Orientation (square/vertical, default: square): ").strip().lower() or "square"
    )
    if orientation not in ["square", "vertical"]:
        orientation = "square"

    # Caption generation option
    enable_captions = input("Generate captions? (y/N): ").strip().lower() == "y"
    enabled_platforms = None
    if enable_captions:
        enabled_platforms = select_platforms()

    print(f"\n📝 Generating infographic for: '{prompt}'")
    print(f"   Orientation: {orientation}")
    if enable_captions:
        print("   (with caption generation)")
    print("⏳ This may take 30-60 seconds...\n")

    try:
        result = await generate_infographic_from_prompt(
            user_prompt=prompt,
            username=username,
            tagline=tagline,
            font_name="Inter",
            background_info="Dark background #101010",
            color_schema="White text with cyan accents",
            orientation=orientation,
            enable_captions=enable_captions,
            enabled_platforms=enabled_platforms,
        )

        print("\n✅ Success!")
        print(f"   Type: {result.get('type')}")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Image saved to: {result.get('image_path', 'N/A')}")

        # Show captions if generated
        captions = result.get("captions", {})
        if captions:
            print(f"\n📝 Generated {len(captions)} captions (saved to JSON)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_carousel_from_prompt():
    """Test: Generate Carousel from Prompt"""
    print("\n" + "=" * 60)
    print("🧪 Generate Carousel from Prompt")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file")
        return

    prompt = input("\nEnter prompt (e.g., 'top 5 AI tools'): ").strip() or "top 5 AI tools"
    max_slides = input("Max slides (default: 5): ").strip() or "5"
    username = input("Username (default: @test): ").strip() or "@test"
    tagline = input("Tagline (default: test tagline): ").strip() or "test tagline"
    orientation = (
        input("Orientation (square/vertical, default: square): ").strip().lower() or "square"
    )
    if orientation not in ["square", "vertical"]:
        orientation = "square"

    # Caption generation option
    enable_captions = input("Generate captions? (y/N): ").strip().lower() == "y"
    enabled_platforms = None
    if enable_captions:
        enabled_platforms = select_platforms()

    try:
        max_slides = int(max_slides)
    except ValueError:
        max_slides = 5

    print(f"\n📝 Generating carousel for: '{prompt}'")
    print(f"   Slides: {max_slides}")
    print(f"   Orientation: {orientation}")
    if enable_captions:
        print("   (with caption generation)")
    print("⏳ This may take 2-5 minutes...\n")

    try:
        result = await generate_carousel_from_prompt(
            user_prompt=prompt,
            max_slides=max_slides,
            username=username,
            tagline=tagline,
            orientation=orientation,
            enable_captions=enable_captions,
            enabled_platforms=enabled_platforms,
        )

        print("\n✅ Success!")
        print(f"   Total slides: {result.get('total_slides', 0)}")

        if result.get("slides"):
            print("\n📋 Slides generated:")
            for slide in result.get("slides", []):
                print(f"   Slide {slide.get('slide_number')}: {slide.get('title', 'N/A')[:50]}")
                print(f"      Image: {slide.get('image_path', 'N/A')}")

        # Show captions if generated
        captions = result.get("captions", {})
        if captions:
            print(f"\n📝 Generated {len(captions)} captions (saved to JSON)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_single_image():
    """Test: Generate Single Informational Image"""
    print("\n" + "=" * 60)
    print("🧪 Generate Single Informational Image")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file")
        return

    url = (
        input("\nEnter article URL (e.g., 'https://example.com'): ").strip()
        or "https://example.com"
    )
    username = input("Username (default: @test): ").strip() or "@test"
    tagline = input("Tagline (default: test tagline): ").strip() or "test tagline"
    orientation = (
        input("Orientation (square/vertical, default: square): ").strip().lower() or "square"
    )
    if orientation not in ["square", "vertical"]:
        orientation = "square"

    # Caption generation option
    enable_captions = input("Generate captions? (y/N): ").strip().lower() == "y"
    enabled_platforms = None
    if enable_captions:
        enabled_platforms = select_platforms()

    print(f"\n📝 Generating image from: {url}")
    print(f"   Orientation: {orientation}")
    if enable_captions:
        print("   (with caption generation)")
    print("⏳ This may take 30-60 seconds...\n")

    try:
        result = await generate_single_informational_image(
            url=url,
            username=username,
            tagline=tagline,
            orientation=orientation,
            enable_captions=enable_captions,
            enabled_platforms=enabled_platforms,
        )

        print("\n✅ Success!")
        print(f"   Title: {result.get('title', 'N/A')[:50]}")
        print(f"   Image saved to: {result.get('image_path', 'N/A')}")

        # Show captions if generated
        captions = result.get("captions", {})
        if captions:
            print(f"\n📝 Generated {len(captions)} captions (saved to JSON)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_infographic_with_reference():
    """Test: Generate Infographic with Reference Image"""
    print("\n" + "=" * 60)
    print("🧪 Generate Infographic with Reference Image")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file")
        return

    prompt = input("\nEnter prompt (e.g., 'top 5 tools'): ").strip() or "top 5 tools"
    image_path = input("Reference image path: ").strip()

    if not image_path or not Path(image_path).exists():
        print("❌ Reference image file not found")
        return

    username = input("Username (default: @test): ").strip() or "@test"
    tagline = input("Tagline (default: test tagline): ").strip() or "test tagline"
    orientation = (
        input("Orientation (square/vertical, default: square): ").strip().lower() or "square"
    )
    if orientation not in ["square", "vertical"]:
        orientation = "square"

    # Caption generation option
    enable_captions = input("Generate captions? (y/N): ").strip().lower() == "y"
    enabled_platforms = None
    if enable_captions:
        enabled_platforms = select_platforms()

    print(f"\n📝 Generating infographic with reference image...")
    print(f"   Orientation: {orientation}")
    if enable_captions:
        print("   (with caption generation)")
    print("⏳ This may take 30-60 seconds...\n")

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = await generate_infographic_with_reference_image(
            user_prompt=prompt,
            reference_image_bytes=image_bytes,
            username=username,
            tagline=tagline,
            orientation=orientation,
            image_provider="openai",
            image_model="gpt-image-1.5",
            enable_captions=enable_captions,
            enabled_platforms=enabled_platforms,
        )

        print("\n✅ Success!")
        print(f"   Type: {result.get('type')}")
        print(f"   Title: {result.get('title', 'N/A')[:50]}")
        print(f"   Image saved to: {result.get('image_path', 'N/A')}")

        # Show captions if generated
        captions = result.get("captions", {})
        if captions:
            print(f"\n📝 Generated {len(captions)} captions (saved to JSON)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_carousel_from_text():
    """Test: Generate Carousel from Text"""
    print("\n" + "=" * 60)
    print("🧪 Generate Carousel from Text")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file")
        return

    print("\nEnter article text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if not line and lines:  # Empty line after content
            break
        if line:
            lines.append(line)

    article_text = "\n".join(lines)
    if not article_text.strip():
        print("❌ Article text is required")
        return

    max_slides = input("\nMax slides (default: 5): ").strip() or "5"
    title = input("Title (optional, press Enter to auto-extract): ").strip() or None
    username = input("Username (default: @test): ").strip() or "@test"
    tagline = input("Tagline (default: test tagline): ").strip() or "test tagline"
    orientation = (
        input("Orientation (square/vertical, default: square): ").strip().lower() or "square"
    )
    if orientation not in ["square", "vertical"]:
        orientation = "square"
    font_name = input("Font name (optional, e.g., 'Arial', 'Roboto'): ").strip() or None
    background_info = (
        input("Background info (optional, e.g., 'dark navy gradient'): ").strip() or None
    )
    color_schema = (
        input("Color schema (optional, e.g., 'navy background with white text'): ").strip() or None
    )
    extra_instructions = input("Extra instructions (optional): ").strip() or None

    # Caption generation option
    enable_captions = input("Generate captions? (y/N): ").strip().lower() == "y"
    enabled_platforms = None
    if enable_captions:
        enabled_platforms = select_platforms()

    try:
        max_slides = int(max_slides)
    except ValueError:
        max_slides = 5

    print(f"\n📝 Generating carousel from text ({len(article_text)} chars)")
    print(f"   Slides: {max_slides}")
    print(f"   Orientation: {orientation}")
    if enable_captions:
        print("   (with caption generation)")
    print("⏳ This may take 2-5 minutes...\n")

    try:
        result = await generate_carousel_from_text(
            article_text=article_text,
            max_slides=max_slides,
            title=title,
            username=username,
            tagline=tagline,
            orientation=orientation,
            font_name=font_name,
            background_info=background_info,
            color_schema=color_schema,
            extra_instructions=extra_instructions,
            enable_captions=enable_captions,
            enabled_platforms=enabled_platforms,
        )

        print("\n✅ Success!")
        print(f"   Total slides: {result.get('total_slides', 0)}")
        print(f"   Article: {result.get('article_title', 'N/A')}")

        if result.get("slides"):
            print("\n📋 Slides generated:")
            for slide in result.get("slides", [])[:3]:  # Show first 3
                print(f"   Slide {slide.get('slide_number')}: {slide.get('title', 'N/A')[:50]}")

        # Show captions if generated
        captions = result.get("captions", {})
        if captions:
            print(f"\n📝 Generated {len(captions)} captions (saved to JSON)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_single_image_from_text():
    """Test: Generate Single Image from Text"""
    print("\n" + "=" * 60)
    print("🧪 Generate Single Image from Text")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file")
        return

    print("\nEnter article text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if not line and lines:  # Empty line after content
            break
        if line:
            lines.append(line)

    article_text = "\n".join(lines)
    if not article_text.strip():
        print("❌ Article text is required")
        return

    title = input("\nTitle (optional, press Enter to auto-extract): ").strip() or None
    username = input("Username (default: @test): ").strip() or "@test"
    tagline = input("Tagline (default: test tagline): ").strip() or "test tagline"
    orientation = (
        input("Orientation (square/vertical, default: square): ").strip().lower() or "square"
    )
    if orientation not in ["square", "vertical"]:
        orientation = "square"
    font_name = input("Font name (optional, e.g., 'Arial', 'Roboto'): ").strip() or None
    background_info = (
        input("Background info (optional, e.g., 'dark navy gradient'): ").strip() or None
    )
    color_schema = (
        input("Color schema (optional, e.g., 'navy background with white text'): ").strip() or None
    )
    extra_instructions = input("Extra instructions (optional): ").strip() or None

    # Caption generation option
    enable_captions = input("Generate captions? (y/N): ").strip().lower() == "y"
    enabled_platforms = None
    if enable_captions:
        enabled_platforms = select_platforms()

    print(f"\n📝 Generating image from text ({len(article_text)} chars)")
    print(f"   Orientation: {orientation}")
    if enable_captions:
        print("   (with caption generation)")
    print("⏳ This may take 30-60 seconds...\n")

    try:
        result = await generate_single_informational_image_from_text(
            article_text=article_text,
            title=title,
            username=username,
            tagline=tagline,
            orientation=orientation,
            font_name=font_name,
            background_info=background_info,
            color_schema=color_schema,
            extra_instructions=extra_instructions,
            enable_captions=enable_captions,
            enabled_platforms=enabled_platforms,
        )

        print("\n✅ Success!")
        print(f"   Title: {result.get('title', 'N/A')[:50]}")
        print(f"   Image saved to: {result.get('image_path', 'N/A')}")

        # Show captions if generated
        captions = result.get("captions", {})
        if captions:
            print(f"\n📝 Generated {len(captions)} captions (saved to JSON)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_agent_process():
    """Test: Agent Process (Carousel from URL)"""
    print("\n" + "=" * 60)
    print("🧪 Agent Process - Carousel from URL")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file")
        return

    url = input("\nEnter article URL: ").strip()
    if not url:
        print("❌ URL is required")
        return

    max_slides = input("Max slides (default: 5): ").strip() or "5"
    username = input("Username (default: @test): ").strip() or "@test"
    tagline = input("Tagline (default: test tagline): ").strip() or "test tagline"
    orientation = (
        input("Orientation (square/vertical, default: square): ").strip().lower() or "square"
    )
    if orientation not in ["square", "vertical"]:
        orientation = "square"

    # Caption generation option
    enable_captions = input("Generate captions? (y/N): ").strip().lower() == "y"
    enabled_platforms = None
    if enable_captions:
        enabled_platforms = select_platforms()

    try:
        max_slides = int(max_slides)
    except ValueError:
        max_slides = 5

    print(f"\n📝 Processing article from: {url}")
    print(f"   Slides: {max_slides}")
    print(f"   Orientation: {orientation}")
    if enable_captions:
        print("   (with caption generation)")
    print("⏳ This may take 2-5 minutes...\n")

    try:
        agent = create_agent()
        result = await agent.process(
            url=url,
            max_slides=max_slides,
            username=username,
            tagline=tagline,
            orientation=orientation,
            enable_captions=enable_captions,
            enabled_platforms=enabled_platforms,
        )

        print("\n✅ Success!")
        print(f"   Total slides: {result.get('total_slides', 0)}")
        print(f"   Article: {result.get('article_title', 'N/A')}")

        if result.get("slides"):
            print("\n📋 Slides generated:")
            for slide in result.get("slides", [])[:3]:  # Show first 3
                print(f"   Slide {slide.get('slide_number')}: {slide.get('title', 'N/A')[:50]}")

        # Show captions if generated
        captions = result.get("captions", {})
        if captions:
            print(f"\n📝 Generated {len(captions)} captions (saved to JSON)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_agent_process_from_text():
    """Test: Agent Process from Text"""
    print("\n" + "=" * 60)
    print("🧪 Agent Process - Carousel from Text")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file")
        return

    print("\nEnter article text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if not line and lines:  # Empty line after content
            break
        if line:
            lines.append(line)

    article_text = "\n".join(lines)
    if not article_text.strip():
        print("❌ Article text is required")
        return

    max_slides = input("\nMax slides (default: 5): ").strip() or "5"
    title = input("Title (optional, press Enter to auto-extract): ").strip() or None
    username = input("Username (default: @test): ").strip() or "@test"
    tagline = input("Tagline (default: test tagline): ").strip() or "test tagline"
    orientation = (
        input("Orientation (square/vertical, default: square): ").strip().lower() or "square"
    )
    if orientation not in ["square", "vertical"]:
        orientation = "square"
    font_name = input("Font name (optional, e.g., 'Arial', 'Roboto'): ").strip() or None
    background_info = (
        input("Background info (optional, e.g., 'dark navy gradient'): ").strip() or None
    )
    color_schema = (
        input("Color schema (optional, e.g., 'navy background with white text'): ").strip() or None
    )
    extra_instructions = input("Extra instructions (optional): ").strip() or None

    # Caption generation option
    enable_captions = input("Generate captions? (y/N): ").strip().lower() == "y"
    enabled_platforms = None
    if enable_captions:
        enabled_platforms = select_platforms()

    try:
        max_slides = int(max_slides)
    except ValueError:
        max_slides = 5

    print(f"\n📝 Processing article from text ({len(article_text)} chars)")
    print(f"   Slides: {max_slides}")
    print(f"   Orientation: {orientation}")
    if enable_captions:
        print("   (with caption generation)")
    print("⏳ This may take 2-5 minutes...\n")

    try:
        agent = create_agent()
        result = await agent.process_from_text(
            article_text=article_text,
            max_slides=max_slides,
            title=title,
            username=username,
            tagline=tagline,
            orientation=orientation,
            font_name=font_name,
            background_info=background_info,
            color_schema=color_schema,
            extra_instructions=extra_instructions,
            enable_captions=enable_captions,
            enabled_platforms=enabled_platforms,
        )

        print("\n✅ Success!")
        print(f"   Total slides: {result.get('total_slides', 0)}")
        print(f"   Article: {result.get('article_title', 'N/A')}")

        if result.get("slides"):
            print("\n📋 Slides generated:")
            for slide in result.get("slides", [])[:3]:  # Show first 3
                print(f"   Slide {slide.get('slide_number')}: {slide.get('title', 'N/A')[:50]}")

        # Show captions if generated
        captions = result.get("captions", {})
        if captions:
            print(f"\n📝 Generated {len(captions)} captions (saved to JSON)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


def select_platforms():
    """Interactive platform selection for caption generation."""
    print("\n📱 Available platforms for caption generation:")
    print("  1. Facebook Group")
    print("  2. Facebook Page")
    print("  3. Instagram")
    print("  4. LinkedIn")
    print("  5. Pinterest")
    print("  6. Reddit")
    print("  A. All platforms")
    print()

    platform_map = {
        "1": FACEBOOK_GROUP,
        "2": FACEBOOK_PAGE,
        "3": INSTAGRAM,
        "4": LINKEDIN,
        "5": PINTEREST,
        "6": REDDIT,
    }

    selection = input("Select platforms (e.g., '1,3,4' or 'A' for all): ").strip().upper()

    if selection == "A" or not selection:
        return ALL_PLATFORMS

    selected = []
    for choice in selection.split(","):
        choice = choice.strip()
        if choice in platform_map:
            selected.append(platform_map[choice])

    return selected if selected else ALL_PLATFORMS


def show_menu():
    """Display test menu"""
    print("\n" + "=" * 60)
    print("🧪 CORE PACKAGE TEST MENU")
    print("=" * 60)
    print("\nChoose a test to run:")
    print("  1. Generate Infographic from Prompt")
    print("  2. Generate Carousel from Prompt")
    print("  3. Generate Single Image from URL")
    print("  4. Generate Infographic with Reference Image")
    print("  5. Agent Process - Carousel from URL")
    print("  6. Generate Carousel from Text")
    print("  7. Generate Single Image from Text")
    print("  8. Agent Process - Carousel from Text")
    print("  0. Exit")
    print()


async def main():
    """Main test runner"""
    while True:
        show_menu()
        choice = input("Enter choice (0-8): ").strip()

        if choice == "0":
            print("\n👋 Goodbye!")
            break
        elif choice == "1":
            await test_infographic_from_prompt()
        elif choice == "2":
            await test_carousel_from_prompt()
        elif choice == "3":
            await test_single_image()
        elif choice == "4":
            await test_infographic_with_reference()
        elif choice == "5":
            await test_agent_process()
        elif choice == "6":
            await test_carousel_from_text()
        elif choice == "7":
            await test_single_image_from_text()
        elif choice == "8":
            await test_agent_process_from_text()
        else:
            print("❌ Invalid choice. Please enter 0-8.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())
