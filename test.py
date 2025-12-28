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
    generate_single_informational_image,
    generate_infographic_with_reference_image,
    create_agent,
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

    print(f"\n📝 Generating infographic for: '{prompt}'")
    print("⏳ This may take 30-60 seconds...\n")

    try:
        result = await generate_infographic_from_prompt(
            user_prompt=prompt,
            username=username,
            tagline=tagline,
            font_name="Inter",
            background_info="Dark background #101010",
            color_schema="White text with cyan accents",
        )

        print("\n✅ Success!")
        print(f"   Type: {result.get('type')}")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Image saved to: {result.get('image_path', 'N/A')}")

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

    try:
        max_slides = int(max_slides)
    except ValueError:
        max_slides = 5

    print(f"\n📝 Generating carousel for: '{prompt}'")
    print(f"   Slides: {max_slides}")
    print("⏳ This may take 2-5 minutes...\n")

    try:
        result = await generate_carousel_from_prompt(
            user_prompt=prompt,
            max_slides=max_slides,
            username=username,
            tagline=tagline,
        )

        print("\n✅ Success!")
        print(f"   Total slides: {result.get('total_slides', 0)}")

        if result.get("slides"):
            print("\n📋 Slides generated:")
            for slide in result.get("slides", []):
                print(f"   Slide {slide.get('slide_number')}: {slide.get('title', 'N/A')[:50]}")
                print(f"      Image: {slide.get('image_path', 'N/A')}")

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

    print(f"\n📝 Generating image from: {url}")
    print("⏳ This may take 30-60 seconds...\n")

    try:
        result = await generate_single_informational_image(
            url=url,
            username=username,
            tagline=tagline,
        )

        print("\n✅ Success!")
        print(f"   Title: {result.get('title', 'N/A')[:50]}")
        print(f"   Image saved to: {result.get('image_path', 'N/A')}")

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

    print(f"\n📝 Generating infographic with reference image...")
    print("⏳ This may take 30-60 seconds...\n")

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = await generate_infographic_with_reference_image(
            user_prompt=prompt,
            reference_image_bytes=image_bytes,
            username=username,
            tagline=tagline,
            image_provider="openai",
            image_model="gpt-image-1.5",
        )

        print("\n✅ Success!")
        print(f"   Type: {result.get('type')}")
        print(f"   Title: {result.get('title', 'N/A')[:50]}")
        print(f"   Image saved to: {result.get('image_path', 'N/A')}")

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

    try:
        max_slides = int(max_slides)
    except ValueError:
        max_slides = 5

    print(f"\n📝 Processing article from: {url}")
    print(f"   Slides: {max_slides}")
    print("⏳ This may take 2-5 minutes...\n")

    try:
        agent = create_agent()
        result = await agent.process(
            url=url,
            max_slides=max_slides,
            username=username,
            tagline=tagline,
        )

        print("\n✅ Success!")
        print(f"   Total slides: {result.get('total_slides', 0)}")
        print(f"   Article: {result.get('article_title', 'N/A')}")

        if result.get("slides"):
            print("\n📋 Slides generated:")
            for slide in result.get("slides", [])[:3]:  # Show first 3
                print(f"   Slide {slide.get('slide_number')}: {slide.get('title', 'N/A')[:50]}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


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
    print("  0. Exit")
    print()


async def main():
    """Main test runner"""
    while True:
        show_menu()
        choice = input("Enter choice (0-5): ").strip()

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
        else:
            print("❌ Invalid choice. Please enter 0-5.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())
