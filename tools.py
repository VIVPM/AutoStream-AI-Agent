from langchain_core.tools import tool


@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Capture a lead by collecting the user's name, email, and creator platform.
    Only call this tool after collecting ALL three values from the user.

    Args:
        name: The user's full name
        email: The user's email address
        platform: The user's creator platform (e.g., YouTube, Instagram, TikTok)
    """
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return f"Lead captured successfully: {name}, {email}, {platform}"
