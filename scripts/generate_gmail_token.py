"""
One-off script to generate a Gmail OAuth refresh token with the correct scope.

Run once locally, not part of the app:
    uv run --with google-auth-oauthlib python scripts/generate_gmail_token.py

Requires GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET already set in your .env
(same values used by nxb_chatbot.core.config.settings).

Opens a browser consent screen for the https://mail.google.com/ scope
(matches GMAIL_SCOPES in nxb_chatbot/tools/gmail.py) and prints the
resulting refresh token — paste it into GMAIL_REFRESH_TOKEN in your .env.
"""

from google_auth_oauthlib.flow import InstalledAppFlow

from nxb_chatbot.core.config import settings

SCOPES = ["https://mail.google.com/"]


def main() -> None:
    client_config = {
        "installed": {
            "client_id": settings.GMAIL_CLIENT_ID,
            "client_secret": settings.GMAIL_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": settings.GMAIL_TOKEN_URI,
            "redirect_uris": ["http://localhost"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0)

    print("\nAuthorization complete.\n")
    print(f"GMAIL_REFRESH_TOKEN={creds.refresh_token}")
    print("\nCopy the line above into your .env, replacing the existing GMAIL_REFRESH_TOKEN.")


if __name__ == "__main__":
    main()
