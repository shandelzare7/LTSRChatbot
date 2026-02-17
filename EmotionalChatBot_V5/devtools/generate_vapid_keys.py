"""
Generate Web Push VAPID keys.

Usage:
  cd EmotionalChatBot_V5
  python devtools/generate_vapid_keys.py

It prints:
  - VAPID_PUBLIC_KEY (base64url)
  - VAPID_PRIVATE_KEY (base64url)

Do NOT commit these keys. Put them into your deployment environment variables (Render).
"""

from __future__ import annotations


def main() -> None:
    try:
        # pywebpush depends on py-vapid; this is the simplest stable generator.
        from py_vapid import Vapid  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install pywebpush\n"
            f"\nImport error: {e}\n"
        )

    v = Vapid()
    v.generate_keys()
    pub = v.public_key
    priv = v.private_key
    if not pub or not priv:
        raise SystemExit("Failed to generate VAPID keys.")

    print("VAPID_PUBLIC_KEY=" + str(pub))
    print("VAPID_PRIVATE_KEY=" + str(priv))
    print("VAPID_SUBJECT=mailto:you@example.com")


if __name__ == "__main__":
    main()

