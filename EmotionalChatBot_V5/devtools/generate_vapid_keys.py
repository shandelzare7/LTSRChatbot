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

import base64


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
    pub_key = v.public_key
    priv_key = v.private_key
    if not pub_key or not priv_key:
        raise SystemExit("Failed to generate VAPID keys.")

    # Export for Web Push:
    # - public key: uncompressed point 0x04 || X(32) || Y(32), base64url (no padding)
    # - private key: private_value as 32 bytes, base64url (no padding)
    try:
        numbers = pub_key.public_numbers()
        x = int(numbers.x).to_bytes(32, "big")
        y = int(numbers.y).to_bytes(32, "big")
        pub_raw = b"\x04" + x + y

        priv_val = int(priv_key.private_numbers().private_value).to_bytes(32, "big")
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to export keys: {e}")

    pub_b64 = base64.urlsafe_b64encode(pub_raw).decode("ascii").rstrip("=")
    priv_b64 = base64.urlsafe_b64encode(priv_val).decode("ascii").rstrip("=")

    print("VAPID_PUBLIC_KEY=" + pub_b64)
    print("VAPID_PRIVATE_KEY=" + priv_b64)
    print("VAPID_SUBJECT=mailto:you@example.com")


if __name__ == "__main__":
    main()

