#!/usr/bin/env python3
"""Validate NVIDIA API key and model access."""
from __future__ import annotations

import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from llm_client import (
    BIG_MODEL,
    FALLBACK_SMALL_MODEL,
    NEMOTRON_NANO_MODEL,
    SMALL_MODEL,
    NvidiaClient,
    is_nano_model,
)

NVIDIA_HOST = "integrate.api.nvidia.com"
NANO_TEST_TIMEOUT_S = 50


def _check_network() -> str | None:
    try:
        socket.getaddrinfo(NVIDIA_HOST, 443)
    except socket.gaierror as exc:
        return f"Cannot resolve {NVIDIA_HOST} ({exc}). Check Wi‑Fi/VPN."
    return None


def _test_model(client: NvidiaClient, model: str) -> tuple[bool, str, float]:
    resp = client.chat(
        model,
        [{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=16,
    )
    return True, resp.content[:60], resp.latency_ms


def main() -> int:
    key = os.getenv("NVIDIA_API_KEY", "")
    if not key:
        print("❌ NVIDIA_API_KEY not set in .env")
        return 1

    if err := _check_network():
        print(f"❌ Network: {err}")
        return 1

    client = NvidiaClient(mock=False)
    if client.mock:
        print("❌ Client fell back to mock mode")
        return 1

    print(f"Testing BIG ({BIG_MODEL})…", flush=True)
    try:
        ok, text, ms = _test_model(client, BIG_MODEL)
        print(f"✅ BIG: {text!r} — {ms:.0f} ms")
    except Exception as exc:
        print(f"❌ BIG: {exc}")
        return 1

    print(f"Testing SMALL ({SMALL_MODEL})…", flush=True)
    try:
        if is_nano_model(SMALL_MODEL):
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_test_model, client, SMALL_MODEL)
                ok, text, ms = fut.result(timeout=NANO_TEST_TIMEOUT_S)
        else:
            ok, text, ms = _test_model(client, SMALL_MODEL)
        print(f"✅ SMALL: {text!r} — {ms:.0f} ms")
    except FuturesTimeout:
        print(f"❌ Nemotron Nano timed out after {NANO_TEST_TIMEOUT_S}s on hosted NIM.")
        print(f"   Fix: remove SMALL_MODEL from .env (uses fast default), or set:")
        print(f"   SMALL_MODEL={FALLBACK_SMALL_MODEL}")
        print(f"   (Llama 3.1 8B on NVIDIA NIM — works reliably for demos)")
        return 1
    except Exception as exc:
        print(f"❌ SMALL ({SMALL_MODEL}): {exc}")
        if is_nano_model(SMALL_MODEL):
            print(f"   Try: SMALL_MODEL={FALLBACK_SMALL_MODEL} in .env")
        return 1

    if SMALL_MODEL == FALLBACK_SMALL_MODEL:
        print(f"\nℹ️  Using default small model ({FALLBACK_SMALL_MODEL}).")
        print(f"   Optional Nano: SMALL_MODEL={NEMOTRON_NANO_MODEL} (may be slow on NIM)")

    print("\n✅ API key valid — run: python demo.py --real --quick --yes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
