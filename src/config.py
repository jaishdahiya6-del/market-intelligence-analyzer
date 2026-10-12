"""Configuration. Keys come ONLY from environment variables / .env (never hard-coded)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"   # always the project folder, not the cwd
load_dotenv(ENV_PATH)
load_dotenv()  # also honour a .env in the current directory

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))


def alpha_key() -> str:
    return os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()


def coingecko_key() -> str:
    return os.getenv("COINGECKO_API_KEY", "").strip()


def coingecko_key_type() -> str:
    return os.getenv("COINGECKO_KEY_TYPE", "demo").strip().lower()


def coingecko_base() -> str:
    if coingecko_key_type() == "pro":
        return "https://pro-api.coingecko.com/api/v3"
    return "https://api.coingecko.com/api/v3"


def coingecko_header() -> dict:
    key = coingecko_key()
    if not key:
        return {}
    name = "x-cg-pro-api-key" if coingecko_key_type() == "pro" else "x-cg-demo-api-key"
    return {name: key}


def _write_env(updates: dict) -> None:
    """Create/update the project .env, preserving other lines."""
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    seen = set()
    for i, line in enumerate(lines):
        k = line.split("=", 1)[0].strip()
        if k in updates:
            lines[i] = f"{k}={updates[k]}"
            seen.add(k)
    lines += [f"{k}={v}" for k, v in updates.items() if k not in seen]
    ENV_PATH.write_text("\n".join(lines) + "\n")


def set_keys(alpha: str | None = None, coingecko: str | None = None, cg_type: str | None = None,
             persist: bool = False) -> None:
    """Set keys for this running process (and optionally save them to the local .env)."""
    updates = {}
    if alpha:
        updates["ALPHA_VANTAGE_API_KEY"] = alpha.strip()
    if coingecko:
        updates["COINGECKO_API_KEY"] = coingecko.strip()
    if cg_type:
        updates["COINGECKO_KEY_TYPE"] = cg_type
    os.environ.update(updates)
    if persist and updates:
        _write_env(updates)
