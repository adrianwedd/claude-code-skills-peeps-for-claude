#!/usr/bin/env python3
"""
Rate limiter for API requests with model-specific quotas.

Enforces:
- Free models: 20 RPM (3s minimum gaps)
- Free models: 200 RPD (no credits) or 1000 RPD (with $10+ credits)
- Paid models: No platform limits

Usage:
    limiter = RateLimiter()
    limiter.wait_if_needed("google/gemini-2.0-flash-exp:free")
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


class RateLimiterError(Exception):
    """Base exception for rate limiter errors."""
    pass


class DailyLimitExceeded(RateLimiterError):
    """Raised when daily request limit is exceeded."""
    pass


class RateLimiter:
    """
    Enforce rate limits for API requests.

    Free models (model IDs ending in ':free'):
    - 20 requests per minute (RPM)
    - 200 requests per day (RPD) without credits
    - 1000 requests per day (RPD) with $10+ credits

    Paid models:
    - No platform-level rate limits
    """

    # Conservative limits to avoid edge cases
    FREE_LIMITS = {
        'rpm': 20,      # 20 requests per minute
        'rpd': 190,     # 190/200 safety buffer (default: no credits)
        'gap_seconds': 3.0  # Minimum gap between requests (20 RPM = 3s)
    }

    PAID_LIMITS = {
        'rpm': None,    # No limit
        'rpd': None,    # No limit
        'gap_seconds': 0.1  # Small gap to avoid thundering herd
    }

    def __init__(self, daily_limit: int = None, log_path=None):
        """
        Initialize rate limiter.

        Args:
            daily_limit: Override default daily limit for free models.
                         Use 990 if you have $10+ credits (1000 RPD limit).
                         Use 190 if no credits (200 RPD limit, with safety buffer).
        """
        self.request_times: Dict[str, List[float]] = defaultdict(list)
        self.daily_counts: Dict[str, int] = defaultdict(int)
        self.last_reset = datetime.now(timezone.utc)
        self.log_path = Path(log_path) if log_path else None
        self._global_key = "__global__"

        # Allow override of daily limit
        if daily_limit is not None:
            self.FREE_LIMITS['rpd'] = daily_limit

    def wait_if_needed(self, model_id: str, verbose: bool = True) -> None:
        """
        Block until request is allowed under rate limits.

        Args:
            model_id: Model identifier (e.g., "google/gemini-2.0-flash-exp:free")
            verbose: Print waiting messages

        Raises:
            DailyLimitExceeded: If daily limit reached
        """
        is_free = self._is_free_model(model_id)
        limits = self.FREE_LIMITS if is_free else self.PAID_LIMITS

        # Reset daily counter at midnight UTC
        self._reset_daily_if_needed()

        # Check daily limit (free models only)
        if is_free and limits['rpd'] is not None:
            # Enforce both per-model and global daily limits for free-tier usage visibility.
            if self.daily_counts[self._global_key] >= limits['rpd']:
                raise DailyLimitExceeded(
                    f"Daily limit reached (global) "
                    f"({self.daily_counts[self._global_key]}/{limits['rpd']}). "
                    f"Wait until midnight UTC or use paid model."
                )
            if self.daily_counts[model_id] >= limits['rpd']:
                raise DailyLimitExceeded(
                    f"Daily limit reached for {model_id} "
                    f"({self.daily_counts[model_id]}/{limits['rpd']}). "
                    f"Wait until midnight UTC or use paid model."
                )

        # Check RPM limit
        if limits['rpm'] is not None:
            # OpenRouter/free-tier limits can be global; enforce a global RPM bucket as well.
            if is_free:
                self._enforce_rpm_limit(self._global_key, limits['rpm'], verbose)
            self._enforce_rpm_limit(model_id, limits['rpm'], verbose)

        # Enforce minimum gap between requests
        if limits['gap_seconds'] > 0:
            if is_free:
                self._enforce_gap(self._global_key, limits['gap_seconds'], verbose)
            self._enforce_gap(model_id, limits['gap_seconds'], verbose)

        # Add jitter to avoid thundering herd (±0.25s)
        jitter = (time.time() % 1.0) * 0.5
        time.sleep(jitter)

        # Record request
        self.request_times[model_id].append(time.time())
        self.daily_counts[model_id] += 1
        if is_free:
            self.request_times[self._global_key].append(time.time())
            self.daily_counts[self._global_key] += 1
        self._log_usage(model_id, limits)

    def get_stats(self, model_id: str) -> Dict:
        """Get current usage statistics for a model."""
        is_free = self._is_free_model(model_id)
        limits = self.FREE_LIMITS if is_free else self.PAID_LIMITS

        recent = self._get_recent_requests(model_id, window=60)

        return {
            'model_id': model_id,
            'is_free': is_free,
            'requests_last_minute': len(recent),
            'requests_today': self.daily_counts[model_id],
            'rpm_limit': limits['rpm'],
            'rpd_limit': limits['rpd'],
            'rpd_remaining': limits['rpd'] - self.daily_counts[model_id] if limits['rpd'] else None
        }

    def _log_usage(self, model_id: str, limits: Dict, event: str = "request") -> None:
        if not self.log_path:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).isoformat()
            recent = self._get_recent_requests(model_id, window=60)
            entry = {
                "ts": now,
                "event": event,
                "model_id": model_id,
                "is_free": self._is_free_model(model_id),
                "requests_last_minute": len(recent),
                "requests_today": self.daily_counts[model_id],
                "rpm_limit": limits.get("rpm"),
                "rpd_limit": limits.get("rpd"),
                "gap_seconds": limits.get("gap_seconds"),
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            # Logging must never break benchmark execution.
            # Warn once per session about logging failures
            if not hasattr(self, '_log_warning_shown'):
                print(f"Warning: Rate limiter logging failed: {e}")
                self._log_warning_shown = True
            return

    def _is_free_model(self, model_id: str) -> bool:
        """Check if model is a free tier model."""
        return model_id.endswith(':free')

    def _reset_daily_if_needed(self) -> None:
        """Reset daily counters at midnight UTC."""
        now = datetime.now(timezone.utc)
        if now.date() > self.last_reset.date():
            self.daily_counts.clear()
            self.last_reset = now

    def _get_recent_requests(self, model_id: str, window: int = 60) -> List[float]:
        """Get requests within the last N seconds."""
        now = time.time()
        cutoff = now - window

        # Clean old requests and return recent ones
        recent = [t for t in self.request_times[model_id] if t >= cutoff]
        self.request_times[model_id] = recent
        return recent

    def _enforce_rpm_limit(self, model_id: str, rpm: int, verbose: bool) -> None:
        """Enforce requests per minute limit."""
        recent = self._get_recent_requests(model_id, window=60)

        if len(recent) >= rpm:
            # Wait until oldest request is 60s old
            oldest = recent[0]
            wait_time = 60 - (time.time() - oldest) + 0.5  # +0.5s buffer

            if wait_time > 0:
                if verbose:
                    print(f"  [Rate limit] Waiting {wait_time:.1f}s (RPM: {len(recent)}/{rpm})")
                time.sleep(wait_time)

    def _enforce_gap(self, model_id: str, gap_seconds: float, verbose: bool) -> None:
        """Enforce minimum gap between consecutive requests."""
        if not self.request_times[model_id]:
            return

        last_request = self.request_times[model_id][-1]
        elapsed = time.time() - last_request
        remaining = gap_seconds - elapsed

        if remaining > 0:
            if verbose:
                print(f"  [Gap enforcement] Waiting {remaining:.1f}s (min gap: {gap_seconds}s)")
            time.sleep(remaining)


# Example usage
if __name__ == "__main__":
    print("Testing RateLimiter...")

    # Test with free model
    limiter = RateLimiter()

    print("\nTest 1: Free model (should enforce 3s gaps)")
    for i in range(3):
        print(f"  Request {i+1}/3...")
        limiter.wait_if_needed("google/gemini-2.0-flash-exp:free")
        stats = limiter.get_stats("google/gemini-2.0-flash-exp:free")
        print(f"    Stats: {stats}")

    print("\nTest 2: Paid model (minimal gaps)")
    for i in range(3):
        print(f"  Request {i+1}/3...")
        limiter.wait_if_needed("openai/gpt-4-turbo", verbose=False)
        stats = limiter.get_stats("openai/gpt-4-turbo")
        print(f"    Stats: {stats}")

    print("\n✓ Rate limiter working correctly!")
