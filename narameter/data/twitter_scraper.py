"""
Twitter/X scraper for Narameter pipeline.

Uses the Scweet library to scrape tweets via X's internal GraphQL API.
Authentication is done via browser session cookies (auth_token).

Setup:
  1. Log in to x.com in your browser
  2. DevTools (F12) → Application → Cookies → x.com → copy `auth_token`
  3. Pass it via --auth-token flag, or export all cookies via Cookie-Editor
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TwitterScraper:
    """
    Scrapes tweets from Twitter/X using Scweet.

    Usage:
        scraper = TwitterScraper(auth_token="YOUR_AUTH_TOKEN")
        df = scraper.search("IKN Nusantara", since="2024-01-01", limit=500)

    Or with a cookies file:
        scraper = TwitterScraper.from_cookies("cookies.json")
        df = scraper.search("keyword", limit=500)
    """

    COOKIES_PATH = "data/raw/twitter_cookies.json"

    def __init__(self, auth_token: str):
        """
        Initialize the scraper with an auth_token.

        Args:
            auth_token: The auth_token cookie value from x.com.
        """
        try:
            from Scweet import Scweet
        except ImportError:
            raise ImportError(
                "Scweet is required for Twitter scraping. "
                "Install it with: pip install -U Scweet"
            )

        self.auth_token = auth_token
        self._scweet = Scweet(auth_token=auth_token)
        logger.info("Scweet client initialized")

    @classmethod
    def from_cookies(cls, cookies_path: str) -> "TwitterScraper":
        """
        Create a scraper from a cookies JSON file.

        Supports two formats:
        - Browser export: [{"name": "auth_token", "value": "..."}, ...]
        - Flat dict: {"auth_token": "...", "ct0": "..."}

        Args:
            cookies_path: Path to the cookies JSON file.

        Returns:
            TwitterScraper instance.
        """
        with open(cookies_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Normalize format
        if isinstance(raw, list):
            cookies = {
                item["name"]: item["value"]
                for item in raw
                if isinstance(item, dict) and "name" in item and "value" in item
            }
        elif isinstance(raw, dict):
            cookies = raw
        else:
            raise ValueError(f"Unrecognized cookie format in {cookies_path}")

        if "auth_token" not in cookies:
            raise ValueError(
                f"'auth_token' not found in {cookies_path}. "
                "Make sure you're logged in to x.com before exporting cookies."
            )

        logger.info(f"Loaded auth_token from {cookies_path}")
        return cls(auth_token=cookies["auth_token"])

    @classmethod
    def from_default_cookies(cls) -> Optional["TwitterScraper"]:
        """
        Try to create a scraper from the default cookies path.

        Returns:
            TwitterScraper instance, or None if no cookies found.
        """
        if not os.path.exists(cls.COOKIES_PATH):
            return None

        try:
            return cls.from_cookies(cls.COOKIES_PATH)
        except Exception as e:
            logger.error(f"Failed to load cookies: {e}")
            return None

    def search(
        self,
        query: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 500,
        lang: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Search for tweets matching a query.

        Args:
            query: Search query (supports Twitter search operators).
            since: Start date in "YYYY-MM-DD" format.
            until: End date in "YYYY-MM-DD" format.
            limit: Maximum number of tweets to collect.
            lang: Language filter (e.g., "id" for Indonesian).

        Returns:
            DataFrame with columns [text, timestamp, source, user,
            tweet_id, likes, retweets].
        """
        logger.info(f"Searching tweets for: '{query}' (limit={limit})")

        # Build Scweet search params
        kwargs = {
            "since": since,
            "limit": limit,
        }
        if until:
            kwargs["until"] = until
        if lang:
            kwargs["lang"] = lang

        # Run the search
        tweets = self._scweet.search(query, **kwargs)

        # Convert to standardized DataFrame
        return self._to_dataframe(tweets)

    def scrape_topic(
        self,
        topic: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 1000,
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Scrape tweets about a topic with optional date filtering.

        Args:
            topic: Search keyword or query.
            start_date: Start date in "YYYY-MM-DD" format.
            end_date: End date in "YYYY-MM-DD" format.
            max_results: Maximum tweets to collect.
            save_raw: Whether to save raw data to disk.

        Returns:
            DataFrame with tweet data.
        """
        df = self.search(
            query=topic,
            since=start_date,
            until=end_date,
            limit=max_results,
        )

        if save_raw and not df.empty:
            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = topic.replace(" ", "_")[:50]
            filename = f"twitter_{safe_topic}_{timestamp}.json"

            output_path = raw_dir / filename
            df.to_json(output_path, orient="records", date_format="iso", indent=2)
            logger.info(f"Raw data saved to {output_path}")

        return df

    def _to_dataframe(self, tweets) -> pd.DataFrame:
        """Convert Scweet result to standardized DataFrame."""
        if tweets is None:
            return pd.DataFrame()

        # Scweet returns a list of Tweet-like objects or dicts
        rows = []
        for tweet in tweets:
            try:
                if isinstance(tweet, dict):
                    row = {
                        "text": tweet.get("text", tweet.get("full_text", "")),
                        "timestamp": tweet.get("timestamp", tweet.get("created_at", "")),
                        "source": "twitter",
                        "user": tweet.get("user", tweet.get("username", "")),
                        "tweet_id": str(tweet.get("tweet_id", tweet.get("id", ""))),
                        "likes": tweet.get("likes", tweet.get("favorite_count", 0)),
                        "retweets": tweet.get("retweets", tweet.get("retweet_count", 0)),
                    }
                else:
                    # Object with attributes
                    row = {
                        "text": getattr(tweet, "text", "") or getattr(tweet, "full_text", ""),
                        "timestamp": getattr(tweet, "timestamp", "") or getattr(tweet, "created_at", ""),
                        "source": "twitter",
                        "user": getattr(tweet, "user", "") or getattr(tweet, "username", ""),
                        "tweet_id": str(getattr(tweet, "tweet_id", "") or getattr(tweet, "id", "")),
                        "likes": getattr(tweet, "likes", 0) or getattr(tweet, "favorite_count", 0),
                        "retweets": getattr(tweet, "retweets", 0) or getattr(tweet, "retweet_count", 0),
                    }

                if row["text"]:
                    rows.append(row)

            except Exception as e:
                logger.debug(f"Skipping unparseable tweet: {e}")
                continue

        df = pd.DataFrame(rows)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(
                f"Collected {len(df)} tweets "
                f"({df['timestamp'].min()} to {df['timestamp'].max()})"
            )

        return df
