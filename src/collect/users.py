# src/data/users.py

import asyncio
import random
from datetime import datetime, timezone
from typing import Dict, Set
import json
from pathlib import Path
from tqdm import tqdm
import aiolimiter

from .async_client import BlueskyAsyncClient


REPOST_API  = "https://public.api.bsky.app/xrpc/app.bsky.feed.getRepostedBy"
PROFILE_API = "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile"
FEED_API    = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
FOLLOW_API  = "https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows"

HEADERS = {"User-Agent": "repost-prediction-research/1.0"}


class UserDataCollector:

    def __init__(
        self,
        posts: Dict[str, dict],
        rprp: int = 3,
        reposter_rps: int = 100,
        follow_rps: int = 50,
        history_limit: int = 50,
    ):
        self.posts = posts
        self.users = {}
        self.rprp = rprp
        self.reposter_rps = reposter_rps
        self.follow_rps = follow_rps
        self.history_limit = history_limit
        self.user_dids: Set[str] = set()

    # =====================================================
    # PUBLIC ENTRYPOINT
    # =====================================================

    async def collect(self):

        async with BlueskyAsyncClient() as client:
            self.client = client

            await self._collect_reposters()
            followed = await self._collect_follow_relations()
            await self._collect_profiles_and_history(followed)

        return self.users

    # =====================================================
    # HELPERS
    # =====================================================

    def _parse_dt(self, ts):
        return datetime.fromisoformat(
            ts.replace("Z", "+00:00")
        ).astimezone(timezone.utc)

    def _author_dids(self):
        return {
            did
            for p in self.posts.values()
            if (did := p.get("author", {}).get("did"))
        }

    # =====================================================
    # REPOSTERS
    # =====================================================

    async def _collect_reposters(self):

        limiter = aiolimiter.AsyncLimiter(self.reposter_rps, 1)

        async def fetch(uri):
            async with limiter:
                data = await self.client.get(
                    REPOST_API,
                    params={"uri": uri, "limit": 100},
                    headers=HEADERS
                )
                if not data:
                    return []
                return [u["did"] for u in data.get("repostedBy", [])]

        uris = [u for u, p in self.posts.items() if p.get("repostCount", 0) > 0]

        failed = 0
        for uri, task in tqdm(
            zip(uris, asyncio.as_completed([fetch(u) for u in uris])),
            total=len(uris),
            desc="Fetching reposters",
            unit="post"
        ):
            reposters = await task

            self.posts[uri]["reposted_by"] = reposters

            if not reposters:
                failed += 1
                continue

            sampled = random.sample(reposters, min(len(reposters), self.rprp))
            self.posts[uri]["stored_reposters"] = sampled
            self.user_dids.update(sampled)

        print(
            f"\nReposter fetch summary: "
            f"({failed/len(uris):.1%}) returned no reposters"
        )

        self.user_dids |= self._author_dids()

    # =====================================================
    # FOLLOW RELATIONS
    # =====================================================

    async def _collect_follow_relations(self):

        author_set = self._author_dids()
        limiter = aiolimiter.AsyncLimiter(self.follow_rps, 1)
        total = len(self.user_dids)
        counter = 0

        async def fetch(did):

            nonlocal counter
            follows = set()
            cursor = None
            pages = 0
            hits = 0

            while True:

                async with limiter:
                    data = await self.client.get(
                        FOLLOW_API,
                        params={"actor": did, "limit": 100, "cursor": cursor} if cursor
                        else {"actor": did, "limit": 100},
                        headers=HEADERS
                    )

                if not data:
                    break

                page = data.get("follows", [])

                for u in page:
                    if u.get("did") in author_set:
                        if u["did"] not in follows:
                            follows.add(u["did"])
                            hits += 1

                pages += 1
                cursor = data.get("cursor")

                if not cursor:
                    break

                if pages >= 5:
                    density = hits / (pages * 100)
                    if density < 0.02:
                        break

            counter += 1
            if counter % 100 == 0 or counter == total:
                print(
                    f"\rFetching followed authors: "
                    f"{counter}/{total} "
                    f"({counter/total:.1%})",
                    end="",
                    flush=True
                )

            return did, list(follows)

        results = await asyncio.gather(
            *[fetch(d) for d in self.user_dids]
        )

        print()

        return {d: f for d, f in results if f}

    # =====================================================
    # PROFILE + HISTORY
    # =====================================================

    async def _collect_profiles_and_history(self, followed):

        async def process(did):

            profile = await self.client.get(
                PROFILE_API,
                params={"actor": did},
                headers=HEADERS
            )
            if not profile:
                return

            history = await self._fetch_history(did)

            created = profile.get("createdAt")
            age_days = (
                max(
                    1,
                    (datetime.now(timezone.utc) - self._parse_dt(created)).days
                )
                if created else None
            )

            self.users[did] = {
                "profile": {
                    "did": did,
                    "handle": profile.get("handle"),
                    "display_name": profile.get("displayName"),
                    "description": profile.get("description"),
                    "created_at": created,
                },
                "stats": {
                    "followers": profile.get("followersCount"),
                    "follows": profile.get("followsCount"),
                    "posts": profile.get("postsCount"),
                    "account_age_days": age_days,
                },
                "history": history,
                "follows_authors": followed.get(did, []),
            }

        for task in tqdm(
            asyncio.as_completed([process(d) for d in self.user_dids]),
            total=len(self.user_dids),
            desc="Building users",
            unit="user"
        ):
            await task

    async def _fetch_history(self, did):

        history = []
        cursor = None

        while len(history) < self.history_limit:

            data = await self.client.get(
                FEED_API,
                params={"actor": did, "limit": min(100, self.history_limit - len(history)), "cursor": cursor}
                if cursor else
                {"actor": did, "limit": min(100, self.history_limit - len(history))},
                headers=HEADERS
            )

            if not data:
                break

            for item in data.get("feed", []):
                post = item.get("post")
                if not post:
                    continue

                record = post.get("record", {})
                reason = item.get("reason", {})

                if reason.get("$type", "").endswith("reasonRepost"):
                    activity_type = "repost"
                    parent_post_uri = post["uri"]
                    parent_author_did = post["author"]["did"]
                    reposted_at = reason.get("indexedAt")
                elif "reply" not in record:
                    activity_type = "post"
                    parent_post_uri = None
                    parent_author_did = None
                    reposted_at = None
                else:
                    continue

                facets = record.get("facets", [])
                has_links = any(
                    f["features"][0]["$type"].endswith("link")
                    for f in facets if f.get("features")
                )

                embed = record.get("embed", {})
                media_type = None
                media_count = 0

                if embed:
                    et = embed.get("$type", "")
                    if et.endswith("images"):
                        media_type = "image"
                        media_count = len(embed.get("images", []))
                    elif et.endswith("video"):
                        media_type = "video"
                        media_count = 1
                    elif et.endswith("external"):
                        media_type = "external"
                        media_count = 1
                    elif et.endswith("recordWithMedia"):
                        media_type = "mixed"
                        media_count = 1

                history.append({
                    "activity_type": activity_type,
                    "created_at": record.get("createdAt"),
                    "reposted_at": reposted_at,
                    "post_uri": post["uri"],
                    "post_author_did": post["author"]["did"],
                    "parent_post_uri": parent_post_uri,
                    "parent_author_did": parent_author_did,
                    "text": record.get("text", ""),
                    "langs": record.get("langs", []),
                    "like_count": post.get("likeCount"),
                    "repost_count": post.get("repostCount"),
                    "reply_count": post.get("replyCount"),
                    "quote_count": post.get("quoteCount"),
                    "has_links": has_links,
                    "media_type": media_type,
                    "media_count": media_count,
                    "labels": post.get("labels", []),
                })

                if len(history) >= self.history_limit:
                    break

            cursor = data.get("cursor")
            if not cursor:
                break

        return history
    

