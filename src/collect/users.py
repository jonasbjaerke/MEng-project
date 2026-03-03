# src/data/users.py

import asyncio
import random
from datetime import datetime, timezone
from typing import Dict, Set
from tqdm import tqdm

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
        history_limit: int = 50,
        rps: int = 100,
        concurrency: int = 100,
    ):
        """
        rps + concurrency fully control performance.
        Change them here — nowhere else.
        """
        self.posts = posts
        self.users = {}
        self.rprp = rprp
        self.history_limit = history_limit
        self.user_dids: Set[str] = set()
        self.follows = {}
        self.rps = rps
        self.concurrency = concurrency

    # =====================================================
    # ENTRYPOINT
    # =====================================================

    async def collect(self):

        async with BlueskyAsyncClient(
            rps=self.rps,
            concurrency=self.concurrency
        ) as client:

            self.client = client

            await self._collect_reposters()
            await self._collect_follow_relations()
            await self._collect_profiles_and_history()

        return self.users

    # =====================================================
    # HELPERS
    # =====================================================

    def _author_dids(self):
        return {
            did
            for p in self.posts.values()
            if (did := p.get("author", {}).get("did"))
        }

    def _parse_dt(self, ts):
        return datetime.fromisoformat(
            ts.replace("Z", "+00:00")
        ).astimezone(timezone.utc)

    # =====================================================
    # REPOSTERS
    # =====================================================

    async def _collect_reposters(self):

        async def fetch(uri):
            data = await self.client.get(
                REPOST_API,
                params={"uri": uri, "limit": 100},
                headers=HEADERS
            )
            if not data:
                return uri, []
            return uri, [u["did"] for u in data.get("repostedBy", [])]

        uris = [
            u for u, p in self.posts.items()
            if p.get("repostCount", 0) > 0
        ]

        tasks = [fetch(u) for u in uris]

        failed=0
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Fetching reposters",
            unit="post"
        ):
            uri, reposters = await task

            self.posts[uri]["reposted_by"] = reposters

            if not reposters:
                failed+=1
                continue

            sampled = random.sample(
                reposters,
                min(len(reposters), self.rprp)
            )

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

        async def fetch(did):

            follows = set()
            cursor = None
            pages = 0
            hits = 0

            while True:

                data = await self.client.get(
                    FOLLOW_API,
                    params={
                        "actor": did,
                        "limit": 100,
                        "cursor": cursor
                    } if cursor else {
                        "actor": did,
                        "limit": 100
                    },
                    headers=HEADERS
                )

                if not data:
                    break

                page = data.get("follows", [])

                for u in page:
                    if u.get("did") in author_set:
                        follows.add(u["did"])
                        hits += 1

                pages += 1
                cursor = data.get("cursor")

                if not cursor:
                    break

                # probabilistic early stop
                if pages >= 5:
                    density = hits / (pages * 100)
                    if density < 0.02:
                        break

            return did, list(follows)

        tasks = [fetch(d) for d in self.user_dids]

        results = []

        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Fetching follows",
            unit="user"
        ):
            results.append(await task)

        self.follows = {
            did: follows
            for did, follows in results
            if follows
            }

    # =====================================================
    # PROFILE + HISTORY
    # =====================================================

    async def _collect_profiles_and_history(self):

        async def process(did):

            # Fetch profile + history concurrently
            profile_task = self.client.get(
                PROFILE_API,
                params={"actor": did},
                headers=HEADERS
            )

            history_task = self._fetch_history(did)

            profile, history = await asyncio.gather(
                profile_task,
                history_task
            )

            if not profile:
                return

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
                "follows_authors": self.follows.get(did),
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

            params = {
                "actor": did,
                "limit": min(100, self.history_limit - len(history))
            }

            if cursor:
                params["cursor"] = cursor

            data = await self.client.get(
                FEED_API,
                params=params,
                headers=HEADERS
            )

            if not data:
                break

            feed_items = data.get("feed") or []
            if not feed_items:
                break

            for item in feed_items:

                post = item.get("post")
                if not post:
                    continue

                record = post.get("record") or {}
                reason = item.get("reason") or {}

                # -----------------------------------------
                # Activity Type
                # -----------------------------------------

                if reason.get("$type", "").endswith("reasonRepost"):
                    activity_type = "repost"
                    reposted_at = reason.get("indexedAt")
                elif record.get("reply"):
                    activity_type = "reply"
                    reposted_at = None
                else:
                    activity_type = "post"
                    reposted_at = None

                # -----------------------------------------
                # Parent info (only for replies)
                # -----------------------------------------

                parent_post_uri = None
                parent_author_did = None

                if activity_type == "reply":
                    reply = record.get("reply") or {}
                    parent = reply.get("parent") or {}

                    parent_post_uri = parent.get("uri")
                    parent_author_did = (
                        (parent.get("author") or {}).get("did")
                    )

                # -----------------------------------------
                # Text + link detection
                # -----------------------------------------

                text = record.get("text") or ""
                has_links = ("http://" in text) or ("https://" in text)

                # -----------------------------------------
                # Media detection
                # -----------------------------------------

                media_type = None
                media_count = 0

                embed = record.get("embed") or {}
                embed_type = embed.get("$type", "")

                if "embed.images" in embed_type:
                    media_type = "image"
                    media_count = len(embed.get("images") or [])

                elif "embed.video" in embed_type:
                    media_type = "video"
                    media_count = 1

                elif "embed.external" in embed_type:
                    media_type = "external"
                    media_count = 1

                elif "embed.recordWithMedia" in embed_type:
                    media_type = "mixed"
                    media_count = 1


                history.append({
                    "activity_type": activity_type,
                    "created_at": record.get("createdAt"),
                    "reposted_at": reposted_at,
                    "post_uri": post.get("uri"),
                    "post_author_did": (post.get("author") or {}).get("did"),
                    "parent_post_uri": parent_post_uri,
                    "parent_author_did": parent_author_did,
                    "text": text,
                    "like_count": post.get("likeCount"),
                    "repost_count": post.get("repostCount"),
                    "reply_count": post.get("replyCount"),
                    "quote_count": post.get("quoteCount"),
                    "has_links": has_links,
                    "media_type": media_type,
                    "media_count": media_count,
                })

                if len(history) >= self.history_limit:
                    break

            cursor = data.get("cursor")
            if not cursor:
                break

        return history