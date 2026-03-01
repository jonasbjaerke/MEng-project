#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests
import json
import os
from datetime import datetime, timezone
import random
import asyncio, aiohttp
import aiolimiter
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path

import aiohttp
import asyncio
from datetime import datetime, timezone
from tqdm import tqdm
import aiolimiter
import os
from funcs import save_to_json, get_json


# In[24]:


def flatten_posts_dict(posts_by_tag):
    flat_posts = {}
    for posts in posts_by_tag.values():
        for uri, post in posts.items():
            flat_posts[uri] = post
    return flat_posts


# In[ ]:


# async def fetch_follows(session, did, author_set, retries=3, first_page_only=False): #change to true for faster runtime
#     follows = set()
#     cursor = None
#     delay = 1

#     for attempt in range(retries):
#         try:
#             while True:
#                 params = {"actor": did, "limit": 100}
#                 if cursor:
#                     params["cursor"] = cursor

#                 async with session.get(FOLLOW_API, params=params, headers=HEADERS) as r:
#                     if r.status != 200:
#                         if 500 <= r.status < 600 and attempt < retries - 1:
#                             await asyncio.sleep(delay)
#                             delay *= 2
#                             continue
#                         return did, []

#                     data = await r.json()
#                     follows.update(
#                         u["did"] for u in data.get("follows", [])
#                         if u.get("did") in author_set
#                     )

#                     cursor = data.get("cursor")
#                     if not cursor or first_page_only:
#                         break

#             return did, list(follows)

#         except Exception:
#             if attempt < retries - 1:
#                 await asyncio.sleep(delay)
#                 delay *= 2
#                 continue
#             return did, []


# In[2]:


def load_posts_dict(file_paths):
    all_posts = {}

    for path in file_paths:
        source = os.path.splitext(os.path.basename(path))[0]

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    post = json.loads(line.strip())
                except Exception:
                    continue

                post_id = (post.get("uri"))

                if not post_id:
                    continue

                post["hashtag"] = source
                all_posts[post_id] = post

    return all_posts


# In[5]:


folder = Path("hashtags_mini")
files = folder.glob("*.jsonl")

posts = load_posts_dict(files)


# In[8]:


def count_posts_with_reposts(posts_dict):
    count = 0

    for post in posts_dict.values():
        if (post.get("repostCount")) > 0:
            count += 1

    print(f"Posts with repost_count > 0: {count}")
    return count
count_posts_with_reposts(posts)


# In[10]:


REPOST_API  = "https://public.api.bsky.app/xrpc/app.bsky.feed.getRepostedBy"
PROFILE_API = "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile"
FEED_API    = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
FOLLOW_API  = "https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows"

HEADERS = {"User-Agent": "repost-prediction-research/1.0"}



def parse_dt(ts: str):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)

def get_author_dids(posts_dict):
    authors = set()
    for post in posts_dict.values():
        did = (
            post.get("author", {}).get("did")
            or post.get("post", {}).get("author", {}).get("did")
        )
        if did:
            authors.add(did)
    return authors



# In[12]:


RPRP = 1


# In[ ]:


# -------------------------
# FETCH REPOSTERS
# -------------------------


async def fetch_reposters(session, uri):
    try:
        async with session.get(
            REPOST_API,
            params={"uri": uri, "limit": 100},
            headers=HEADERS
        ) as r:
            if r.status != 200:
                return []
            data = await r.json()
            return [u["did"] for u in data.get("repostedBy", [])]
    except Exception:
        return []


async def collect_reposters(posts, concurrency=25, reqs_per_sec=100):
    posts_, tasks = [], []

    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:

        rate_limiter = aiolimiter.AsyncLimiter(reqs_per_sec, 1)

        async def limited_fetch(uri):
            async with rate_limiter:
                return await fetch_reposters(session, uri)

        for uri, post in posts.items():
            if post.get("repostCount", 0) > 0:
                posts_.append(uri)
                tasks.append(limited_fetch(uri))
            else:
                posts[uri]["reposted_by"] = []

        stored_reposters = set()
        failed_posts = 0

        pbar = tqdm(posts_, desc="Fetching reposters", unit="post")

        for uri, task in zip(pbar, asyncio.as_completed(tasks)):
            reposters = await task

            posts[uri]["reposted_by"] = reposters

            if not reposters:
                failed_posts += 1
                pbar.set_postfix(
                    failed=f"{failed_posts}/{len(posts_)}",
                    refresh=False
                )
                continue


            # Randomly sample up to RPRP reposters
            if len(reposters) > RPRP:
                sampled_reposters = random.sample(reposters, RPRP)
            else:
                sampled_reposters = reposters

            posts[uri]["stored_reposters"] = sampled_reposters

            stored_reposters.update(sampled_reposters)


    print(
        f"\nReposter fetch summary: "
        f"{failed_posts}/{len(posts_)} posts "
        f"({failed_posts/len(posts_):.1%}) returned no reposters"
    )

    return stored_reposters



# -------------------------
# FOLLOW FETCHING
# -------------------------

async def fetch_follows_probabilistic(
    session,
    did,
    author_set,
    retries=3,
    first_page_only=False,
    min_pages_before_stop=5,
    expected_hit_threshold=0.02
):
    """
    Fetch authors followed by user `did`.

    Probabilistic early stop:
    - Estimate hit density from observed pages
    - Stop if expected future hits are very small
    """

    follows = set()
    cursor = None
    delay = 1

    pages_seen = 0
    total_hits = 0

    for attempt in range(retries):
        try:
            while True:
                params = {"actor": did, "limit": 100}
                if cursor:
                    params["cursor"] = cursor

                async with session.get(
                    FOLLOW_API,
                    params=params,
                    headers=HEADERS
                ) as r:

                    if r.status != 200:
                        if 500 <= r.status < 600 and attempt < retries - 1:
                            await asyncio.sleep(delay)
                            delay *= 2
                            continue
                        return did, list(follows)

                    data = await r.json()
                    page_follows = data.get("follows", [])

                    page_hits = 0
                    for u in page_follows:
                        u_did = u.get("did")
                        if u_did in author_set:
                            if u_did not in follows:
                                follows.add(u_did)
                                page_hits += 1

                    pages_seen += 1
                    total_hits += page_hits

                    cursor = data.get("cursor")

                    # If no more pages or first_page_only → stop
                    if not cursor or first_page_only:
                        break

                    if pages_seen >= min_pages_before_stop:

                        hit_density = total_hits / (pages_seen * 100)

                        if hit_density < expected_hit_threshold:
                            return did, list(follows)

            return did, list(follows)

        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            return did, list(follows)
        
        

async def collect_follow_relations(
    user_dids,
    posts,
    concurrency=100,
    reqs_per_sec=50
):
    
    user_dids = list(user_dids)

    total = len(user_dids)
    author_set = get_author_dids(posts)
    counter = {"done": 0}

    rate_limiter = aiolimiter.AsyncLimiter(reqs_per_sec, 1)
    connector = aiohttp.TCPConnector(limit=300, limit_per_host=50, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_read=5)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        sem = asyncio.Semaphore(concurrency)

        async def limited_fetch(did):
            async with sem, rate_limiter:
                rep, authors = await fetch_follows_probabilistic(session, did, author_set)
                counter["done"] += 1
                if counter["done"] % 100 == 0 or counter["done"] == total:
                    print(
                        f"\rFetching followed authors: "
                        f"{counter['done']}/{total} "
                        f"({counter['done']/total:.1%})",
                        end="",
                        flush=True
                    )
                return rep, authors

        tasks = [limited_fetch(d) for d in user_dids]
        responses = await asyncio.gather(*tasks)

    print()
    return {r: authors for r, authors in responses if authors}

# -------------------------
# FETCH PROFILE
# -------------------------

async def fetch_profile(session, did):
    async with session.get(PROFILE_API, params={"actor": did}, headers=HEADERS) as r:
        if r.status != 200:
            return None
        return await r.json()

# -------------------------
# FETCH USER HISTORY (UNCHANGED)
# -------------------------

async def fetch_user_history(session, did, limit=50):
    history = []
    cursor = None

    while len(history) < limit:
        params = {"actor": did, "limit": min(100, limit - len(history))}
        if cursor:
            params["cursor"] = cursor

        async with session.get(FEED_API, params=params, headers=HEADERS) as r:
            if r.status != 200:
                break

            data = await r.json()
            feed = data.get("feed", [])

            for item in feed:
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
                    for f in facets
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

                if len(history) >= limit:
                    break

            cursor = data.get("cursor")
            if not cursor:
                break

    return history

# -------------------------
# MAIN BUILDER
# -------------------------

async def build_users_from_posts(posts):
    user_dids = await collect_reposters(posts)
    user_dids |= get_author_dids(posts)

    save_to_json(name="newposts",dict=posts)

    followed_authors = await collect_follow_relations(
        user_dids, posts
    )


    users = {}

    connector = aiohttp.TCPConnector(limit=200, limit_per_host=50, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_read=5)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout
    ) as session:

        sem = asyncio.Semaphore(100)

        async def process_user(did):
            async with sem:
                try:
                    profile = await fetch_profile(session, did)
                    if not profile:
                        return

                    history = await fetch_user_history(session, did)
                    created = profile.get("createdAt")

                    age_days = (
                        max(1, (datetime.now(timezone.utc) - parse_dt(created)).days)
                        if created else None
                    )

                    users[did] = {
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
                      #  "reposted_posts": reposter_dict.get(did, []),
                        "follows_authors": followed_authors.get(did, []),
                    }

                except Exception:
                    return

        tasks = [process_user(d) for d in user_dids]

        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Building users",
            unit="user"
        ):
            await task

    return users


# In[ ]:


# IDEAS FOR LATER: number of reposters per post should be picked proportional to repostcount to signal these posts are more likely to be reposted


# In[16]:


users = await build_users_from_posts(posts)


# In[17]:


with open("userdata_mini.json", "w", encoding="utf-8") as f:
    json.dump(users, f, ensure_ascii=False, indent=2)


# In[ ]:


# async def recompute_follows_authors(
#     users,
#     posts_dict,
#     concurrency=150,
#     reqs_per_sec=70,
#     first_page_only=False
# ):
#     """
#     Recomputes follows_authors for ALL users.
#     Overwrites existing users[did]["follows_authors"].
#     """

#     author_set = get_author_dids(posts_dict)
#     user_dids = list(users.keys())

#     rate_limiter = aiolimiter.AsyncLimiter(reqs_per_sec, 1)
#     connector = aiohttp.TCPConnector(limit=300, limit_per_host=50, ttl_dns_cache=300)
#     timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_read=5)

#     async with aiohttp.ClientSession(
#         connector=connector,
#         timeout=timeout
#     ) as session:

#         sem = asyncio.Semaphore(concurrency)

#         async def limited_fetch(did):
#             async with sem, rate_limiter:
#                 rep, authors = await fetch_follows(
#                     session,
#                     did,
#                     author_set,
#                     first_page_only=first_page_only
#                 )
#                 return rep, authors

#         tasks = [limited_fetch(did) for did in user_dids]

#         for future in tqdm(
#             asyncio.as_completed(tasks),
#             total=len(tasks),
#             desc="Recomputing follows",
#             unit="user"
#         ):
#             did, authors = await future

#             users[did]["follows_authors"] = authors

#     return users

