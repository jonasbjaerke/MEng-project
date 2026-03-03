
from datetime import datetime, timezone
from ..utils import parse_dt


class UserFeatureExtractor:
    """
    Computes all user-level and interaction features
    for a (A, S, P) triplet.
    """

    def __init__(self, users):
        self.users = users
        self.max_posts = max(
            u["stats"]["posts"] or 1
            for u in users.values()
        )

    # =====================================================
    # History statistics
    # =====================================================

    def history_stats(self, history, exclude_post_id):

        if not history:
            return 0.0, None, 0, 0, 0, 0

        hist_len = sum(
            1 for h in history
            if h.get("post_uri") != exclude_post_id and h.get("parent_post_uri") != exclude_post_id
        )

        reposts = sum(
            1 for h in history
            if h.get("activity_type") == "repost" or h.get("activity_type") == "reply"
            and h.get("post_uri") != exclude_post_id 
            and h.get("parent_post_uri") != exclude_post_id
        )

        repost_pct = reposts / hist_len if hist_len else 0.0

        times = []
        for h in history:
            ts = (
                h.get("reposted_at")
                if h.get("activity_type") == "repost"
                else h.get("created_at")
            )
            parsed = parse_dt(ts)
            if parsed:
                times.append(parsed)

        if len(times) >= 2:
            times.sort()
            span_days = (times[-1] - times[0]).total_seconds() / 86400
            avg_interval = span_days / (len(times) - 1)
        else:
            avg_interval = None

        total_likes = total_reposts = total_replies = total_quotes = post_count = 0

        for h in history:
            if h.get("activity_type") == "post" or h.get("activity_type") == "reply":  #only consider likes made by your post or reply as repost is not "your post"
                total_likes += h.get("like_count", 0)
                total_reposts += h.get("repost_count", 0)
                total_replies += h.get("reply_count", 0)
                total_quotes += h.get("quote_count", 0)
                post_count += 1

        if post_count:
            return (
                repost_pct,
                avg_interval,
                total_reposts / post_count,
                total_quotes / post_count,
                total_replies / post_count,
                total_likes / post_count,
            )

        return repost_pct, avg_interval, 0, 0, 0, 0

    # =====================================================
    # Interaction helpers
    # =====================================================

    @staticmethod
    def mention_stats(history, handle, exclude_post_id):
        if not history or not handle:
            return 0, 0.0

        total = len(history)

        count = sum(
            handle in (h.get("text") or "")
            for h in history
            if h.get("post_uri") != exclude_post_id
            and h.get("parent_post_uri") != exclude_post_id
        )

        return count, count / total if total else 0.0

    @staticmethod
    def reposts_from_author(history, author_did, exclude_post_id):
        return sum(
            1
            for h in history
            if (h.get("activity_type") == "repost" or h.get("activity_type") == "reply")
            and (h.get("post_author_did") == author_did or h.get("parent_author_did") == author_did)
            and h.get("post_uri") != exclude_post_id
            and h.get("parent_post_uri") != exclude_post_id
        )

    # =====================================================
    # Main feature builder
    # =====================================================

    def build_features(self, A_id, S_id, P_id, post, label):

        A = self.users[A_id]
        S = self.users[S_id]

        P_time = parse_dt(post.get("indexedAt"))
        if not P_time:
            return None

        # History stats
        stats_R = self.history_stats(A["history"], P_id)
        stats_S = self.history_stats(S["history"], P_id)

        # Mentions
        A_m_S, A_m_S_per = self.mention_stats(
            A["history"],
            S["profile"]["handle"],
            P_id
        )

        S_m_A, S_m_A_per = self.mention_stats(
            S["history"],
            A["profile"]["handle"],
            P_id
        )

        row = {
            "A_id": A_id,
            "S_id": S_id,
            "P_id": P_id,
            "hashtag": post.get("hashtag"),
            "label": label,

            # Interaction
            "U-P_R_FollowS": int(S_id in A.get("follows_authors", [])),
            "U-HA_R_MentionS": A_m_S,
            "U-HA_R_MentionPerS": A_m_S_per,
            "U-HA_S_MentionR": S_m_A,
            "U-HA_S_MentionPerR": S_m_A_per,
            "U-HA-R_repostsS":
                self.reposts_from_author(A["history"], S_id, P_id),
            "U-P_SR_followersDiff":
                S["stats"]["followers"] - A["stats"]["followers"],
            "U-P_R_activeBeforeP":
                int(parse_dt(A["profile"].get("created_at")) < P_time)
                if A["profile"].get("created_at") else 0,
        }

        # Profile features
        for prefix, user in [("R", A), ("S", S)]:
            age = user["stats"]["account_age_days"] or 1
            posts = user["stats"]["posts"] or 0

            row.update({
                f"U-P_{prefix}_AccountAge": age,
                f"U-P_{prefix}_FollowerNum": user["stats"]["followers"],
                f"U-P_{prefix}_FolloweeNum": user["stats"]["follows"],
                f"U-P_{prefix}_TweetNum": posts,
                f"U-P_{prefix}_SpreadActivity": posts / self.max_posts,
                f"U-P_{prefix}_FollowerNumDay":
                    user["stats"]["followers"] / age,
                f"U-P_{prefix}_FolloweeNumDay":
                    user["stats"]["follows"] / age,
                f"U-P_{prefix}_TweetNumDay":
                    posts / age,
                f"U-P_{prefix}_ProfileUrl":
                    int("http" in (user["profile"]["description"] or "")),
            })

        # History features
        row.update({
            "U-HA_R_RetweetPercent": stats_R[0],
            "U-HA_R_AverageInterval": stats_R[1],
            "U-HA_R_RetweetedRate": stats_R[2],
            "U-HA_R_QuotedRate": stats_R[3],
            "U-HA_R_RepliedRate": stats_R[4],
            "U-HA_R_LikedRate": stats_R[5],
            "U-HA_R_TweetNum": len(A["history"]),

            "U-HA_S_RetweetPercent": stats_S[0],
            "U-HA_S_AverageInterval": stats_S[1],
            "U-HA_S_RetweetedRate": stats_S[2],
            "U-HA_S_QuotedRate": stats_S[3],
            "U-HA_S_RepliedRate": stats_S[4],
            "U-HA_S_LikedRate": stats_S[5],
            "U-HA_S_TweetNum": len(S["history"]),
        })

        return row