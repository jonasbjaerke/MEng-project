

def build_text_dict(posts, users):
        post_text_dict = {}

        for user in users.values():
            for activity in user.get("history", []):
                uri = activity.get("post_uri")
                text = activity.get("text")
                if uri and text:
                    post_text_dict[uri] = text

        for uri, post in posts.items():
            text = post.get("text")
            if uri and text:
                post_text_dict[uri] = text
        return post_text_dict