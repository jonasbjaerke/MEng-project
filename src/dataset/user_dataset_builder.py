

from .dataset_builder import DatasetBuilder
from .user_feature_extractor import UserFeatureExtractor


class UserDatasetBuilder(DatasetBuilder):

    def __init__(self, users, posts):
        super().__init__(users, posts)
        self.extractor = UserFeatureExtractor(users)

    def build_features(self, A_id, S_id, P_id, post, label):
        return self.extractor.build_features(
            A_id=A_id,
            S_id=S_id,
            P_id=P_id,
            post=post,
            label=label
        )