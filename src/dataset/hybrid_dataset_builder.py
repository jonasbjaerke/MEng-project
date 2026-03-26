from .dataset_builder import DatasetBuilder
from .message_feature_extractor import MessageFeatureExtractor
from .user_feature_extractor import UserFeatureExtractor


class HybridDatasetBuilder(DatasetBuilder):

    def __init__(self, users, posts):
        super().__init__(users, posts)
        self.M_extractor = MessageFeatureExtractor()
        self.U_extractor = UserFeatureExtractor(users)

    def build_features(self, A_id, S_id, P_id, post, label):
        M_features = self.M_extractor.calc_features(
            A_id=A_id,
            S_id=S_id,
            P_id=P_id,
            post=post,
            label=label
        )

        U_features = self.U_extractor.calc_features(
            A_id=A_id,
            S_id=S_id,
            P_id=P_id,
            post=post,
            label=label
        )

        merged = dict(M_features)
        for k, v in U_features.items():
            if k not in merged:
                merged[k] = v

        return merged
    

                


    