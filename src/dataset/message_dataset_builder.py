
from .dataset_builder import DatasetBuilder
from .message_feature_extractor import MessageFeatureExtractor


class MessageDatasetBuilder(DatasetBuilder):

    def __init__(self,users, posts):
        super().__init__(users,posts)
        self.extractor = MessageFeatureExtractor()

    def build_features(self, A_id, S_id, P_id, post, label):
        return self.extractor.calc_features(
            A_id=A_id,
            S_id=S_id,
            P_id=P_id,
            post=post,
            label=label
        )
    
