from .dataset_builder import DatasetBuilder
from .bert_classifier_extractor import MessageBertClassifierExtractor


class MessageBertDatasetBuilder(DatasetBuilder):

    def __init__(self, users, posts):
        super().__init__(users, posts)
        self.extractor = MessageBertClassifierExtractor()

    def build_features(self, A_id, S_id, P_id, post, label):
        return self.extractor.calc_features(
            A_id=A_id,
            S_id=S_id,
            P_id=P_id,
            post=post,
            label=label
        )