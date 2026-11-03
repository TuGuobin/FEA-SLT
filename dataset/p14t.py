from dataset.features import FeatureDataset


class Phoenix14T(FeatureDataset):
    language = "German"
    dataset_name = "Phoenix14T"

    def _normalize_text(self, text):
        text = text.strip()
        return text if text.endswith(".") else f"{text}."
