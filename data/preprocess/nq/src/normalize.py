from src.classes.qadataset import MRQANaturalQuetsionsDataset

DATASETS = {
    "MRQANaturalQuestionsTrain-closedbookfiltered": (
        MRQANaturalQuetsionsDataset,
        "datasets/original/MRQANaturalQuestionsTrain-closedbookfiltered.jsonl.gz",
    ),
    "MRQANaturalQuestionsDev-closedbookfiltered": (
        MRQANaturalQuetsionsDataset,
        "datasets/original/MRQANaturalQuestionsDev-closedbookfiltered.jsonl.gz",
    ),
}

wikidata = "wikidata/entity_info.json.gz"
ner_model = "models/kc-ner-model"


if __name__ == "__main__":
    for dataset in DATASETS:
        dataset_class, url_or_path = DATASETS[dataset]
        dataset = dataset_class.new(dataset, url_or_path)
        dataset.preprocess(wikidata, ner_model)
