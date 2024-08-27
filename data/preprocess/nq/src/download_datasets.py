from src.classes.qadataset import MRQANaturalQuetsionsDataset

# NB: Feel free to add custom datasets here.
DATASETS = {
    # MRQA Datasets available at: https://github.com/mrqa/MRQA-Shared-Task-2019
    "MRQANaturalQuestionsTrain": (
        MRQANaturalQuetsionsDataset,
        "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz",
    ),
    "MRQANaturalQuestionsDev": (
        MRQANaturalQuetsionsDataset,
        "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz",
    ),
}


if __name__ == "__main__":
    for dataset in DATASETS:
        dataset_class, url_or_path = DATASETS[dataset]
        dataset = dataset_class.new(dataset, url_or_path)
