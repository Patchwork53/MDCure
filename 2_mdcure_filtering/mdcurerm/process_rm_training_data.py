import logging
from pathlib import Path
from distilabel.distiset import create_distiset
from distilabel.steps import LoadDataFromDisk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process(raw_path: Path, data_path: str, json_path: str) -> None:
    """
    Process the dataset from the raw path, save it to disk, and convert the training data to JSON.

    Args:
        raw_path (Path): The path to the raw data.
        data_path (str): The path where the processed data will be saved.
        json_path (str): The path where the JSON file will be saved.
    """
    try:
        logging.info(f"Processing dataset from {raw_path}")
        ds = create_distiset(raw_path)

        ds.save_to_disk(
            data_path,
            save_card=True,
            save_pipeline_config=True,
            save_pipeline_log=True
        )
        logging.info(f"Dataset saved to {data_path}")

        df = ds['default']['train'].to_pandas()
        df.to_json(json_path, orient='records', lines=True)
        logging.info(f"Training data saved to JSONL at {json_path}")

    except Exception as e:
        logging.error(f"An error occurred while processing the dataset: {e}")

if __name__ == "__main__":
    paths = [
        (Path("./cache/prometheus-reward-model/relevance/data"), "./relevance", "./train_relevance.jsonl"),
        (Path("./cache/prometheus-reward-model/clarity/data"), "./clarity", "./train_clarity.jsonl"),
        (Path("./cache/prometheus-reward-model/comprehensive/data"), "./comprehensive", "./train_comprehensive.jsonl"),
        (Path("./cache/prometheus-reward-model/conciseness/data"), "./conciseness", "./train_conciseness.jsonl"),
        (Path("./cache/prometheus-reward-model/multi-doc/data"), "./multi_doc", "./train_multi_doc.jsonl")
    ]

    for raw_path, data_path, json_path in paths:
        process(raw_path, data_path, json_path)
