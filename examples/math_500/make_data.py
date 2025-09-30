from typing import Any, Dict, List, Tuple, cast

import datasets
import polars as pl

from slime.rollout.rm_hub.math_utils import extract_answer


def load_and_merge_dataset() -> pl.DataFrame:
    """Load HuggingFace dataset and merge train+test splits."""
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source)

    # Merge train and test splits
    merged_dataset = datasets.concatenate_datasets([dataset["train"], dataset["test"]])
    merged_dataset = merged_dataset.filter(lambda row: row["level"] != "Level ?")
    merged_dataset = cast(pl.DataFrame, merged_dataset.to_polars())
    merged_dataset = merged_dataset.with_columns(
        pl.col("level").map_elements(lambda level: int(level.split(" ")[1]), return_dtype=pl.Int32)
    )
    merged_dataset = merged_dataset.with_row_index()

    return merged_dataset


def format_dataset(data: pl.DataFrame):
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}. "

    def process_fn(example: Dict[str, Any]):
        question = example["problem"]
        question = question + " " + instruction_following

        solution = example["solution"]
        answer = extract_answer(solution)

        data = {
            # "prompt": [{"role": "user", "content": question}],
            "text": question,
            "label": answer,
            "metadata": {"problem": example["problem"]},
        }
        return data

    datums: List[Dict[str, Any]] = []
    for i, row in enumerate(data.iter_rows(named=True)):
        datums.append(process_fn(row))

    df = pl.from_dicts(datums)

    return df


data = load_and_merge_dataset()
data = format_dataset(data)

data.write_ndjson("/root/datasets/maths/maths.jsonl")
print(data)
