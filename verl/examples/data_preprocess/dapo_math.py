# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the DAPO-Math-17k dataset to parquet format.

Source: ``open-r1/DAPO-Math-17k-Processed`` — already in verl schema
(``prompt`` / ``reward_model`` / ``extra_info``), so we mostly re-wrap it:
append the boxed-answer instruction to the question and carve out a small
held-out test split. Output layout (train.parquet / test.parquet) matches
the other scripts in this directory.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/dapo_math", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--test_size", type=float, default=0.01, help="Fraction of examples held out for the test split."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # The HuggingFace repo we load from...
    hf_repo = "open-r1/DAPO-Math-17k-Processed"
    # ...versus the data_source tag written into each row, which is what verl's
    # reward manager (default_compute_score) keys on to select the verifier.
    # "math_dapo" routes to verl/utils/reward_score/math_dapo.py — the DAPO
    # boxed-answer exact-match scorer (+1 correct / -1 incorrect). Using the raw
    # HF repo id here instead would raise NotImplementedError at reward time.
    data_source = "math_dapo"
    print(f"Loading the {hf_repo} dataset from huggingface...", flush=True)
    # DAPO-Math is a plain parquet dataset (no loading script), so no
    # trust_remote_code is needed — and datasets>=4 rejects/ignores it anyway.
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "en")
    else:
        dataset = datasets.load_dataset(hf_repo, "en")

    # Only a "train" split is published; carve out a small test split from it.
    dataset = dataset["train"]
    split_datasets = dataset.train_test_split(test_size=args.test_size, shuffle=True, seed=42)
    train_dataset = split_datasets["train"]
    test_dataset = split_datasets["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")
            question = question + " " + instruction_following

            # The source dataset is already in verl format; pass these through.
            reward_model = example.pop("reward_model")
            extra_info = example.pop("extra_info", {}) or {}
            extra_info.setdefault("split", split)
            extra_info.setdefault("index", idx)

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": reward_model,
                "extra_info": extra_info,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    hdfs_dir = args.hdfs_dir

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example from each split as JSON for quick inspection.
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(test_dataset[0], f, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
