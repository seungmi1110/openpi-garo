"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import os
import numpy as np
import tqdm
import tyro
import torch
import pandas as pd

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_direct_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """Load dataset directly from local parquet files (bypass LeRobotDataset and transforms)."""
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    
    hf_home = os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
    dataset_path = os.path.join(hf_home, data_config.repo_id)
    data_dir = os.path.join(dataset_path, "data")
    
    print(f"Loading dataset from: {data_dir}")
    
    # 모든 parquet 파일 찾기 (재귀적으로)
    parquet_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in sorted(files):
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    # Parquet 파일들에서 필요한 컬럼만 읽기
    dataframes = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf, columns=["state", "actions"])
            dataframes.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {pf}: {e}")
            continue
    
    if len(dataframes) == 0:
        raise ValueError("No valid parquet files loaded")
    
    hf_dataset = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded {len(hf_dataset)} total samples")
    print(f"Columns: {list(hf_dataset.columns)}")
    
    # Transforms 없이 직접 dict로 변환 (state와 action만)
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.df = df
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            return {
                "state": np.array(row["state"], dtype=np.float32),
                "actions": np.array(row["actions"], dtype=np.float32),
            }
    
    dataset = SimpleDataset(hf_dataset)
    
    if max_frames is not None:
        num_samples = min(max_frames, len(dataset))
        num_batches = max(1, num_samples // batch_size)
    else:
        num_samples = len(dataset)
        num_batches = num_samples // batch_size
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: {k: np.array([item[k] for item in x]) for k in x[0].keys()},
    )
    
    return data_loader, num_batches


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 직접 로더 먼저 시도
    try:
        print("Attempting to load dataset directly (bypassing LeRobotDataset)...")
        data_loader, num_batches = create_direct_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, max_frames
        )
    except Exception as e:
        print(f"Direct loader failed: {e}\nFalling back to LeRobot loader...")
        if data_config.rlds_data_dir is not None:
            data_loader, num_batches = create_rlds_dataloader(
                data_config, config.model.action_horizon, config.batch_size, max_frames
            )
        else:
            data_loader, num_batches = create_torch_dataloader(
                data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
            )

    # 원본 데이터 키 이름 사용 (GaroDataConfig repack 전)
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            if key in batch:
                stats[key].update(np.asarray(batch[key]))
            else:
                print(f"Warning: key '{key}' not found in batch. Available keys: {list(batch.keys())}")

    norm_stats = {key: stats[key].get_statistics() for key in keys}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
