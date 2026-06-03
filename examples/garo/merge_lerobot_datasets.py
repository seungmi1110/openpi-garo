"""전처리된 LeRobot 데이터셋 여러 개를 비율에 맞게 빠르게 합쳐 새 데이터셋을 만드는 스크립트.

핵심 아이디어:
- 느린 방식: 프레임 디코딩 -> add_frame -> 이미지 재인코딩
- 빠른 방식(이 스크립트): 에피소드 parquet를 직접 읽어 인덱스 컬럼만 재매핑 후 재저장

즉, 이미지 픽셀 디코딩/재인코딩 없이 `parquet + meta`를 일관되게 병합한다.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_NAME = "garo_merged"


def parse_source(source: str) -> tuple[Path, float]:
    """'경로:비율' 형식 파싱. 예) /data/.../pi05_v01:0.7"""
    if ":" not in source:
        raise ValueError(f"Invalid source '{source}'. Expected '<path>:<ratio>'.")

    raw_path, raw_ratio = source.rsplit(":", 1)
    path = Path(raw_path.strip())
    ratio = float(raw_ratio)

    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"Ratio must be in (0, 1]. Got {ratio} for source '{source}'.")
    return path, ratio


def load_info(src_root: Path) -> dict[str, Any]:
    with open(src_root / "meta" / "info.json", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_episodes(num_episodes: int, ratio: float, rng: np.random.Generator) -> list[int]:
    n = max(1, round(num_episodes * ratio))
    chosen = rng.choice(num_episodes, size=n, replace=False)
    return sorted(int(x) for x in chosen)


def resolve_parquet_path(src_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    candidate = src_root / template.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    if candidate.exists():
        return candidate

    # fallback: locate by filename under data/chunk-*/
    filename = f"episode_{episode_index:06d}.parquet"
    matches = list((src_root / "data").glob(f"chunk-*/{filename}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not find parquet for episode {episode_index} in {src_root}")


def normalize_list_feature_type(obj: Any) -> Any:
    """HF parquet metadata의 _type=List를 _type=Sequence로 정규화.

    openpi 환경의 datasets==3.6.0에서 List 타입 메타데이터를 읽을 때
    ValueError('Feature type 'List' not found')가 날 수 있어 병합 시 정규화한다.
    """
    if isinstance(obj, dict):
        out = {k: normalize_list_feature_type(v) for k, v in obj.items()}
        if out.get("_type") == "List":
            out["_type"] = "Sequence"
        return out
    if isinstance(obj, list):
        return [normalize_list_feature_type(v) for v in obj]
    return obj


def normalize_parquet_hf_metadata(table: pa.Table) -> pa.Table:
    metadata = table.schema.metadata
    if not metadata or b"huggingface" not in metadata:
        return table

    payload = json.loads(metadata[b"huggingface"].decode("utf-8"))
    payload = normalize_list_feature_type(payload)

    new_metadata = dict(metadata)
    new_metadata[b"huggingface"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return table.replace_schema_metadata(new_metadata)


def set_int_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name not in table.column_names:
        return table

    col_idx = table.column_names.index(name)
    field = table.schema.field(name)
    arr = pa.array(values, type=field.type)
    return table.set_column(col_idx, field, arr)


def cast_like(value: float | int, reference: Any) -> float | int:
    if isinstance(reference, bool):
        return bool(value)
    if isinstance(reference, int):
        return int(value)
    return float(value)


def remap_episode_stats(
    src_record: dict[str, Any],
    *,
    new_episode_index: int,
    new_task_index: int,
    new_global_start_index: int,
    episode_length: int,
) -> dict[str, Any]:
    out = copy.deepcopy(src_record)
    out["episode_index"] = new_episode_index

    stats = out.get("stats", {})

    if "episode_index" in stats:
        ref = stats["episode_index"]
        ref_min = ref.get("min", [0])[0]
        ref_max = ref.get("max", [0])[0]
        ref_mean = ref.get("mean", [0.0])[0]
        ref["min"] = [cast_like(new_episode_index, ref_min)]
        ref["max"] = [cast_like(new_episode_index, ref_max)]
        ref["mean"] = [cast_like(float(new_episode_index), ref_mean)]
        ref["std"] = [cast_like(0.0, ref.get("std", [0.0])[0])]
        ref["count"] = [cast_like(episode_length, ref.get("count", [episode_length])[0])]

    if "task_index" in stats:
        ref = stats["task_index"]
        ref_min = ref.get("min", [0])[0]
        ref_max = ref.get("max", [0])[0]
        ref_mean = ref.get("mean", [0.0])[0]
        ref["min"] = [cast_like(new_task_index, ref_min)]
        ref["max"] = [cast_like(new_task_index, ref_max)]
        ref["mean"] = [cast_like(float(new_task_index), ref_mean)]
        ref["std"] = [cast_like(0.0, ref.get("std", [0.0])[0])]
        ref["count"] = [cast_like(episode_length, ref.get("count", [episode_length])[0])]

    if "index" in stats and episode_length > 0:
        ref = stats["index"]
        new_min = new_global_start_index
        new_max = new_global_start_index + episode_length - 1
        new_mean = (new_min + new_max) / 2.0
        ref["min"] = [cast_like(new_min, ref.get("min", [new_min])[0])]
        ref["max"] = [cast_like(new_max, ref.get("max", [new_max])[0])]
        ref["mean"] = [cast_like(new_mean, ref.get("mean", [new_mean])[0])]
        # std는 상수 이동에 불변이므로 유지
        ref["count"] = [cast_like(episode_length, ref.get("count", [episode_length])[0])]

    return out


def ensure_compatible(base_info: dict[str, Any], other_info: dict[str, Any], other_root: Path) -> None:
    for key in ("robot_type", "fps"):
        if base_info.get(key) != other_info.get(key):
            raise ValueError(
                f"Incompatible '{key}' between datasets: "
                f"base={base_info.get(key)} vs {other_root.name}={other_info.get(key)}"
            )

    # 최소한 feature 스키마는 같아야 안전하게 병합 가능
    if base_info.get("features") != other_info.get("features"):
        raise ValueError(
            f"Feature schema mismatch: base vs {other_root.name}. "
            "Use datasets with identical LeRobot features."
        )


def main(
    output_dir: str,
    sources: list[str],
    *,
    seed: int = 42,
    push_to_hub: bool = False,
):
    """
    Args:
        output_dir: 병합된 데이터셋 저장 경로
        sources: "경로:비율" 목록
        seed: 에피소드 샘플링 시드
        push_to_hub: True이면 병합 후 Hub 업로드 시도
    """
    if not sources:
        raise ValueError("Please provide at least one source dataset.")

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    parsed = [parse_source(s) for s in sources]
    rng = np.random.default_rng(seed)

    first_root, _ = parsed[0]
    first_info = load_info(first_root)

    for src_root, _ in parsed[1:]:
        ensure_compatible(first_info, load_info(src_root), src_root)

    (output_path / "meta").mkdir(parents=True, exist_ok=True)
    (output_path / "data").mkdir(parents=True, exist_ok=True)

    chunks_size = int(first_info.get("chunks_size", 1000))
    data_template = first_info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    task_to_index: dict[str, int] = {}
    merged_episodes: list[dict[str, Any]] = []
    merged_episode_stats: list[dict[str, Any]] = []

    total_frames = 0
    total_episodes = 0

    for src_root, ratio in parsed:
        info = load_info(src_root)

        episodes = load_jsonl(src_root / "meta" / "episodes.jsonl")
        episodes_by_idx = {int(e["episode_index"]): e for e in episodes}

        stats = load_jsonl(src_root / "meta" / "episodes_stats.jsonl")
        stats_by_idx = {int(s["episode_index"]): s for s in stats}

        num_episodes = len(episodes_by_idx)
        selected = select_episodes(num_episodes, ratio, rng)

        print(
            f"\n[{src_root.name}] 총 {num_episodes}개 중 {len(selected)}개 에피소드 사용 "
            f"(지정 비율 {ratio * 100:.0f}%)"
        )

        for src_ep_idx in selected:
            episode = episodes_by_idx[src_ep_idx]
            task_name = episode["tasks"][0] if episode.get("tasks") else ""

            if task_name not in task_to_index:
                task_to_index[task_name] = len(task_to_index)
            new_task_index = task_to_index[task_name]

            src_parquet = resolve_parquet_path(src_root, info, src_ep_idx)
            table = pq.read_table(src_parquet)
            table = normalize_parquet_hf_metadata(table)

            episode_len = int(table.num_rows)
            new_ep_idx = total_episodes

            frame_index = np.arange(episode_len, dtype=np.int64)
            global_index = frame_index + total_frames
            task_index = np.full(episode_len, new_task_index, dtype=np.int64)
            episode_index = np.full(episode_len, new_ep_idx, dtype=np.int64)

            table = set_int_column(table, "frame_index", frame_index)
            table = set_int_column(table, "index", global_index)
            table = set_int_column(table, "task_index", task_index)
            table = set_int_column(table, "episode_index", episode_index)

            out_rel = data_template.format(
                episode_chunk=new_ep_idx // chunks_size,
                episode_index=new_ep_idx,
            )
            out_parquet = output_path / out_rel
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, out_parquet)

            merged_episodes.append(
                {
                    "episode_index": new_ep_idx,
                    "tasks": [task_name],
                    "length": episode_len,
                }
            )

            if src_ep_idx in stats_by_idx:
                merged_episode_stats.append(
                    remap_episode_stats(
                        stats_by_idx[src_ep_idx],
                        new_episode_index=new_ep_idx,
                        new_task_index=new_task_index,
                        new_global_start_index=total_frames,
                        episode_length=episode_len,
                    )
                )

            total_frames += episode_len
            total_episodes += 1

            print(
                f"  ep {src_ep_idx:06d} ({episode_len} frames) -> 새 ep {new_ep_idx:06d}"
            )

    # tasks.jsonl
    tasks_rows = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1])
    ]

    write_jsonl(output_path / "meta" / "tasks.jsonl", tasks_rows)
    write_jsonl(output_path / "meta" / "episodes.jsonl", merged_episodes)
    write_jsonl(output_path / "meta" / "episodes_stats.jsonl", merged_episode_stats)

    merged_info = copy.deepcopy(first_info)
    merged_info["total_episodes"] = total_episodes
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(tasks_rows)
    merged_info["total_chunks"] = (total_episodes + chunks_size - 1) // chunks_size if total_episodes > 0 else 0
    merged_info["splits"] = {"train": f"0:{total_episodes}"}

    # 보통 이미지 parquet 기반에서는 0
    if int(first_info.get("total_videos", 0)) == 0:
        merged_info["total_videos"] = 0

    with open(output_path / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(merged_info, f, ensure_ascii=False, indent=4)

    print(f"\n병합 완료: 총 {total_episodes}개 에피소드 / {total_frames} frames -> {output_path}")

    if push_to_hub:
        dataset = LeRobotDataset(repo_id=REPO_NAME, root=output_path)
        dataset.push_to_hub(
            tags=["garo", "pi05", "rx1_dual_arm"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
