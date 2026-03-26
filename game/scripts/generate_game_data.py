#!/usr/bin/env python3
"""
Generate static game data for the Elephant Game.

Reads elephant crop images from an ImageFolder-style directory and generates:
  - public/crops/  (sampled images per elephant)
  - public/game-data/match-pairs.json
  - public/game-data/name-questions.json
  - public/game-data/behavior-questions.json (if behavior data available)

Usage:
  python generate_game_data.py --data-path /media/mu/zoo_vision/data/full_data/train
  python generate_game_data.py --data-path /path/to/train --samples-per-elephant 80
"""

import argparse
import json
import random
import shutil
from pathlib import Path

ELEPHANT_MAP = {
    "01_Chandra": {"id": 1, "name": "Chandra"},
    "02_Indi": {"id": 2, "name": "Indi"},
    "03_Fahra": {"id": 3, "name": "Fahra"},
    "04_Panang": {"id": 4, "name": "Panang"},
    "05_Thai": {"id": 5, "name": "Thai"},
}

GAME_DIR = Path(__file__).resolve().parent.parent
PUBLIC_CROPS = GAME_DIR / "public" / "crops"
PUBLIC_DATA = GAME_DIR / "public" / "game-data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_samples(data_path: Path, samples_per_elephant: int) -> dict[str, list[str]]:
    """Copy sampled images to public/crops/ and return {elephant_name: [relative_paths]}."""
    PUBLIC_CROPS.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[str]] = {}

    for folder_name, info in ELEPHANT_MAP.items():
        src_folder = data_path / folder_name
        if not src_folder.is_dir():
            print(f"  Warning: {src_folder} not found, skipping {info['name']}")
            continue

        images = find_images(src_folder)
        if not images:
            print(f"  Warning: No images in {src_folder}")
            continue

        sampled = random.sample(images, min(samples_per_elephant, len(images)))
        dst_folder = PUBLIC_CROPS / info["name"]
        dst_folder.mkdir(parents=True, exist_ok=True)

        paths = []
        for img in sampled:
            dst = dst_folder / img.name
            shutil.copy2(img, dst)
            # Relative path from public/
            paths.append(f"/crops/{info['name']}/{img.name}")

        result[info["name"]] = paths
        print(f"  {info['name']}: {len(paths)} images copied")

    return result


def generate_match_pairs(
    elephant_images: dict[str, list[str]], num_pairs: int = 200
) -> list[dict]:
    """Generate match pairs (50% same, 50% different)."""
    pairs = []
    names = [n for n in elephant_images if len(elephant_images[n]) >= 2]

    if len(names) < 2:
        print("  Warning: Need at least 2 elephants with 2+ images for match pairs")
        return pairs

    half = num_pairs // 2

    # Same elephant pairs
    for _ in range(half):
        name = random.choice(names)
        imgs = elephant_images[name]
        a, b = random.sample(imgs, 2)
        info = next(v for v in ELEPHANT_MAP.values() if v["name"] == name)
        pairs.append({
            "left": {"image": a, "elephant_id": info["id"], "elephant_name": name},
            "right": {"image": b, "elephant_id": info["id"], "elephant_name": name},
            "is_same_elephant": True,
        })

    # Different elephant pairs
    for _ in range(half):
        n1, n2 = random.sample(names, 2)
        a = random.choice(elephant_images[n1])
        b = random.choice(elephant_images[n2])
        info1 = next(v for v in ELEPHANT_MAP.values() if v["name"] == n1)
        info2 = next(v for v in ELEPHANT_MAP.values() if v["name"] == n2)
        pairs.append({
            "left": {"image": a, "elephant_id": info1["id"], "elephant_name": n1},
            "right": {"image": b, "elephant_id": info2["id"], "elephant_name": n2},
            "is_same_elephant": False,
        })

    random.shuffle(pairs)
    return pairs


def generate_name_questions(
    elephant_images: dict[str, list[str]], num_questions: int = 100
) -> list[dict]:
    """Generate name identification questions."""
    questions = []
    names = list(elephant_images.keys())

    for _ in range(num_questions):
        name = random.choice(names)
        img = random.choice(elephant_images[name])
        info = next(v for v in ELEPHANT_MAP.values() if v["name"] == name)
        questions.append({
            "image": img,
            "elephant_id": info["id"],
            "elephant_name": name,
        })

    random.shuffle(questions)
    return questions


def generate_behavior_questions(
    elephant_images: dict[str, list[str]], num_questions: int = 100
) -> list[dict]:
    """Generate behavior questions.

    Since we don't have behavior-labeled crops in the identity dataset,
    this generates placeholder questions with 'Standing' as the default.
    In production, this should read from a behavior-labeled dataset.
    """
    questions = []
    names = list(elephant_images.keys())
    behaviors = ["Standing", "SleepL", "SleepR"]

    for _ in range(num_questions):
        name = random.choice(names)
        img = random.choice(elephant_images[name])
        # Default to Standing since we don't have behavior labels
        behavior = random.choice(behaviors)
        questions.append({
            "image": img,
            "behavior": behavior,
            "elephant_name": name,
        })

    random.shuffle(questions)
    return questions


def main():
    parser = argparse.ArgumentParser(description="Generate elephant game data")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to ImageFolder train directory (e.g., /media/mu/zoo_vision/data/full_data/train)",
    )
    parser.add_argument(
        "--samples-per-elephant",
        type=int,
        default=50,
        help="Number of images to sample per elephant (default: 50)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=200,
        help="Number of match pairs to generate (default: 200)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of name/behavior questions (default: 100)",
    )
    args = parser.parse_args()

    if not args.data_path.is_dir():
        print(f"Error: {args.data_path} is not a valid directory")
        return

    print(f"Data path: {args.data_path}")
    print(f"Output: {PUBLIC_CROPS} / {PUBLIC_DATA}")
    print()

    # Step 1: Copy samples
    print("Copying elephant crop images...")
    elephant_images = copy_samples(args.data_path, args.samples_per_elephant)
    total_images = sum(len(v) for v in elephant_images.values())
    print(f"Total: {total_images} images for {len(elephant_images)} elephants\n")

    if not elephant_images:
        print("Error: No images found. Check your data path.")
        return

    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)

    # Step 2: Generate match pairs
    print("Generating match pairs...")
    pairs = generate_match_pairs(elephant_images, args.num_pairs)
    match_path = PUBLIC_DATA / "match-pairs.json"
    with open(match_path, "w") as f:
        json.dump({"pairs": pairs}, f, indent=2)
    print(f"  {len(pairs)} pairs → {match_path}\n")

    # Step 3: Generate name questions
    print("Generating name questions...")
    name_qs = generate_name_questions(elephant_images, args.num_questions)
    name_path = PUBLIC_DATA / "name-questions.json"
    with open(name_path, "w") as f:
        json.dump({"questions": name_qs}, f, indent=2)
    print(f"  {len(name_qs)} questions → {name_path}\n")

    # Step 4: Generate behavior questions
    print("Generating behavior questions...")
    behavior_qs = generate_behavior_questions(elephant_images, args.num_questions)
    behavior_path = PUBLIC_DATA / "behavior-questions.json"
    with open(behavior_path, "w") as f:
        json.dump({"questions": behavior_qs}, f, indent=2)
    print(f"  {len(behavior_qs)} questions → {behavior_path}\n")

    print("Done! Run `npm run dev` to start the game.")


if __name__ == "__main__":
    main()
