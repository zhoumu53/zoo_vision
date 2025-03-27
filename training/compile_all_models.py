import torch
import argparse
from pathlib import Path
from enlighten.counter import Counter as ECounter

from project_root import PROJECT_ROOT
from scripts.model_serialization import load_model

from transformers import Mask2FormerForUniversalSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compiles models with torchscript.")
    parser.add_argument("--force", "-f", nargs="+", default=[], type=str)
    parser.add_argument("--force_all", "-fa", action="store_true")
    return parser.parse_args()


def compile_model(weights_path: Path, output_path: Path) -> None:
    print(f"Compiling {weights_path}")
    model = load_model(weights_path)
    model = model.to(torch.device("cuda"))
    if isinstance(model, Mask2FormerForUniversalSegmentation):
        image_size = model.input_image_size
        sample_inputs = torch.rand(
            (1, 3, image_size["height"], image_size["width"]),
            dtype=torch.float32,
            device=model.device,
        )
        traced_module = torch.jit.trace(model, [sample_inputs], strict=False)
    else:
        traced_module = torch.jit.script(model)
    traced_module.save(output_path)
    print(f"Saved to {output_path}")


def main() -> None:
    args = parse_args()

    with torch.inference_mode():
        weights_paths = list(PROJECT_ROOT.glob("models/**/*.pth")) + list(
            PROJECT_ROOT.glob("models/**/config.json")
        )
        pbar = ECounter(total=len(weights_paths))
        for weights_path in pbar(weights_paths):
            pbar.desc = str(weights_path.relative_to(PROJECT_ROOT))

            output_path = PROJECT_ROOT / weights_path.with_suffix(".ptc")
            if (
                args.force_all
                or str(weights_path) in args.force
                or not output_path.exists()
            ):
                compile_model(weights_path, output_path)
            else:
                print(f"Skipping {str(weights_path)}")


if __name__ == "__main__":
    main()
