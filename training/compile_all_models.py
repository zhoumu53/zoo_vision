import torch
import torchvision
from pathlib import Path
import enlighten

from project_root import PROJECT_ROOT
from scripts.model_serialization import load_model

pbar_manager = enlighten.get_manager()


def compile_model(weights_path: Path, output_path: Path) -> None:
    print(f"Compiling {weights_path}")
    model = load_model(weights_path)

    traced_module = torch.jit.script(model)
    traced_module.save(output_path)


def main() -> None:
    with torch.no_grad():
        weights_paths = list(PROJECT_ROOT.glob("models/**/*.pth"))
        pbar = pbar_manager.counter(total=len(weights_paths))
        for weights_path in pbar(weights_paths):
            pbar.desc = str(weights_path.relative_to(PROJECT_ROOT))
            output_path = PROJECT_ROOT / weights_path.with_suffix(".ptc")
            if not output_path.exists():
                compile_model(weights_path, output_path)


if __name__ == "__main__":
    main()
