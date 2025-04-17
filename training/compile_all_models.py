import torch
import torch_tensorrt
import argparse
from pathlib import Path
from enlighten.counter import Counter as ECounter

from project_root import PROJECT_ROOT
from scripts.model_serialization import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compiles models with torchscript.")
    parser.add_argument("--force", "-f", nargs="+", default=[], type=str)
    parser.add_argument("--force_all", "-fa", action="store_true")
    parser.add_argument("--only")
    return parser.parse_args()


def compile_model(weights_path: Path, output_path: Path) -> None:
    print(f"Compiling {weights_path}")
    model = load_model(weights_path)
    device = torch.device("cuda")
    model = model.to(device)

    image_size = model.input_image_size
    sample_inputs = torch.rand(
        (1, 3, image_size["height"], image_size["width"]),
        dtype=torch.float32,
        device=device,
    )

    if not model.supports_jit:
        print(f"Tracing module with image input: [{sample_inputs.shape}]")
        traced_module = torch.jit.trace(model, [sample_inputs], strict=False)
    else:
        traced_module = torch.jit.script(model)

    # Compile with tensorrt
    # None of this works, keep it at 0 to avoid tensorrt
    method = 0
    if method == 0:
        trt_model = traced_module
    else:
        print(f"Compiling with tensorrt...")

        trt_input = torch_tensorrt.Input(
            [1, 3, image_size["height"], image_size["width"]],
            dtype=torch.float,
        )
        if method == 1:
            exp_model = torch.export.export(model, tuple([sample_inputs]))
            trt_model = torch_tensorrt.dynamo.compile(exp_model, [sample_inputs])
        elif method == 2:
            trt_model = torch_tensorrt.ts.compile(
                traced_module,
                inputs=[trt_input],
                truncate_long_and_double=True,
            )
        elif method == 3:
            spec = {
                "forward": torch_tensorrt.ts.TensorRTCompileSpec(
                    **{
                        "inputs": [trt_input],
                        "enabled_precisions": {torch.int, torch.float, torch.half},
                        "truncate_long_and_double": True,
                        "refit": False,
                        "debug": False,
                        "device": {
                            "device_type": torch_tensorrt.DeviceType.GPU,
                            "gpu_id": 0,
                            "dla_core": 0,
                            "allow_gpu_fallback": True,
                        },
                        "capability": torch_tensorrt.EngineCapability.STANDARD,
                        "num_avg_timing_iters": 1,
                    }
                )
            }
            trt_model = torch._C._jit_to_backend("tensorrt", traced_module, spec)

    # Save
    torch.jit.save(trt_model, output_path)
    print(f"Saved to {output_path}")

    if hasattr(model, "vit"):
        embeddings_output = str(output_path).replace(".ptc", "_embeddings.ptc")
        traced_embeddings = torch.jit.trace(
            model.vit.embeddings, [sample_inputs], strict=False
        )
        traced_embeddings.save(embeddings_output)
        print(f"Saved embeddings sub-module to {embeddings_output}")


def main() -> None:
    args = parse_args()

    with torch.inference_mode():
        weights_paths = list(PROJECT_ROOT.glob("models/**/*.pth")) + list(
            PROJECT_ROOT.glob("models/**/config.json")
        )
        pbar = ECounter(total=len(weights_paths))
        for weights_path in pbar(weights_paths):
            if args.only != None and args.only != str(weights_path):
                continue

            # Ignore other files that we know are not models
            if weights_path.name == "rng_state.pth":
                continue

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
