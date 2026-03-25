from __future__ import annotations

import argparse
import glob
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_SERVICE_NAME = "empty-video-hourly-scan"
DEFAULT_FOLDER_TEMPLATE = "ELP-Kamera-01/%Y%m%d%p"
DEFAULT_HOST_VIDEO_ROOT = "/mnt/camera_nas"
DEFAULT_DIRECT_FOLDER = "/mnt/camera_nas/ZAG-ELP-CAM-01*"


@dataclass(frozen=True)
class HourlyScanSettings:
    host_video_root: str
    host_model_root: str
    service_name: str
    target_mode: str
    folder_template: str
    direct_folder: str
    recursive: bool
    interval_minutes: int
    confidence_threshold: float
    weights_path: str
    force_rescan: bool


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path(project_root: Path) -> Path:
    return project_root / ".hourly-scan.env"


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_file(config_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            raise ValueError(f"Invalid config line in {config_path}: {line}")
        values[key.strip()] = _strip_optional_quotes(value.strip())
    return values


def _bool_value(raw_value: str | None, *, name: str) -> bool:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw_value}")


def create_default_config(config_path: Path, *, project_root: Path) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    model_root = project_root / "models"
    config_path.write_text(
        "\n".join(
            [
                "# Docker mounts",
                f"HOST_VIDEO_ROOT={DEFAULT_HOST_VIDEO_ROOT}",
                f"HOST_MODEL_ROOT={model_root}",
                "",
                "# systemd naming",
                f"AUTO_SCAN_SERVICE_NAME={DEFAULT_SERVICE_NAME}",
                "",
                "# Folder selection",
                "# direct mode default scans all matching current camera folders",
                "AUTO_SCAN_TARGET_MODE=direct",
                f"AUTO_SCAN_FOLDER_TEMPLATE={DEFAULT_FOLDER_TEMPLATE}",
                f"AUTO_SCAN_DIRECT_FOLDER={DEFAULT_DIRECT_FOLDER}",
                "",
                "# Scan settings",
                "AUTO_SCAN_RECURSIVE=false",
                "AUTO_SCAN_INTERVAL_MINUTES=2",
                "AUTO_SCAN_CONFIDENCE_THRESHOLD=0.65",
                "AUTO_SCAN_WEIGHTS_PATH=",
                "AUTO_SCAN_FORCE_RESCAN=true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def load_settings(config_path: Path, *, project_root: Path) -> HourlyScanSettings:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_values = _parse_env_file(config_path)
    host_video_root = raw_values.get("HOST_VIDEO_ROOT", DEFAULT_HOST_VIDEO_ROOT).strip() or DEFAULT_HOST_VIDEO_ROOT
    host_model_root = raw_values.get("HOST_MODEL_ROOT", str(project_root / "models")).strip() or str(project_root / "models")
    service_name = raw_values.get("AUTO_SCAN_SERVICE_NAME", DEFAULT_SERVICE_NAME).strip() or DEFAULT_SERVICE_NAME
    target_mode = raw_values.get("AUTO_SCAN_TARGET_MODE", "direct").strip() or "direct"
    folder_template = raw_values.get("AUTO_SCAN_FOLDER_TEMPLATE", DEFAULT_FOLDER_TEMPLATE).strip() or DEFAULT_FOLDER_TEMPLATE
    direct_folder = raw_values.get("AUTO_SCAN_DIRECT_FOLDER", DEFAULT_DIRECT_FOLDER).strip() or DEFAULT_DIRECT_FOLDER

    try:
        interval_minutes = int(raw_values.get("AUTO_SCAN_INTERVAL_MINUTES", "2").strip() or "2")
    except ValueError as exc:
        raise ValueError("AUTO_SCAN_INTERVAL_MINUTES must be an integer") from exc

    try:
        confidence_threshold = float(raw_values.get("AUTO_SCAN_CONFIDENCE_THRESHOLD", "0.65").strip() or "0.65")
    except ValueError as exc:
        raise ValueError("AUTO_SCAN_CONFIDENCE_THRESHOLD must be a float") from exc

    if interval_minutes < 1:
        raise ValueError("AUTO_SCAN_INTERVAL_MINUTES must be at least 1")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("AUTO_SCAN_CONFIDENCE_THRESHOLD must be between 0 and 1")
    if target_mode not in {"template", "direct"}:
        raise ValueError(f"Unsupported AUTO_SCAN_TARGET_MODE: {target_mode}")

    return HourlyScanSettings(
        host_video_root=host_video_root,
        host_model_root=host_model_root,
        service_name=service_name,
        target_mode=target_mode,
        folder_template=folder_template,
        direct_folder=direct_folder,
        recursive=_bool_value(raw_values.get("AUTO_SCAN_RECURSIVE", "false"), name="AUTO_SCAN_RECURSIVE"),
        interval_minutes=interval_minutes,
        confidence_threshold=confidence_threshold,
        weights_path=raw_values.get("AUTO_SCAN_WEIGHTS_PATH", "").strip(),
        force_rescan=_bool_value(raw_values.get("AUTO_SCAN_FORCE_RESCAN", "true"), name="AUTO_SCAN_FORCE_RESCAN"),
    )


def resolve_target_folder(settings: HourlyScanSettings, *, now: datetime | None = None) -> str:
    if settings.target_mode == "template":
        current_time = now or datetime.now()
        return current_time.strftime(settings.folder_template)
    if settings.target_mode == "direct":
        if not settings.direct_folder:
            raise ValueError("AUTO_SCAN_DIRECT_FOLDER is required in direct mode")
        return settings.direct_folder
    raise ValueError(f"Unsupported AUTO_SCAN_TARGET_MODE: {settings.target_mode}")


def normalize_folder_for_cli(target_folder: str, *, host_video_root: str) -> str:
    normalized_root = host_video_root.rstrip("/")
    if target_folder == normalized_root:
        return "."
    root_prefix = f"{normalized_root}/"
    if target_folder.startswith(root_prefix):
        return target_folder[len(root_prefix):]
    return target_folder


def host_target_path(cli_folder: str, *, host_video_root: str) -> str:
    normalized_root = host_video_root.rstrip("/")
    if cli_folder.startswith("/"):
        return cli_folder
    if cli_folder in {"", ".", "/"}:
        return normalized_root
    return f"{normalized_root}/{cli_folder}"


def resolve_scan_targets(
    settings: HourlyScanSettings,
    *,
    now: datetime | None = None,
) -> list[tuple[str, str]]:
    raw_target_folder = resolve_target_folder(settings, now=now)
    absolute_pattern = host_target_path(raw_target_folder, host_video_root=settings.host_video_root)

    if glob.has_magic(absolute_pattern):
        matched_host_folders = [
            str(Path(match).resolve())
            for match in sorted(glob.glob(absolute_pattern))
            if Path(match).is_dir()
        ]
    else:
        matched_host_folders = [str(Path(absolute_pattern).resolve())]

    targets: list[tuple[str, str]] = []
    for host_folder in matched_host_folders:
        cli_folder = normalize_folder_for_cli(host_folder, host_video_root=settings.host_video_root)
        targets.append((cli_folder, host_folder))
    return targets


def build_compose_env(settings: HourlyScanSettings) -> dict[str, str]:
    env = os.environ.copy()
    env["HOST_VIDEO_ROOT"] = settings.host_video_root
    env["HOST_MODEL_ROOT"] = settings.host_model_root
    return env


def build_scan_command(settings: HourlyScanSettings, *, cli_folder: str) -> list[str]:
    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-T",
        "app",
        "empty-video-cli",
        "--folder",
        cli_folder,
        "--interval-minutes",
        str(settings.interval_minutes),
        "--confidence-threshold",
        str(settings.confidence_threshold),
    ]
    command.append("--recursive" if settings.recursive else "--no-recursive")
    if settings.weights_path:
        command.extend(["--weights-path", settings.weights_path])
    if settings.force_rescan:
        command.append("--rescan")
    return command


def _require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found: {name}")


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def _systemctl_command(*, use_user_units: bool) -> list[str]:
    return ["systemctl", "--user"] if use_user_units else ["systemctl"]


def build_service_unit_text(
    *,
    project_root: Path,
    config_path: Path,
) -> str:
    exec_command = shlex.join(
        [
            "/usr/bin/env",
            "python3",
            "-m",
            "empty_video_tool.hourly_scan",
            "--project-root",
            str(project_root),
            "run",
            "--config",
            str(config_path),
        ]
    )
    return "\n".join(
        [
            "[Unit]",
            "Description=Empty video hourly scan",
            "",
            "[Service]",
            "Type=oneshot",
            f"WorkingDirectory={project_root}",
            "Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            f"Environment=PYTHONPATH={project_root / 'src'}",
            f"ExecStart={exec_command}",
            "",
        ]
    )


def build_timer_unit_text(service_name: str) -> str:
    return "\n".join(
        [
            "[Unit]",
            "Description=Run empty video scan every hour",
            "",
            "[Timer]",
            "OnCalendar=hourly",
            "Persistent=true",
            f"Unit={service_name}.service",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )


def _unit_directory(*, use_user_units: bool) -> Path:
    if use_user_units:
        return Path.home() / ".config/systemd/user"
    return Path("/etc/systemd/system")


def _resolve_config_path(raw_config_path: str | None, *, project_root: Path) -> Path:
    if raw_config_path:
        return Path(raw_config_path).expanduser().resolve()
    return default_config_path(project_root).resolve()


def run_scan(
    *,
    project_root: Path,
    config_path: Path,
    dry_run: bool,
) -> int:
    settings = load_settings(config_path, project_root=project_root)
    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    (project_root / "models").mkdir(parents=True, exist_ok=True)

    targets = resolve_scan_targets(settings)
    if not targets:
        print("No matching target folders were found. Skipping this hourly run.")
        return 0

    for cli_folder, resolved_host_folder in targets:
        print(f"Resolved host folder: {resolved_host_folder}")
        print(f"CLI folder argument: {cli_folder}")

    if dry_run:
        return 0

    _require_command("docker")
    compose_env = build_compose_env(settings)
    for cli_folder, resolved_host_folder in targets:
        if not Path(resolved_host_folder).is_dir():
            print(f"Target folder does not exist yet. Skipping: {resolved_host_folder}")
            continue
        _run_command(
            build_scan_command(settings, cli_folder=cli_folder),
            cwd=project_root,
            env=compose_env,
        )
    return 0


def install_units(
    *,
    project_root: Path,
    config_path: Path,
    use_user_units: bool,
) -> int:
    _require_command("docker")
    _require_command("systemctl")

    if not config_path.exists():
        create_default_config(config_path, project_root=project_root)
        print(f"Created config file: {config_path}")
        print("Edit it before enabling the timer if you need a different target folder.")

    settings = load_settings(config_path, project_root=project_root)

    if not use_user_units and os.geteuid() != 0:
        raise PermissionError("System-wide install needs root. Use sudo or pass --user.")

    unit_dir = _unit_directory(use_user_units=use_user_units)
    unit_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "runs").mkdir(parents=True, exist_ok=True)
    (project_root / "models").mkdir(parents=True, exist_ok=True)

    service_path = unit_dir / f"{settings.service_name}.service"
    timer_path = unit_dir / f"{settings.service_name}.timer"

    service_path.write_text(
        build_service_unit_text(project_root=project_root, config_path=config_path),
        encoding="utf-8",
    )
    timer_path.write_text(build_timer_unit_text(settings.service_name), encoding="utf-8")

    compose_env = build_compose_env(settings)
    _run_command(["docker", "compose", "build", "app"], cwd=project_root, env=compose_env)

    systemctl_command = _systemctl_command(use_user_units=use_user_units)
    _run_command([*systemctl_command, "daemon-reload"], cwd=project_root)
    _run_command([*systemctl_command, "enable", "--now", f"{settings.service_name}.timer"], cwd=project_root)

    print(f"Installed timer: {settings.service_name}.timer")
    print(f"Config file: {config_path}")
    print(f"Service file: {service_path}")
    print(f"Timer file: {timer_path}")
    return 0


def status_units(
    *,
    project_root: Path,
    config_path: Path,
    use_user_units: bool,
) -> int:
    _require_command("systemctl")
    if config_path.exists():
        service_name = load_settings(config_path, project_root=project_root).service_name
    else:
        service_name = DEFAULT_SERVICE_NAME

    systemctl_command = _systemctl_command(use_user_units=use_user_units)
    subprocess.run([*systemctl_command, "status", f"{service_name}.service", "--no-pager"], check=False)
    subprocess.run([*systemctl_command, "status", f"{service_name}.timer", "--no-pager"], check=False)
    return 0


def uninstall_units(
    *,
    project_root: Path,
    config_path: Path,
    use_user_units: bool,
) -> int:
    _require_command("systemctl")
    if not use_user_units and os.geteuid() != 0:
        raise PermissionError("System-wide uninstall needs root. Use sudo or pass --user.")

    if config_path.exists():
        service_name = load_settings(config_path, project_root=project_root).service_name
    else:
        service_name = DEFAULT_SERVICE_NAME

    unit_dir = _unit_directory(use_user_units=use_user_units)
    systemctl_command = _systemctl_command(use_user_units=use_user_units)

    subprocess.run([*systemctl_command, "disable", "--now", f"{service_name}.timer"], check=False)
    (unit_dir / f"{service_name}.timer").unlink(missing_ok=True)
    (unit_dir / f"{service_name}.service").unlink(missing_ok=True)
    _run_command([*systemctl_command, "daemon-reload"], cwd=project_root)

    print(f"Removed {service_name}.service and {service_name}.timer")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="empty-video-hourly",
        description="Install or run the hourly empty-video scan automation.",
    )
    parser.add_argument(
        "--project-root",
        default=str(default_project_root()),
        help="Project root that contains docker-compose.yml, runs/, models/, and src/.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install", help="Install the systemd service and timer.")
    install_parser.add_argument("--config", help="Path to the hourly scan .env file.")
    install_parser.add_argument("--user", action="store_true", help="Install as a user systemd unit.")

    run_parser = subparsers.add_parser("run", help="Resolve the current folder and run one scan.")
    run_parser.add_argument("--config", help="Path to the hourly scan .env file.")
    run_parser.add_argument("--dry-run", action="store_true", help="Only print the resolved folder and CLI argument.")

    status_parser = subparsers.add_parser("status", help="Show service and timer status.")
    status_parser.add_argument("--config", help="Path to the hourly scan .env file.")
    status_parser.add_argument("--user", action="store_true", help="Use user systemd units.")

    uninstall_parser = subparsers.add_parser("uninstall", help="Remove the systemd service and timer.")
    uninstall_parser.add_argument("--config", help="Path to the hourly scan .env file.")
    uninstall_parser.add_argument("--user", action="store_true", help="Remove user systemd units.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    config_path = _resolve_config_path(getattr(args, "config", None), project_root=project_root)

    try:
        if args.command == "install":
            return install_units(
                project_root=project_root,
                config_path=config_path,
                use_user_units=bool(args.user),
            )
        if args.command == "run":
            return run_scan(
                project_root=project_root,
                config_path=config_path,
                dry_run=bool(args.dry_run),
            )
        if args.command == "status":
            return status_units(
                project_root=project_root,
                config_path=config_path,
                use_user_units=bool(args.user),
            )
        if args.command == "uninstall":
            return uninstall_units(
                project_root=project_root,
                config_path=config_path,
                use_user_units=bool(args.user),
            )
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
