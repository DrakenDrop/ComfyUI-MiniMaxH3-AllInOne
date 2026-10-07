"""Bounded, cross-platform discovery of an installed llama-server executable."""
from __future__ import annotations
import os
import shutil
import sys

_LAYOUTS = ("", "bin", "build/bin", "build/bin/Release", "build/bin/Debug",
            "build/Release", "Release")

def _exe_name():
    return "llama-server.exe" if os.name == "nt" else "llama-server"

def _expand(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(path).strip().strip('"'))))

def _usable(path):
    return os.path.isfile(path) and (os.name == "nt" or os.access(path, os.X_OK))

def _in_directory(directory):
    for layout in _LAYOUTS:
        candidate = os.path.join(directory, *layout.split("/"), _exe_name())
        if _usable(candidate):
            return os.path.abspath(candidate)
    return None

def _resolve_override(value, pack_dir):
    value = os.path.expandvars(os.path.expanduser(str(value).strip().strip('"')))
    command = shutil.which(value)
    if command and _usable(command):
        return os.path.abspath(command)
    # Relative overrides are relative to the custom-node package.
    path = value if os.path.isabs(value) else os.path.join(pack_dir, value)
    if _usable(path):
        return os.path.abspath(path)
    return _in_directory(path) if os.path.isdir(path) else None

def _search_roots(pack_dir, cfg):
    home = os.path.expanduser("~")
    roots = list(cfg.get("llama_server_search_dirs") or [])
    roots += [pack_dir, os.path.dirname(pack_dir), os.path.dirname(sys.executable)]
    try:
        import folder_paths
        base = getattr(folder_paths, "base_path", "")
        if base:
            roots += [base, os.path.dirname(base)]
        roots.append(os.path.join(folder_paths.models_dir, "LLM"))
        for key in ("LLM", "llm"):
            try:
                roots += list(folder_paths.get_folder_paths(key))
            except (KeyError, AttributeError):
                pass
    except ImportError:
        pass
    roots += list(cfg.get("extra_model_dirs") or [])
    roots += [home, os.path.join(home, "Downloads"), os.path.join(home, ".local", "bin"),
              os.path.join(home, "Applications"), "/workspace", "/opt",
              "/usr/local/bin", "/usr/bin", "/opt/homebrew/bin"]
    if os.name == "nt":
        roots += ["C:/", "C:/tools"]
        for key in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
            if os.environ.get(key):
                roots.append(os.environ[key])
        if os.environ.get("LOCALAPPDATA"):
            roots.append(os.path.join(os.environ["LOCALAPPDATA"], "Programs"))
    seen = set()
    for root in roots:
        expanded = _expand(root)
        key = os.path.normcase(expanded)
        if key not in seen:
            seen.add(key)
            yield expanded

def discover(cfg, pack_dir):
    """Return an executable without launching it or recursively scanning disks."""
    explicit = str(cfg.get("llama_server_path") or "").strip()
    if explicit and explicit.lower() != "auto":
        result = _resolve_override(explicit, pack_dir)
        if result:
            return result
        raise FileNotFoundError(
            f"Configured llama_server_path is not usable: {explicit}. "
            "Set it to an empty string or 'auto' to enable automatic discovery.")
    for key in ("LLAMA_SERVER_PATH", "LLAMA_CPP_DIR"):
        if os.environ.get(key):
            result = _resolve_override(os.environ[key], pack_dir)
            if result:
                return result
    command = shutil.which(_exe_name())
    if command and _usable(command):
        return os.path.abspath(command)
    for root in _search_roots(pack_dir, cfg):
        result = _in_directory(root)
        if result:
            return result
        result = _in_directory(os.path.join(root, "llama.cpp"))
        if result:
            return result
        try:
            with os.scandir(root) as entries:
                candidates = sorted(
                    entry.path for entry in entries
                    if entry.name.lower().startswith("llama")
                    and entry.is_dir(follow_symlinks=False))
        except OSError:
            continue
        for directory in candidates:
            result = _in_directory(directory)
            if result:
                return result
    raise FileNotFoundError(
        "llama-server was not found. Install/extract llama.cpp, then place its folder "
        "inside ComfyUI, beside the portable ComfyUI folder, in /workspace, or on PATH. "
        "Alternatively set llama_server_path or llama_server_search_dirs in config.json. "
        "Keep the executable together with its required libraries.")
