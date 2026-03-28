import json
import os
from typing import Any
from jax import export as jax_export

def save_jax_exported(
    exp: jax_export.Exported,
    bin_file_path: str,
    *,
    vjp_order: int = 0,
) -> None:
    dirname = os.path.dirname(bin_file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    if os.path.exists(bin_file_path):
        raise ValueError(f'File {bin_file_path} already exists.')
    with open(bin_file_path, 'wb') as f:
        f.write(exp.serialize(vjp_order=vjp_order))

def load_jax_exported(bin_file_path: str) -> jax_export.Exported:
    if not os.path.exists(bin_file_path):
        raise ValueError(f'File {bin_file_path} does not exist.')
    with open(bin_file_path, 'rb') as f:
        return jax_export.deserialize(bytearray(f.read()))

def save_native_model(nativemodel_path: str, model_fn_map: dict[str, jax_export.Exported]) -> None:
    model_fn_dir = os.path.join(nativemodel_path, 'model_fn')
    os.makedirs(model_fn_dir, exist_ok=True)
    
    metadata = {}
    for method_key, exp in model_fn_map.items():
        filename = f"{method_key.replace(' ', '_')}.bin"
        file_path = os.path.join(model_fn_dir, filename)
        save_jax_exported(exp, file_path)
        
        metadata[method_key] = {
            "calling_convention_version": exp.calling_convention_version,
            "file_path": filename
        }
        
    metadata_file = os.path.join(model_fn_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_native_model(nativemodel_path: str) -> dict[str, jax_export.Exported]:
    model_fn_dir = os.path.join(nativemodel_path, 'model_fn')
    metadata_file = os.path.join(model_fn_dir, 'metadata.json')
    if not os.path.exists(metadata_file):
        raise ValueError(f'Model path {metadata_file} does not exist.')
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    model_fn_map = {}
    for method_key, method_metadata in metadata.items():
        file_path = os.path.join(model_fn_dir, method_metadata['file_path'])
        model_fn_map[method_key] = load_jax_exported(file_path)
    return model_fn_map
