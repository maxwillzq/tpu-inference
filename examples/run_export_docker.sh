#!/bin/bash
# run_export_docker.sh

IMAGE_NAME="vllm-tpu-cpu"
EXPORT_DIR_HOST="/tmp/exported_model_dir"
WORKSPACE_DIR="/usr/local/google/home/johnqiangzhang/projects/dive-deep-vllm"
VLLM_SUBMODULE="${WORKSPACE_DIR}/submodules/vllm"
TPU_INFERENCE_SUBMODULE="${WORKSPACE_DIR}/submodules/tpu-inference"

echo "Building Docker Image: ${IMAGE_NAME}..."
cd "${TPU_INFERENCE_SUBMODULE}" || exit 1
docker build -f docker/Dockerfile -t "${IMAGE_NAME}" .

echo -e "\nRunning Export Workflow inside Docker..."
docker run --rm \
  -v "${VLLM_SUBMODULE}:/workspace/vllm" \
  -v "${TPU_INFERENCE_SUBMODULE}:/workspace/tpu_inference" \
  -v "${EXPORT_DIR_HOST}:/tmp/exported_model_dir" \
  -e GOOGLE_EXPORT_MODEL_PATH="/tmp/exported_model_dir" \
  -e GOOGLE_EXPORT_TOPOLOGY="v4-8" \
  -e VLLM_TPU_VERSION_OVERRIDE="6" \
  -e XLA_FLAGS="--xla_force_host_platform_device_count=1" \
  "${IMAGE_NAME}" bash -c "pip install flatbuffers && python3 /workspace/tpu_inference/examples/hello_world_offline.py"
