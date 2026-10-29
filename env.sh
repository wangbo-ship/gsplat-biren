#!/usr/bin/env bash
# Source this file before local training, benchmarking, or validation commands.

if [ -n "${BASH_SOURCE:-}" ]; then
  _GSPLAT_BIREN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _GSPLAT_BIREN_ROOT="$(pwd)"
fi

export GSPLAT_BIREN_ROOT="${_GSPLAT_BIREN_ROOT}"
export GSPLAT_AGENT_DATASET_DIR="${GSPLAT_BIREN_ROOT}/.agents/dataset"
export GSPLAT_AGENT_REFERENCE_DIR="${GSPLAT_BIREN_ROOT}/.agents/reference"
export GSPLAT_AGENT_TEMPORARY_DIR="${GSPLAT_BIREN_ROOT}/.agents/temporary"
export GSPLAT_AGENT_OUTPUT_DIR="${GSPLAT_BIREN_ROOT}/.agents/output"

mkdir -p \
  "${GSPLAT_AGENT_DATASET_DIR}" \
  "${GSPLAT_AGENT_REFERENCE_DIR}" \
  "${GSPLAT_AGENT_TEMPORARY_DIR}/cache" \
  "${GSPLAT_AGENT_TEMPORARY_DIR}/native-build" \
  "${GSPLAT_AGENT_TEMPORARY_DIR}/pycache" \
  "${GSPLAT_AGENT_TEMPORARY_DIR}/pytest-cache" \
  "${GSPLAT_AGENT_TEMPORARY_DIR}/pytest-tmp" \
  "${GSPLAT_AGENT_TEMPORARY_DIR}/tmp" \
  "${GSPLAT_AGENT_OUTPUT_DIR}"

_gsplat_biren_prepend_path() {
  _gsplat_biren_var_name="$1"
  _gsplat_biren_entry="$2"
  eval "_gsplat_biren_current=\"\${${_gsplat_biren_var_name}:-}\""
  case ":${_gsplat_biren_current}:" in
    *":${_gsplat_biren_entry}:"*) ;;
    *)
      if [ -n "${_gsplat_biren_current}" ]; then
        export "${_gsplat_biren_var_name}=${_gsplat_biren_entry}:${_gsplat_biren_current}"
      else
        export "${_gsplat_biren_var_name}=${_gsplat_biren_entry}"
      fi
      ;;
  esac
}

_gsplat_biren_prepend_path PATH "/usr/local/birensupa/sdk/latest/brcc/bin"

_gsplat_biren_prepend_path PYTHONPATH "${GSPLAT_BIREN_ROOT}"
_gsplat_biren_prepend_path PYTHONPATH "${GSPLAT_AGENT_TEMPORARY_DIR}/python-local"

export PYTHONPYCACHEPREFIX="${GSPLAT_AGENT_TEMPORARY_DIR}/pycache"
export TORCH_EXTENSIONS_DIR="${GSPLAT_AGENT_TEMPORARY_DIR}/native-build"
export XDG_CACHE_HOME="${GSPLAT_AGENT_TEMPORARY_DIR}/cache"
export TMPDIR="${GSPLAT_AGENT_TEMPORARY_DIR}/tmp"

unset _gsplat_biren_var_name
unset _gsplat_biren_entry
unset _gsplat_biren_current
unset -f _gsplat_biren_prepend_path
