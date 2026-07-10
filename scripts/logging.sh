#!/usr/bin/env bash

init_script_logging() {
  local script_name="$1"
  local project_dir="$2"
  shift 2

  if [[ "${LOG_DISABLED:-0}" == "1" ]]; then
    return
  fi

  local timestamp
  timestamp="${LOG_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
  LOG_DIR="${LOG_DIR:-${project_dir}/log}"
  LOG_FILE="${LOG_FILE:-${LOG_DIR}/${script_name}_${timestamp}.log}"
  export LOG_DIR LOG_FILE

  mkdir -p "${LOG_DIR}"

  if [[ "${NOHUP:-0}" == "1" && "${_PIPELINE_LOG_CHILD:-0}" != "1" ]]; then
    local pid_file="${LOG_FILE%.log}.pid"
    echo "[log] NOHUP=1, start background process"
    echo "[log] LOG_FILE=${LOG_FILE}"
    env "_PIPELINE_LOG_CHILD=1" "NOHUP=0" "LOG_ACTIVE=0" "LOG_DIR=${LOG_DIR}" "LOG_FILE=${LOG_FILE}" \
      nohup bash "$0" "$@" > "${LOG_FILE}" 2>&1 &
    local child_pid=$!
    echo "${child_pid}" > "${pid_file}"
    echo "[log] PID=${child_pid}"
    echo "[log] PID_FILE=${pid_file}"
    echo "[log] tail -f ${LOG_FILE}"
    exit 0
  fi

  if [[ "${LOG_ACTIVE:-0}" == "1" ]]; then
    echo "[log] reuse parent log: ${LOG_FILE}"
    return
  fi

  export LOG_ACTIVE=1

  if [[ "${_PIPELINE_LOG_CHILD:-0}" == "1" ]]; then
    echo "[log] LOG_FILE=${LOG_FILE}"
    echo "[log] PID=$$"
    echo "[log] STARTED_AT=$(date '+%F %T')"
    trap 'status=$?; echo "[log] FINISHED_AT=$(date "+%F %T")"; echo "[log] EXIT_STATUS=${status}"' EXIT
    return
  fi

  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "[log] LOG_FILE=${LOG_FILE}"
  echo "[log] PID=$$"
  echo "[log] STARTED_AT=$(date '+%F %T')"
  trap 'status=$?; echo "[log] FINISHED_AT=$(date "+%F %T")"; echo "[log] EXIT_STATUS=${status}"' EXIT
}
