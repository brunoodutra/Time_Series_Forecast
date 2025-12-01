#!/usr/bin/env bash
set -euo pipefail

# step.sh - Automatiza build, execução de inferência e agendamento via Docker
# Uso: ./step.sh

#############################################
# write_env: Gera arquivo .env para o compose
# - Define CODE_PATH (pasta do repositório bind-mount)
# - Define LOG_PATH (pasta de logs no host)
#############################################
write_env() {
  local script_dir repo_root log_dir env_file
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  repo_root=$(cd "$script_dir/.." && pwd)
  log_dir="$repo_root/docker/logs"
  env_file="$script_dir/.env"

  mkdir -p "$log_dir"

  cat > "$env_file" <<EOF
CODE_PATH=$repo_root
LOG_PATH=$log_dir
EOF

  echo "[step] .env criado em: $env_file"
  echo "       CODE_PATH=$repo_root"
  echo "       LOG_PATH=$log_dir"
}

#############################################
# build_images: Constrói as imagens do compose
# - Usa docker-compose.yml localizado em ./docker
#############################################
build_images() {
  local script_dir
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  (cd "$script_dir" && docker compose build)
  echo "[step] Imagens construídas com sucesso."
}

#############################################
# run_inference: Executa inferência pontual
# - Roda o serviço 'inference' uma vez com --rm
#############################################
run_inference() {
  local script_dir
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  (cd "$script_dir" && docker compose run --rm inference)
  echo "[step] Inferência pontual executada."
}

#############################################
# start_scheduler: Sobe o agendador em background
# - Inicia serviço 'scheduler' com restart unless-stopped
#############################################
start_scheduler() {
  local script_dir
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  (cd "$script_dir" && docker compose up -d scheduler)
  echo "[step] Scheduler iniciado em background. Ver logs em docker/logs/inference.log"
}

#############################################
# main: Fluxo completo
# - Cria .env, dá build, roda inferência e sobe scheduler
#############################################
main() {
  write_env
  build_images
  run_inference
  start_scheduler
  echo "[step] Finalizado. Dicas:"
  echo "  - Parar scheduler: (cd docker && docker compose down)"
  echo "  - Verificar logs: tail -n 100 docker/logs/inference.log"
}

main "$@"