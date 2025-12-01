import argparse
import sys
import time
import subprocess
from pathlib import Path


def build_demo_command(
    python_executable: Path,
    demo_script: Path,
    tflite_path: Path,
    model_dir: Path,
    backend: str,
    use_embedded: bool,
    samples: int,
    skip_pipeline: bool,
    symbol: str,
    interval: str,
):
    """
    Constrói a linha de comando para executar o demo TFLite com os parâmetros fornecidos.

    Parâmetros:
    - python_executable: caminho do Python (idealmente da venv).
    - demo_script: caminho do script demo_tflite_inference.py.
    - tflite_path: caminho do arquivo .tflite.
    - model_dir: pasta de modelo contendo config.json.
    - backend: backend para o Interpreter (ex.: 'tf').
    - use_embedded: se deve usar Embedded_Model.
    - samples: quantidade de amostras a exibir.
    - skip_pipeline: se deve pular a pipeline de dados.
    - symbol: símbolo (ex.: 'BTC').
    - interval: intervalo (ex.: '4h').

    Retorna:
    - lista com os argumentos para subprocess.run.
    """
    cmd = [
        str(python_executable),
        str(demo_script),
        "--tflite_path",
        str(tflite_path),
        "--model_dir",
        str(model_dir),
        "--backend",
        backend,
        "--samples",
        str(samples),
        "--symbol",
        symbol,
        "--interval",
        interval,
    ]

    if use_embedded:
        cmd.append("--use_embedded")
    if skip_pipeline:
        cmd.append("--skip_pipeline")

    return cmd


def run_inference_once(cmd, log_path: Path | None = None) -> int:
    """
    Executa uma inferência única chamando o demo via subprocess.

    Parâmetros:
    - cmd: lista de argumentos preparada por build_demo_command.
    - log_path: arquivo de log opcional para salvar stdout/stderr.

    Retorna:
    - código de saída do processo.
    """
    print(f"Executando: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Imprime e registra saída
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if log_path:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n=== Execução ===\n")
                f.write("Comando:\n")
                f.write(" ".join(cmd) + "\n")
                f.write("STDOUT:\n")
                f.write(proc.stdout + "\n")
                if proc.stderr:
                    f.write("STDERR:\n")
                    f.write(proc.stderr + "\n")
        except Exception as e:
            print(f"Falha ao gravar log: {e}", file=sys.stderr)

    return proc.returncode


def run_scheduler_loop(
    interval_sec: int,
    cmd_args: list[str],
    log_path: Path | None = None,
):
    """
    Executa um loop infinito, chamando o demo periodicamente a cada `interval_sec` segundos.

    Parâmetros:
    - interval_sec: intervalo em segundos entre execuções.
    - cmd_args: lista de argumentos para subprocess (comando completo).
    - log_path: caminho opcional do arquivo de log.
    """
    print(f"Agendador iniciado. Intervalo: {interval_sec}s. Pressione Ctrl+C para encerrar.")
    try:
        while True:
            start = time.time()
            code = run_inference_once(cmd_args, log_path=log_path)
            end = time.time()
            print(f"Execução concluída (exit={code}) em {end - start:.2f}s. Próxima em {interval_sec}s.")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("Agendador finalizado pelo usuário.")


def parse_args():
    """
    Faz o parse dos argumentos de linha de comando para configurar o agendamento.
    """
    p = argparse.ArgumentParser(description="Agendador de inferência TFLite periódica")
    p.add_argument("--tflite_path", required=True, type=str, help="Caminho para o arquivo .tflite")
    p.add_argument("--model_dir", required=True, type=str, help="Pasta do modelo contendo config.json")
    p.add_argument("--interval_sec", type=int, default=900, help="Intervalo em segundos entre execuções (padrão: 900 = 15 min)")
    p.add_argument("--backend", type=str, default="tf", choices=["tf", "tflite_runtime", "auto"], help="Backend para o Interpreter TFLite")
    p.add_argument("--use_embedded", action="store_true", help="Usa Embedded_Model para inferência")
    p.add_argument("--skip_pipeline", action="store_true", help="Pula pipeline (gera lote sintético)")
    p.add_argument("--samples", type=int, default=3, help="Número de amostras para exibir")
    p.add_argument("--symbol", type=str, default="BTC", help="Símbolo (ex.: BTC)")
    p.add_argument("--interval", type=str, default="4h", help="Intervalo de dados (ex.: 4h)")
    p.add_argument("--log_path", type=str, default="", help="Caminho opcional para arquivo de log de saídas")
    p.add_argument("--once", action="store_true", help="Executa apenas uma vez e encerra")
    return p.parse_args()


def main():
    """
    Ponto de entrada do agendador. Monta o comando do demo e inicia o loop
    de execução periódica ou uma execução única.
    """
    args = parse_args()

    project_root = Path(__file__).resolve().parents[3]
    demo_script = project_root / "finance" / "AI" / "Classification" / "Real_Time_Inference" / "demo_tflite_inference.py"

    python_exe = Path(sys.executable)
    tflite_path = Path(args.tflite_path)
    model_dir = Path(args.model_dir)
    log_path = Path(args.log_path) if args.log_path else None

    cmd = build_demo_command(
        python_exe,
        demo_script,
        tflite_path,
        model_dir,
        backend=args.backend,
        use_embedded=args.use_embedded,
        samples=args.samples,
        skip_pipeline=args.skip_pipeline,
        symbol=args.symbol,
        interval=args.interval,
    )

    if args.once:
        code = run_inference_once(cmd, log_path=log_path)
        sys.exit(code)
    else:
        run_scheduler_loop(args.interval_sec, cmd, log_path=log_path)


if __name__ == "__main__":
    main()