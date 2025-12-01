import argparse
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


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


def discover_models(models_root: Path) -> List[Tuple[Path, Path]]:
    """
    Varre a pasta `models_root` e retorna pares `(tflite_path, model_dir)`.

    Regras:
    - Procura arquivos `*.tflite` (recursivo).
    - Infere `model_dir` como uma pasta irmã com o mesmo nome do arquivo
      sem extensão, contendo `config.json`.
    - Ignora arquivos cujo diretório do modelo não seja encontrado.
    """
    pairs: List[Tuple[Path, Path]] = []
    for tflite in models_root.rglob("*.tflite"):
        candidate_dir = tflite.parent / tflite.stem
        if (candidate_dir / "config.json").exists():
            pairs.append((tflite, candidate_dir))
        else:
            print(f"Aviso: config.json não encontrado para {tflite.name}; esperado em {candidate_dir}")
    # Ordena para execução determinística
    pairs.sort(key=lambda p: p[0].name)
    return pairs


def run_auto_once(
    python_executable: Path,
    demo_script: Path,
    models_root: Path,
    backend: str,
    use_embedded: bool,
    samples: int,
    skip_pipeline: bool,
    symbol: str,
    interval: str,
    log_path: Optional[Path] = None,
) -> None:
    """
    Executa uma passada única de inferência para todos os modelos .tflite
    encontrados em `models_root`.
    """
    pairs = discover_models(models_root)
    if not pairs:
        print(f"Nenhum .tflite encontrado em {models_root}")
        return
    for tflite_path, model_dir in pairs:
        cmd = build_demo_command(
            python_executable,
            demo_script,
            tflite_path,
            model_dir,
            backend=backend,
            use_embedded=use_embedded,
            samples=samples,
            skip_pipeline=skip_pipeline,
            symbol=symbol,
            interval=interval,
        )
        run_inference_once(cmd, log_path=log_path)


def run_auto_scheduler_loop(
    interval_sec: int,
    python_executable: Path,
    demo_script: Path,
    models_root: Path,
    backend: str,
    use_embedded: bool,
    samples: int,
    skip_pipeline: bool,
    symbol: str,
    interval: str,
    log_path: Optional[Path] = None,
) -> None:
    """
    Executa, continuamente, inferências para todos os modelos em `models_root`
    a cada `interval_sec` segundos, reavaliando a lista em cada ciclo.
    """
    print(f"Agendador AUTO iniciado. Pasta: {models_root}. Intervalo: {interval_sec}s. Ctrl+C para encerrar.")
    try:
        while True:
            start_cycle = time.time()
            run_auto_once(
                python_executable,
                demo_script,
                models_root,
                backend,
                use_embedded,
                samples,
                skip_pipeline,
                symbol,
                interval,
                log_path,
            )
            end_cycle = time.time()
            print(f"Ciclo AUTO concluído em {end_cycle - start_cycle:.2f}s. Próximo em {interval_sec}s.")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("Agendador AUTO finalizado pelo usuário.")


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
    # Modo manual: requer tflite_path e model_dir
    # Modo auto: varre pasta de modelos e ignora tflite_path/model_dir
    p.add_argument("--tflite_path", type=str, help="Caminho para o arquivo .tflite (modo manual)")
    p.add_argument("--model_dir", type=str, help="Pasta do modelo contendo config.json (modo manual)")
    p.add_argument("--auto", action="store_true", help="Ativa modo auto: executa todos os .tflite de uma pasta de modelos")
    p.add_argument(
        "--models_root",
        type=str,
        default="",
        help="Pasta raiz de modelos (.tflite). Padrão: Experiments/Cryptos/models do projeto",
    )
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

    # Corrige a raiz do projeto: schedule_inference.py está em
    # finance/AI/Classification/Real_Time_Inference; precisamos subir 4 níveis
    # para chegar à raiz do repositório (ex.: /app ou Time_Series_Forecast).
    project_root = Path(__file__).resolve().parents[4]
    demo_script = project_root / "finance" / "AI" / "Classification" / "Real_Time_Inference" / "demo_tflite_inference.py"

    python_exe = Path(sys.executable)
    log_path = Path(args.log_path) if args.log_path else None
    # Determina pasta padrão de modelos se não fornecida
    default_models_root = project_root / "finance" / "AI" / "Classification" / "Experiments" / "Cryptos" / "models"
    models_root = Path(args.models_root) if args.models_root else default_models_root

    if args.auto:
        # Modo AUTO: roda todos os modelos da pasta especificada
        if args.once:
            run_auto_once(
                python_exe,
                demo_script,
                models_root,
                backend=args.backend,
                use_embedded=args.use_embedded,
                samples=args.samples,
                skip_pipeline=args.skip_pipeline,
                symbol=args.symbol,
                interval=args.interval,
                log_path=log_path,
            )
            sys.exit(0)
        else:
            run_auto_scheduler_loop(
                args.interval_sec,
                python_exe,
                demo_script,
                models_root,
                backend=args.backend,
                use_embedded=args.use_embedded,
                samples=args.samples,
                skip_pipeline=args.skip_pipeline,
                symbol=args.symbol,
                interval=args.interval,
                log_path=log_path,
            )
    else:
        # Modo MANUAL: requer caminhos específicos
        if not args.tflite_path or not args.model_dir:
            print("Erro: forneça --tflite_path e --model_dir, ou use --auto.", file=sys.stderr)
            sys.exit(2)
        tflite_path = Path(args.tflite_path)
        model_dir = Path(args.model_dir)

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