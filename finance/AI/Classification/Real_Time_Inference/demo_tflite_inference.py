"""
Demonstração de inferência com modelo TFLite quantizado usando o pipeline de features.

Este script:
- Carrega `config.json` do modelo para recuperar parâmetros de features e normalização.
- Coleta dados recentes via `scrapingHistoricalData` (Binance) para o símbolo e intervalo.
- Gera `x_float` com `FeaturesDataGenerator` e executa inferência no modelo TFLite.
- Converte probabilidades em sinais de trade usando thresholds `TH`.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import importlib


def _attach_processing_path() -> None:
    """Adiciona o diretório `finance/Processing` ao sys.path baseado no caminho do arquivo."""
    here = Path(__file__).resolve()
    ai_root = here.parents[2]  # .../AI
    processing_source_path = str(ai_root / 'Processing')
    if processing_source_path not in sys.path:
        sys.path.append(processing_source_path)

def _attach_quant_functions_path() -> None:
    """Adiciona o diretório `Quantization/functions` ao sys.path para usar Embedded_Model."""
    here = Path(__file__).resolve()
    classification_root = here.parents[1]  # .../Classification
    quant_funcs_path = str(classification_root / 'Quantization' / 'functions')
    if quant_funcs_path not in sys.path:
        sys.path.append(quant_funcs_path)


def load_parameters(model_dir: Path) -> dict:
    """Carrega parâmetros do `config.json` presente na pasta do modelo."""
    config_path = model_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"config.json não encontrado em {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _create_interpreter(tflite_path: Path, backend: str = 'auto') -> Optional[object]:
    """Cria um Interpreter para TFLite tentando TensorFlow ou tflite_runtime.

    - backend='tf': força uso de TensorFlow (`tf.lite.Interpreter`).
    - backend='tflite_runtime': força uso de `tflite_runtime.Interpreter`.
    - backend='auto': tenta TensorFlow e, se falhar, tenta tflite_runtime.

    Retorna o objeto Interpreter ou None se nenhum backend estiver disponível.
    """
    tf_mod = None
    tr_mod = None
    if backend in ('auto', 'tf'):
        try:
            tf_mod = importlib.import_module('tensorflow')
        except Exception:
            tf_mod = None
    if backend in ('auto', 'tflite_runtime') and tf_mod is None:
        try:
            tr_mod = importlib.import_module('tflite_runtime.interpreter')
        except Exception:
            tr_mod = None

    if tf_mod is not None:
        try:
            return tf_mod.lite.Interpreter(model_path=str(tflite_path))
        except Exception:
            return None
    if tr_mod is not None:
        try:
            return tr_mod.Interpreter(model_path=str(tflite_path))
        except Exception:
            return None
    return None


class TFLiteModel:
    """Wrapper para executar inferência com TFLite (suporta float e int8)."""
    def __init__(self, interpreter: object):
        """Inicializa o wrapper com um Interpreter já criado.

        interpreter: instância de `tf.lite.Interpreter` ou `tflite_runtime.Interpreter`.
        """
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']
        self.input_scale, self.input_zero_point = self.input_details[0].get('quantization', (0.0, 0))
        self.output_scale, self.output_zero_point = self.output_details[0].get('quantization', (0.0, 0))
        self.input_dtype = self.input_details[0]['dtype']
        self.output_dtype = self.output_details[0]['dtype']
        self.input_shape = self.input_details[0]['shape']

    def _ensure_shape(self, x: np.ndarray) -> None:
        """Redimensiona tensor de entrada se o shape divergir do esperado."""
        if tuple(x.shape) != tuple(self.input_shape):
            self.interpreter.resize_tensor_input(self.input_index, x.shape)
            self.interpreter.allocate_tensors()

    def predict_proba(self, x_float: np.ndarray) -> np.ndarray:
        """Executa inferência e retorna probabilidades em float32."""
        if x_float.ndim == 3:
            x_float = x_float[..., np.newaxis]
        outs: List[np.ndarray] = []
        for i in range(x_float.shape[0]):
            x_in = x_float[i:i+1]
            self._ensure_shape(x_in)
            if self.input_dtype == np.int8:
                if self.input_scale == 0:
                    raise ValueError("Input scale=0 para INT8.")
                x_cast = (x_in / self.input_scale + self.input_zero_point).astype(np.int8)
            else:
                x_cast = x_in.astype(self.input_dtype)
            self.interpreter.set_tensor(self.input_index, x_cast)
            self.interpreter.invoke()
            y = self.interpreter.get_tensor(self.output_index)
            if self.output_dtype == np.int8:
                y = (y.astype(np.float32) - self.output_zero_point) * self.output_scale
            else:
                y = y.astype(np.float32)
            outs.append(y[0])
        return np.stack(outs, axis=0)


def generate_signals(preds: np.ndarray, TH: List[float]) -> List[str]:
    """Converte as probabilidades em sinais ['Hold','Buy','Sell'] considerando TH."""
    trade = ['Hold', 'Buy', 'Sell']
    signals = [ trade[np.argmax(p)] if np.max(p) > TH[np.argmax(p)] else trade[0] for p in preds ]
    return signals


def main():
    """Ponto de entrada: coleta dados, gera features, executa TFLite e imprime sinais."""
    import argparse

    parser = argparse.ArgumentParser(description='Demo inferência TFLite')
    parser.add_argument('--tflite_path', type=str, required=True, help='Caminho para o arquivo .tflite')
    parser.add_argument('--model_dir', type=str, required=True, help='Pasta do modelo com config.json')
    parser.add_argument('--symbol', type=str, default='BTC', help='Símbolo da cripto (ex.: BTC)')
    parser.add_argument('--interval', type=str, default='4h', help='Intervalo (ex.: 1h, 4h, 1d)')
    parser.add_argument('--window_days', type=int, default=60, help='Dias de janela para coleta de dados')
    parser.add_argument('--samples', type=int, default=5, help='Número de amostras para exibir')
    parser.add_argument('--backend', type=str, default='auto', choices=['auto','tf','tflite_runtime'], help='Backend para o Interpreter TFLite')
    parser.add_argument('--dry_run', action='store_true', help='Executa pipeline de dados sem inferência (útil quando não há backend disponível)')
    parser.add_argument('--use_embedded', action='store_true', help='Usa funções de Quantization/functions (Embedded_Model) para inferência')
    args = parser.parse_args()

    _attach_processing_path()
    _attach_quant_functions_path()
    from DataLoaderPipeline import FeaturesDataGenerator, scrapingHistoricalData
    # Tenta importar Embedded_Model
    Embedded_Model = None
    try:
        from Embedded_Model import Embedded_Model as _EM
        Embedded_Model = _EM
    except Exception:
        Embedded_Model = None

    # Carrega parâmetros do modelo
    parameters = load_parameters(Path(args.model_dir))

    # Coleta dados
    SHD = scrapingHistoricalData()
    start_time = (datetime.today() - timedelta(days=args.window_days)).strftime('%Y-%m-%d')
    data_df = SHD.get_crypto_historical_data([args.symbol], args.interval, start_time)

    # Gera features e normaliza
    dataGen = FeaturesDataGenerator(
        data_df,
        datatype=parameters.get('datatype', '2D'),
        lookback=parameters['lookback'],
        pred_days=parameters.get('pred_days', 0),
        shuffle=False,
        batch_size=32,
        selected_features=parameters['features_indicators'],
        data_augmentation=False,
        min_max_norm_features=[parameters.get('min_norm', 0.0), parameters.get('max_norm', 1.0)],
    )

    x_inf = dataGen.comput_features(data_df, pred_days=0)
    x_float = dataGen.apply_NomrMinmax(x_inf, parameters.get('min_norm', 0.0), parameters.get('max_norm', 1.0), axis=0)
    if parameters.get('datatype', '2D') == '2D':
        x_float = np.transpose(x_float, [0, 2, 1]).reshape(-1, dataGen.inputShape[1], dataGen.inputShape[2], 1)

    # Seleciona runner: Embedded_Model (se pedido e disponível) ou Interpreter direto
    preds = None
    if args.use_embedded and Embedded_Model is not None:
        try:
            # Embedded_Model espera destination_path sem a extensão .tflite
            dest_no_ext = str(Path(args.tflite_path).with_suffix(''))
            em = Embedded_Model(destination_path=dest_no_ext)
            preds = em.predict_batch_data(x_float)
        except Exception as e:
            print('Falha ao usar Embedded_Model, caindo para Interpreter direto:', e)

    if preds is None:
        # Cria Interpreter do TFLite
        interpreter = _create_interpreter(Path(args.tflite_path), backend=args.backend)
        if interpreter is None:
            print('Nenhum backend TFLite disponível (TensorFlow ou tflite-runtime).')
            print('Dicas:')
            print(' - Em Windows, TensorFlow costuma ser o caminho mais estável:')
            print('   Use Python 3.11/3.12 e instale: pip install tensorflow')
            print(' - Alternativamente: pip install tflite-runtime (se houver wheel para sua plataforma).')
            print('Como não há backend disponível, entrarei em modo dry-run do pipeline.')
            print('Shape de x_float preparado:', x_float.shape)
            if not args.dry_run:
                print('Para executar inferência real, configure o backend e rode novamente sem --dry_run.')
                return
            # Modo dry-run: sem inferência, apenas finaliza com saída amigável.
            print('Dry-run concluído. Nenhuma inferência foi executada.')
            return

        # Executa inferência TFLite
        tm = TFLiteModel(interpreter)
        preds = tm.predict_proba(x_float)

    # Gera sinais
    TH = parameters.get('TH', [0.5, 0.5, 0.5])
    signals = generate_signals(preds, TH)

    # Exibe últimos resultados
    print('Últimos sinais:')
    for s in signals[-args.samples:]:
        print('-', s)

    # Exibe shape e exemplo de probabilidades
    print('Shape preds:', preds.shape)
    print('Exemplo de probs:', preds[-1])


if __name__ == '__main__':
    main()