"""
Exporta modelos Keras para TFLite com quantização.

Este script carrega um modelo salvo, aplica a conversão para TFLite
com o modo de quantização escolhido e salva o arquivo .tflite para uso posterior.

Modos suportados:
- float32: conversão sem quantização
- dynamic: quantização de faixa dinâmica (sem dataset representativo)
- fp16: pesos em float16
- int8: quantização inteira completa (requer dataset representativo)

Uso:
  python export_tflite_quantized.py --model_dir "<pasta_do_modelo>" \
    --output_dir "<pasta_saida>" --quantization int8 --symbol BTC --interval 4h --samples 128
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np
import tensorflow as tf
from keras import backend as K


def _attach_processing_path() -> None:
    """Adiciona o diretório Processing ao sys.path para permitir importações internas."""
    processing_source_path = os.path.abspath("./../../../Processing/")
    if processing_source_path not in sys.path:
        sys.path.append(processing_source_path)


def matthews_correlation_coefficient(y_true, y_pred):
    """Calcula MCC, usado como métrica customizada em alguns modelos Keras."""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def load_model_with_custom_objects(model_dir: Path):
    """Carrega um modelo Keras com objetos customizados necessários do treino."""
    _attach_processing_path()
    # Import dos utilitários de treino (losses) e geração de features
    from DataLoaderPipeline import FeaturesDataGenerator  # noqa: F401
    from CustomTrainLosses import CustomTrainLosses

    # Perdas customizadas usadas no projeto
    ctl = CustomTrainLosses()
    weighted_categorical_crossentropy_loss = ctl.weighted_categorical_crossentropy(np.ones(3))

    model = tf.keras.models.load_model(
        str(model_dir),
        custom_objects={
            'loss': weighted_categorical_crossentropy_loss,
            'matthews_correlation_coefficient': matthews_correlation_coefficient,
        },
    )
    return model


def load_parameters(model_dir: Path) -> dict:
    """Lê o arquivo config.json salvo junto ao modelo e retorna os parâmetros."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json não encontrado em: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_representative_dataset(parameters: dict, symbol: str, interval: str, samples: int = 128) -> Callable[[], List[np.ndarray]]:
    """Cria um gerador de dataset representativo para quantização INT8 usando dados reais."""
    _attach_processing_path()
    from DataLoaderPipeline import FeaturesDataGenerator, scrapingHistoricalData

    # Coleta uma janela recente de dados para calibrar a quantização
    SHD = scrapingHistoricalData()
    # Usar ~60 dias para garantir diversidade
    import pandas as pd
    from datetime import datetime, timedelta
    start_time = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
    data_df = SHD.get_crypto_historical_data([symbol], interval, start_time)

    # Geração de features igual ao pipeline de inferência
    datatype = parameters.get('datatype', '2D')
    lookback = parameters['lookback']
    pred_days = parameters.get('pred_days', 0)
    selected_features = parameters['features_indicators']
    min_norm = parameters.get('min_norm', 0.0)
    max_norm = parameters.get('max_norm', 1.0)

    dataGen = FeaturesDataGenerator(
        data_df,
        datatype=datatype,
        lookback=lookback,
        pred_days=pred_days,
        shuffle=False,
        batch_size=32,
        selected_features=selected_features,
        data_augmentation=False,
        min_max_norm_features=[min_norm, max_norm],
    )

    x_data_inference = dataGen.comput_features(data_df, pred_days=0)
    x_data = dataGen.apply_NomrMinmax(x_data_inference, min_norm, max_norm, axis=0)
    if datatype == '2D':
        x_data = np.transpose(x_data, [0, 2, 1]).reshape(-1, dataGen.inputShape[1], dataGen.inputShape[2], 1)

    # Limita quantidade de amostras
    num = int(min(samples, x_data.shape[0]))

    def generator():
        for i in range(num):
            yield [x_data[i:i+1].astype(np.float32)]

    return generator


def convert_to_tflite(model: tf.keras.Model,
                      output_path: Path,
                      quantization: str = 'dynamic',
                      representative_dataset: Optional[Callable] = None) -> None:
    """Converte e salva o modelo Keras para TFLite no modo desejado."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == 'float32':
        # Sem otimizações
        pass
    elif quantization == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8':
        if representative_dataset is None:
            raise ValueError("Quantização 'int8' requer representative_dataset.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        raise ValueError(f"Modo de quantização inválido: {quantization}")

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)


def main():
    """Executa a conversão/quantização conforme os argumentos informados."""
    parser = argparse.ArgumentParser(description="Exportar TFLite quantizado de um modelo Keras")
    parser.add_argument('--model_dir', type=str, required=True, help='Diretório do modelo Keras/SavedModel')
    parser.add_argument('--output_dir', type=str, default='./tflite_exports', help='Diretório de saída para .tflite')
    parser.add_argument('--quantization', type=str, default='dynamic', choices=['float32', 'dynamic', 'fp16', 'int8'], help='Modo de quantização')
    parser.add_argument('--symbol', type=str, default='BTC', help='Símbolo para dataset representativo (INT8)')
    parser.add_argument('--interval', type=str, default='4h', help='Intervalo de coleta (ex.: 1h, 4h, 1d)')
    parser.add_argument('--samples', type=int, default=128, help='Número de amostras para representative_dataset (INT8)')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carrega modelo e parâmetros
    model = load_model_with_custom_objects(model_dir)
    try:
        parameters = load_parameters(model_dir)
    except FileNotFoundError:
        # Quantizações sem dataset representativo podem seguir sem config.json
        parameters = {}

    # Cria generator representativo quando necessário
    representative = None
    if args.quantization == 'int8':
        representative = build_representative_dataset(parameters, args.symbol, args.interval, args.samples)

    # Define nome do arquivo de saída
    suffix = args.quantization
    out_path = output_dir / f"model_{suffix}.tflite"

    convert_to_tflite(model, out_path, quantization=args.quantization, representative_dataset=representative)
    print(f"Modelo TFLite salvo em: {out_path}")


if __name__ == '__main__':
    main()