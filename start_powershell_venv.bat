@echo off
:: Script que abre 3 abas no Windows Terminal:
:: - Aba 1: abre em finance\AI\Classification\Real_Time_Inference, ativa venv e executa o script
:: - Abas 2 e 3: abrem na raiz do projeto com venv ativada para uso interativo

set "ROOT=D:\Projetos_python\Time_Series_Forecast"
set "INFER=%ROOT%\finance\AI\Classification\Real_Time_Inference"
set "API=%ROOT%\finance\API"
set "DASH=D:\Projetos_python\cryptodashpro\cryptodash-pro"
set "VENV_PS1=%ROOT%\.venv\Scripts\Activate.ps1"
set "VENV_ACT=%ROOT%\.venv\Scripts\activate"

echo Abrindo 3 abas no Windows Terminal com venv ativada...

wt -w 0 ^
  new-tab -d "%INFER%" powershell -ExecutionPolicy Bypass -NoExit -Command "if (Test-Path '%VENV_PS1%') { . '%VENV_PS1%' } elseif (Test-Path '%VENV_ACT%') { . '%VENV_ACT%' } & python .\classification_in_produtction.py" ^
  ; new-tab -d "%API%" powershell -ExecutionPolicy Bypass -NoExit -Command "if (Test-Path '%VENV_PS1%') { . '%VENV_PS1%' } elseif (Test-Path '%VENV_ACT%') { . '%VENV_ACT%' } & uvicorn API_setup:app --reload --host 0.0.0.0" ^
  ; new-tab -d "%DASH%" powershell -ExecutionPolicy Bypass -NoExit -Command "python -m http.server 8005"
