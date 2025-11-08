@echo off
:: Script para abrir o PowerShell diretamente na pasta de inferência
:: (finance\AI\Classification\Real_Time_Inference), ativar a venv do projeto
:: e executar o arquivo classification_in_produtction.py.
::
:: Uso: duplo clique neste arquivo .bat
:: Requisitos: venv em D:\Projetos_python\Time_Series_Forecast\venv ou D:\Projetos_python\Time_Series_Forecast\.venv

set "PROJECT_DIR=D:\Projetos_python\Time_Series_Forecast"
set "INFERENCE_REL=finance\AI\Classification\Real_Time_Inference"
set "SCRIPT_NAME=classification_in_produtction.py"

start "RealTimeInference" powershell -NoExit -ExecutionPolicy Bypass -NoLogo -Command "
  $project = '%PROJECT_DIR%';
  $inferencePath = Join-Path $project '%INFERENCE_REL%';
  if (-not (Test-Path $inferencePath)) {
    Write-Error \"Pasta de inferência não encontrada: $inferencePath\";
    Set-Location $project;
  } else {
    Set-Location $inferencePath;
  }

  $activated = $false;
  $candidates = @(
    (Join-Path $project 'venv\\Scripts\\Activate.ps1'),
    (Join-Path $project '.venv\\Scripts\\Activate.ps1'),
    (Join-Path $project 'venv\\Scripts\\activate.ps1'),
    (Join-Path $project '.venv\\Scripts\\activate.ps1')
  );
  foreach ($p in $candidates) {
    if (Test-Path $p) {
      Write-Host \"Ativando venv via: $p\";
      try {
        . $p;
        $activated = $true;
        break;
      } catch {
        Write-Warning \"Falha ao ativar com $p: $($_.Exception.Message)\";
      }
    }
  }
  if (-not $activated) {
    Write-Warning \"Não foi possível ativar a venv automaticamente.\";
    Write-Host \"Ative manualmente, por exemplo: .\\.venv\\Scripts\\activate\";
  }

  $script = '.\\%SCRIPT_NAME%';
  if (Test-Path $script) {
    Write-Host \"Executando: python $script\";
    python $script;
  } else {
    Write-Error \"Script não encontrado: $script\";
    Write-Host \"Conteúdo da pasta atual:\";
    Get-ChildItem -Name | Select-Object -First 20 | ForEach-Object { Write-Host $_ };
  }
"