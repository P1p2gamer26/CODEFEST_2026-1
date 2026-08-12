# Crea un entorno APARTE con torch+CUDA para medir si la GPU vuelve factible
# un encoder grande (gte tarda 97 h en esta CPU, ver dev/docs/PLAN_MAESTRO.md).
#
# Deliberadamente NO toca .venv: ese entorno produce hoy una entrega valida y
# reproducible byte a byte, y reinstalar torch encima puede romperlo. El
# riesgo de perder la entrega no vale el ahorro de 2 GB de disco.
#
# La GPU solo se usaria para CONSTRUIR el indice (fase offline). ADL evalua
# con el indice ya entregado y confirmo que para evaluar basta CPU, asi que
# esto no agrega ningun requisito al evaluador.
#
# Uso:  powershell -File dev\scripts\preparar_venv_cuda.ps1

$ErrorActionPreference = "Stop"
$Venv = ".venv-cuda"
# cu124 cubre Turing (sm_75, que es la GTX 1650) y tiene ruedas para 3.13.
$Indice = "https://download.pytorch.org/whl/cu124"

Write-Output "== creando $Venv =="
if (-not (Test-Path $Venv)) { & ".venv\Scripts\python.exe" -m venv $Venv }

$Py = "$Venv\Scripts\python.exe"
Write-Output "== python del entorno nuevo =="
& $Py --version

Write-Output "== instalando torch+CUDA (descarga grande, varios minutos) =="
& $Py -m pip install --upgrade pip --quiet
& $Py -m pip install torch --index-url $Indice
if ($LASTEXITCODE -ne 0) { Write-Output "FALLO instalando torch"; exit 1 }

Write-Output "== instalando sentence-transformers =="
& $Py -m pip install sentence-transformers
if ($LASTEXITCODE -ne 0) { Write-Output "FALLO instalando sentence-transformers"; exit 1 }

Write-Output "== comprobando CUDA =="
& $Py -c @"
import torch
print('torch', torch.__version__)
print('CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('capacidad de computo:', torch.cuda.get_device_capability(0))
    print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1))
"@

Write-Output "LISTO"
