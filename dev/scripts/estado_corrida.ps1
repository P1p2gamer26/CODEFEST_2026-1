# Informe de estado de una corrida larga: cuanto lleva, cuanto falta y que
# recursos esta usando. Pensado para llamarse repetidamente desde un loop.
#
# Uso:  powershell -File dev\scripts\estado_corrida.ps1 -Log <ruta> [-Total <n>]

param(
    [string]$Log = "dev\intermedios\gte_rerank\cuda_install.log",
    [int]$Total = 0
)

function Barra($frac) {
    $n = [int]($frac * 28)
    if ($n -lt 0) { $n = 0 }; if ($n -gt 28) { $n = 28 }
    "[" + ("#" * $n) + ("." * (28 - $n)) + "]"
}

if (-not (Test-Path $Log)) { Write-Output "no existe $Log"; exit 0 }

$item = Get-Item $Log
# El inicio se toma del PROCESO, no del archivo. Windows tiene "file
# tunneling": si un archivo se borra y se recrea con el mismo nombre en
# menos de 15 s, hereda el CreationTime del viejo. Como el log se sobrescribe
# en cada relanzamiento, CreationTime reportaba la corrida ANTERIOR -- daba
# "lleva 37 min" para un proceso de 40 segundos.
$proc = Get-Process python -ErrorAction SilentlyContinue |
        Where-Object { $_.WorkingSet64 -gt 300MB } |
        Sort-Object StartTime -Descending | Select-Object -First 1
$inicio = if ($proc) { $proc.StartTime } else { $item.CreationTime }
$ahora = Get-Date
$transcurrido = $ahora - $inicio
$fuente = if ($proc) { "proceso" } else { "archivo" }
Write-Output ("TIEMPO   inicio {0:HH:mm:ss} ({1})   lleva {2:hh\:mm\:ss}   (ultima escritura {3:HH:mm:ss})" -f `
    $inicio, $fuente, $transcurrido, $item.LastWriteTime)

# Progreso de descarga de pip. Ojo: la barra usa \r, hay que partir por CR, y
# hay que quedarse con la descarga MAS GRANDE en curso -- quedarse con la
# ultima muestra la de cualquier dependencia de 80 kB y no dice nada de la
# rueda de 2,5 GB, que es la que importa.
$texto = Get-Content $Log -Raw -ErrorAction SilentlyContinue
if ($texto) {
    $trozos = $texto -split "`r"
    $mayor = $null; $mayorTot = 0
    foreach ($t in $trozos) {
        if ($t -match '([\d\.]+)/([\d\.]+) MB') {
            if ([double]$Matches[2] -gt $mayorTot) { $mayorTot = [double]$Matches[2]; $mayor = $t }
            elseif ([double]$Matches[2] -eq $mayorTot) { $mayor = $t }  # el mas reciente de esa descarga
        }
    }
    if ($mayor -and $mayor -match '([\d\.]+)/([\d\.]+) MB(.*?([\d\.]+) (MB|kB)/s)?') {
        $hecho = [double]$Matches[1]; $tot = [double]$Matches[2]
        $frac = if ($tot -gt 0) { $hecho / $tot } else { 0 }
        Write-Output ("DESCARGA {0} {1:P0}   {2:N0}/{3:N0} MB" -f (Barra $frac), $frac, $hecho, $tot)
        if ($Matches[4] -and $Matches[5] -eq 'MB' -and [double]$Matches[4] -gt 0) {
            $seg = ($tot - $hecho) / [double]$Matches[4]
            Write-Output ("         a {0} MB/s  ->  faltan ~{1:mm\:ss}" -f $Matches[4], [TimeSpan]::FromSeconds($seg))
        }
    }
}

# Fase actual: la ultima linea con contenido real, ignorando barras de progreso.
$fase = Get-Content $Log -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '^(==|Installing|Successfully|Collecting|Downloading|torch |CUDA|GPU|LISTO|FALLO)' } |
        Select-Object -Last 1
if ($fase) { Write-Output ("FASE     " + $fase.Trim().Substring(0, [Math]::Min(72, $fase.Trim().Length))) }

# Progreso de codificacion. El ETA se calcula ACA a partir de los lotes
# observados; deliberadamente NO se lee el "faltan ~" que imprime el worker,
# porque su formula estuvo mal y daba un ETA creciente. Un informe que
# reenvia el numero de otro hereda sus errores.
$lineas = Get-Content $Log -ErrorAction SilentlyContinue | Where-Object { $_ -match 'codificados ([\d,]+)/([\d,]+)' }
if ($lineas) {
    $ultima = $lineas | Select-Object -Last 1
    if ($ultima -match 'codificados ([\d,]+)/([\d,]+)') {
        # Los contadores llevan separador de miles ("128,526"): sin quitarlo,
        # [int] corta en la coma y el porcentaje sale absurdo (daba "400 %").
        $h = [int]($Matches[1] -replace ',', ''); $t = [int]($Matches[2] -replace ',', '')
        $frac = $h / $t
        # Ritmo medido entre el primer lote y el ultimo: excluye la carga del
        # modelo, que son decenas de segundos y falsea el promedio hacia abajo.
        $vel = 0.0
        if ($lineas.Count -ge 2 -and ($lineas[0] -match 'codificados (\d+)/') ) {
            $h0 = [int]($Matches[1] -replace ',', '')
            if ($ultima -match '\((\d+\.?\d*) min' ) { $minUlt = [double]$Matches[1] }
            if ($lineas[0] -match '\((\d+\.?\d*) min') { $min0 = [double]$Matches[1] }
            if ($minUlt -gt $min0) { $vel = ($h - $h0) / (($minUlt - $min0) * 60) }
        }
        # Con un solo lote no hay dos puntos para medir la pendiente, y usar
        # el tiempo total incluiria la carga del modelo (~100 s), que da una
        # estimacion muy pesimista (daba 15 h para una corrida de 7,5 h). En
        # ese caso se usa el chunks/s que reporta el propio worker, cuya
        # formula ya esta corregida.
        if ($vel -le 0 -and $ultima -match '([\d\.]+) chunks/s') { $vel = [double]$Matches[1] }
        if ($vel -le 0 -and $transcurrido.TotalSeconds -gt 0) { $vel = $h / $transcurrido.TotalSeconds }
        Write-Output ("CHUNKS   {0} {1:P0}   {2:N0}/{3:N0}" -f (Barra $frac), $frac, $h, $t)
        if ($vel -gt 0) {
            $falta = [TimeSpan]::FromSeconds(($t - $h) / $vel)
            $fin = (Get-Date).Add($falta)
            Write-Output ("         a {0:N1} chunks/s   FALTAN ~{1:hh\:mm\:ss}   (termina ~{2:HH:mm})" -f $vel, $falta, $fin)
        }
    }
}

$os = Get-CimInstance Win32_OperatingSystem
$libre = [int]($os.FreePhysicalMemory / 1KB); $tot = [int]($os.TotalVisibleMemorySize / 1KB)
$disco = Get-PSDrive C
Write-Output ("RAM      {0:N0} MB libres de {1:N0}  ({2:P0} en uso)" -f $libre, $tot, (1 - $libre / $tot))
Write-Output ("DISCO    {0:N1} GB libres" -f ($disco.Free / 1GB))

$procs = Get-Process python, pip -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 100MB }
foreach ($p in $procs) {
    Write-Output ("PROCESO  {0} PID {1}  {2:N0} MB  CPU {3:N0}s" -f $p.Name, $p.Id, ($p.WorkingSet64 / 1MB), $p.CPU)
}
if (-not $procs) { Write-Output "PROCESO  ninguno pesado activo" }

try {
    $g = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
    if ($g) {
        $c = $g -split ',\s*'
        Write-Output ("GPU      {0}% uso   {1}/{2} MiB VRAM   {3}C" -f $c[0], $c[1], $c[2], $c[3])
    }
} catch { }
