# Vigilante de RAM para las corridas largas de codificacion.
#
# Esta maquina tiene 7,6 GB y un modelo de 305 M mas el indice no caben
# holgados. Si la RAM libre cae por debajo del umbral, libera memoria matando
# procesos de una lista BLANCA explicita -- nunca algo del sistema, nunca el
# worker de python, nunca la sesion de Claude.
#
# Uso:
#   powershell -File dev/scripts/vigilante_ram.ps1 -UmbralMB 500

param(
    [int]$UmbralMB = 500,
    [int]$IntervaloSeg = 30,
    [string]$Log = "dev\intermedios\gte_rerank\vigilante.log"
)

# Solo estos se pueden matar. Son recuperables: las pestanas del navegador se
# recargan solas y el editor reabre los archivos. Todo lo demas se respeta,
# aunque consuma mucho: matar a ciegas por RAM alta puede tumbar el propio
# trabajo que se esta vigilando.
$Matables = @('brave', 'chrome', 'msedge', 'firefox', 'Teams', 'Spotify', 'Discord')

# Nunca, bajo ninguna condicion.
$Intocables = @('python', 'claude', 'System', 'Idle', 'csrss', 'wininit', 'services', 'lsass', 'smss', 'winlogon')

function Escribir($msg) {
    $linea = "{0}  {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Output $linea
    Add-Content -Path $Log -Value $linea -Encoding utf8
}

Escribir "vigilante iniciado (umbral $UmbralMB MB, intervalo ${IntervaloSeg}s)"
Escribir "matables: $($Matables -join ', ')"

# Periodo de gracia: cargar un modelo de 305 M lleva minutos antes de que el
# worker acumule CPU. Sin esto el vigilante se declaraba sin trabajo que
# cuidar a los 0 segundos y se moria (pasó en la primera corrida).
$Gracia = 6
$vueltas = 0

while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $libreMB = [int]($os.FreePhysicalMemory / 1KB)
    $vueltas++

    # Si el worker ya no existe, el vigilante no tiene nada que cuidar.
    $worker = Get-Process -Name python -ErrorAction SilentlyContinue |
              Where-Object { $_.WorkingSet64 -gt 300MB }
    if (-not $worker -and $vueltas -gt $Gracia) {
        Escribir "no hay worker de python activo -- vigilante termina"
        break
    }

    if ($libreMB -lt $UmbralMB) {
        Escribir "ALERTA: RAM libre $libreMB MB < $UmbralMB MB"
        $victima = Get-Process -ErrorAction SilentlyContinue |
                   Where-Object { $Matables -contains $_.Name -and $Intocables -notcontains $_.Name } |
                   Sort-Object WorkingSet64 -Descending |
                   Select-Object -First 1
        if ($victima) {
            $mb = [int]($victima.WorkingSet64 / 1MB)
            Escribir "  liberando: $($victima.Name) (PID $($victima.Id), $mb MB)"
            try { Stop-Process -Id $victima.Id -Force -ErrorAction Stop }
            catch { Escribir "  no se pudo matar: $_" }
        } else {
            Escribir "  nada matable en la lista blanca -- solo se registra"
        }
    } else {
        Escribir "ok: RAM libre $libreMB MB"
    }
    Start-Sleep -Seconds $IntervaloSeg
}
