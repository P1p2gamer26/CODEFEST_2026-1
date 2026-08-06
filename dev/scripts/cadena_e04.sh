#!/usr/bin/env bash
# Encadena E04 sin depender de ninguna sesion: espera a que la indexacion de
# e5-large termine y arranca su barrido de evaluacion.
#
# Se lanza asi (nunca como hijo de la sesion, punto "Corridas largas"):
#   setsid nohup bash dev/scripts/cadena_e04.sh > dev/intermedios/log_cadena_e04.txt 2>&1 < /dev/null & disown
#
# La indexacion es reanudable; esta cadena NO la relanza a proposito. Si el
# proceso muere sin dejar el indice, hay que mirar el log y decidir a mano:
# relanzar a ciegas 55 h de CPU sobre una causa desconocida es peor que parar.
set -u
cd "$(dirname "$0")/../.."

INDICE=dev/intermedios/e5large/encoder_multilingual-e5-large/index.faiss

echo "[$(date '+%F %T')] esperando a que termine build_corpus_index"
while pgrep -f "build_corpus_index.py .*e5-large" > /dev/null; do
    sleep 600
done

if [ ! -f "$INDICE" ]; then
    echo "[$(date '+%F %T')] la indexacion termino SIN dejar $INDICE; ver dev/intermedios/log_e5large.txt"
    exit 1
fi
echo "[$(date '+%F %T')] indice listo ($(stat -c%s "$INDICE") bytes), arrancando el barrido"

.venv/bin/python dev/scripts/barrido_repuntuador_e04.py
echo "[$(date '+%F %T')] barrido terminado con codigo $?"
