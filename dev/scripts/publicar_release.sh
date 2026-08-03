#!/usr/bin/env bash
# Publica los indices como GitHub Release. Ver README seccion 1.1.
#
# Los .faiss NO se comprimen (floats casi aleatorios, no bajan nada); el texto
# si, 4,4x. La sintaxis archivo#nombre renombra el asset al subirlo: el Release
# es una lista plana y sin eso los tres index.faiss chocarian entre si.
#
# Uso:  bash dev/scripts/publicar_release.sh [tag]
set -euo pipefail

TAG="${1:-indices-v3}"
B=Entrega/base_vectorial
MINILM="$B/encoder_paraphrase-multilingual-MiniLM-L12-v2"

for f in "$MINILM/index.faiss" \
         "$B/encoder_gte-multilingual-base/index.faiss" \
         "$B/encoder_multilingual-e5-base/index.faiss" \
         "$MINILM/metadata.jsonl.gz" \
         "$B/grafo/grafo.graphml.gz"; do
    [ -f "$f" ] || { echo "FALTA: $f"; exit 1; }
done

echo "publicando $TAG ..."
gh release create "$TAG" \
   --title "Indices v3 (cascada de tres encoders)" \
   --notes "MiniLM + gte + e5 sobre 128.526 chunks del corpus de ADL. F1@3 0,386 y NDCG@10 0,406 sobre las 41 consultas anotadas a mano. Rehidratacion: ver README seccion 1.1." \
   "$MINILM/index.faiss#minilm-index.faiss" \
   "$B/encoder_gte-multilingual-base/index.faiss#gte-index.faiss" \
   "$B/encoder_multilingual-e5-base/index.faiss#e5-index.faiss" \
   "$MINILM/metadata.jsonl.gz#metadata.jsonl.gz" \
   "$B/grafo/grafo.graphml.gz#grafo.graphml.gz"

echo "RELEASE LISTO"
