"""E42: la normalizacion por tamano penaliza al documento que inunda el pool."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from barrido_norm_doc_e42 import agregar_normalizado, conteos_del_corpus
from src.retrieval.search import Hit


def _hit(doc_id, score, i):
    return Hit(rank=i, score=score, chunk_id=f"{doc_id}#{i}", doc_id=doc_id,
               fuente="f", texto="t", formato="pdf", fenomeno=1, idioma="es", fila=i)


def _caso_e41():
    """Replica el patron de q037: el relevante tiene el MEJOR chunk y pocos,
    el ganador tiene muchos mediocres y gana por 0.0087 con top5."""
    relevante = [_hit("REL", s, i) for i, s in enumerate([1.75, 1.55, 1.50], start=0)]
    ganador = [_hit("GAN", 1.60, 100 + i) for i in range(30)]
    return relevante + ganador


def test_alpha_cero_es_identico_a_top5():
    from src.retrieval.aggregate import aggregate_documents

    hits = _caso_e41()
    base = aggregate_documents(hits, top_n=2, strategy="top5")
    norm = agregar_normalizado(hits, top_n=2, m=5, alpha=0.0, denominador="n_pool")
    assert [d.doc_id for d in norm] == [d.doc_id for d in base]
    assert norm[0].score == base[0].score


def test_alpha_cero_deja_ganar_al_que_inunda():
    """Sin normalizacion GAN gana: 5*1.60 = 8.00 contra 1.75+1.55+1.50 = 4.80."""
    norm = agregar_normalizado(_caso_e41(), top_n=2, m=5, alpha=0.0, denominador="n_pool")
    assert norm[0].doc_id == "GAN"


def test_alpha_suficiente_rescata_al_relevante():
    """Con alpha=0.5: GAN 8.00/30**0.5 = 1.46, REL 4.80/3**0.5 = 2.77."""
    norm = agregar_normalizado(_caso_e41(), top_n=2, m=5, alpha=0.5, denominador="n_pool")
    assert norm[0].doc_id == "REL"


def test_denominador_corpus_usa_el_conteo_externo():
    """Con n_corpus el denominador NO es cuantos chunks trae al pool."""
    conteos = {"REL": 3, "GAN": 3}
    norm = agregar_normalizado(_caso_e41(), top_n=2, m=5, alpha=0.5,
                               denominador="n_corpus", conteos_corpus=conteos)
    # mismo denominador para los dos => el orden vuelve a ser el de top5
    assert norm[0].doc_id == "GAN"


def test_cache_invalido_se_recalcula(tmp_path):
    """Un cache con firma (tamano, mtime) que no coincide con metadata.jsonl
    se descarta y se recalcula, en vez de servir conteos viejos en silencio."""
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text('{"doc_id": "A"}\n{"doc_id": "A"}\n{"doc_id": "B"}\n',
                        encoding="utf-8")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"firma": [999999, 0.0], "conteos": {"A": 1}}),
                     encoding="utf-8")

    conteos = conteos_del_corpus(path=metadata, cache=cache)
    assert conteos == {"A": 2, "B": 1}
