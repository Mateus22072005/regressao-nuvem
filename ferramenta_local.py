"""
Ferramenta do lado do usuário (ponta-a-ponta): prepara os arquivos ANTES de enviar
para a nuvem e abre os resultados DEPOIS de baixar.

Exemplos:
  python ferramenta_local.py enviar  dados/PI_train.csv            -> PI_train.zip
  python ferramenta_local.py enviar  dados/PI_train.csv minhasenha -> PI_train.zip.enc
  python ferramenta_local.py abrir   PI_test_previsto.zip.enc minhasenha -> PI_test_previsto.csv
"""
import sys
from pathlib import Path

import ml_core as mc


def enviar(caminho, senha=""):
    p = Path(caminho)
    z = mc.compactar(p.name, p.read_bytes())
    if senha:
        destino = p.with_name(p.stem + ".zip.enc")
        destino.write_bytes(mc.criptografar(z, senha))
    else:
        destino = p.with_name(p.stem + ".zip")
        destino.write_bytes(z)
    print(f"Gerado: {destino}  ({p.stat().st_size} -> {destino.stat().st_size} bytes)")


def abrir(caminho, senha=""):
    p = Path(caminho)
    df = mc.ler_arquivo(p.name, p.read_bytes(), senha)
    destino = p.with_name(p.name.split(".")[0] + ".csv")
    df.to_csv(destino, index=False)
    print(f"Gerado: {destino}  ({len(df)} linhas)")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("enviar", "abrir"):
        print(__doc__)
        sys.exit(1)
    acao, arq = sys.argv[1], sys.argv[2]
    senha = sys.argv[3] if len(sys.argv) > 3 else ""
    (enviar if acao == "enviar" else abrir)(arq, senha)
