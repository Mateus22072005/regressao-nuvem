"""
Núcleo de Machine Learning do trabalho (sem interface).

Segue o que foi visto em aula (12524 - Aprendizado Supervisionado):
  - Regressão linear treinada como um perceptron com GRADIENTE DESCENDENTE
    (w <- w - eta * dE/dw ; b <- b - eta * dE/db), minimizando o MSE.
  - Normalização MIN-MAX das entradas (e da saída) para garantir convergência.
  - Expectativa de desempenho via VALIDAÇÃO CRUZADA k-fold:
    a média dos folds é o valor esperado para o "mundo real".
  - Métrica principal: MSE (média dos erros quadráticos). Também RMSE, MAE e R².
"""
import io
import json
import zipfile

import numpy as np
import pandas as pd

ALVO = "time"


# --------------------------------------------------------------------------
# Tratamento de dados
# --------------------------------------------------------------------------
def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas, converte para número e remove linhas inválidas."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # índice salvo por engano
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def separar(df: pd.DataFrame, colunas_x=None):
    """Retorna X, y (y = None se a base não tiver a coluna alvo)."""
    if colunas_x is None:
        colunas_x = [c for c in df.columns if c != ALVO]
    faltando = [c for c in colunas_x if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no arquivo: {faltando}")
    X = df[colunas_x].to_numpy(dtype=float)
    y = df[ALVO].to_numpy(dtype=float) if ALVO in df.columns else None
    return X, y, list(colunas_x)


class MinMax:
    """Normalização min-max: x' = (x - min) / (max - min).
    min e max são aprendidos SOMENTE no treino e reaplicados no teste."""

    def fit(self, a):
        a = np.asarray(a, dtype=float)
        self.min_ = a.min(axis=0)
        self.max_ = a.max(axis=0)
        self.amp_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, a):
        return (np.asarray(a, dtype=float) - self.min_) / self.amp_

    def inverse(self, a):
        return np.asarray(a, dtype=float) * self.amp_ + self.min_

    def to_dict(self):
        return {"min": np.atleast_1d(self.min_).tolist(), "max": np.atleast_1d(self.max_).tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.min_ = np.array(d["min"], dtype=float)
        m.max_ = np.array(d["max"], dtype=float)
        m.amp_ = np.where(m.max_ - m.min_ == 0, 1.0, m.max_ - m.min_)
        return m


# --------------------------------------------------------------------------
# Modelo: regressão linear (perceptron com ativação linear) + gradiente descendente
# --------------------------------------------------------------------------
class RegressaoLinearGD:
    def __init__(self, eta=0.5, epocas=5000, semente=42):
        self.eta = eta
        self.epocas = epocas
        self.semente = semente

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # 1) Normalização min-max (aprendida só nos dados de treino)
        self.norm_x = MinMax().fit(X)
        self.norm_y = MinMax().fit(y.reshape(-1, 1))
        Xn = self.norm_x.transform(X)
        yn = self.norm_y.transform(y.reshape(-1, 1)).ravel()

        # 2) Pesos iniciais aleatórios (como no perceptron)
        rng = np.random.default_rng(self.semente)
        self.w = rng.uniform(-0.5, 0.5, Xn.shape[1])
        self.b = rng.uniform(-0.5, 0.5)

        # 3) Gradiente descendente (batch): usa a média do erro de toda a base
        n = len(yn)
        self.historico_mse = []
        for _ in range(self.epocas):
            yp = Xn @ self.w + self.b          # Y previsto
            erro = yp - yn                     # (Yp - Yr)
            grad_w = (Xn.T @ erro) / n         # dE/dw_i = (Yp - Yr) * x_i  (média)
            grad_b = erro.mean()               # dE/db   = (Yp - Yr)        (média)
            self.w -= self.eta * grad_w        # w(t+1) = w(t) - eta * dE/dw
            self.b -= self.eta * grad_b        # b(t+1) = b(t) - eta * dE/db
            self.historico_mse.append(float((erro ** 2).mean()))
        return self

    def predict(self, X):
        Xn = self.norm_x.transform(np.asarray(X, dtype=float))
        yn = Xn @ self.w + self.b
        return self.norm_y.inverse(yn.reshape(-1, 1)).ravel()

    # --- persistência (o modelo fica salvo no servidor em nuvem) ---
    def to_dict(self):
        return {
            "eta": self.eta, "epocas": self.epocas, "semente": self.semente,
            "w": self.w.tolist(), "b": float(self.b),
            "norm_x": self.norm_x.to_dict(), "norm_y": self.norm_y.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(d["eta"], d["epocas"], d["semente"])
        m.w = np.array(d["w"], dtype=float)
        m.b = float(d["b"])
        m.norm_x = MinMax.from_dict(d["norm_x"])
        m.norm_y = MinMax.from_dict(d["norm_y"])
        return m


# --------------------------------------------------------------------------
# Avaliação
# --------------------------------------------------------------------------
def metricas(y_real, y_prev):
    y_real = np.asarray(y_real, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    erro = y_real - y_prev
    mse = float((erro ** 2).mean())
    ss_tot = float(((y_real - y_real.mean()) ** 2).sum())
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.abs(erro).mean()),
        "R2": float(1 - (erro ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
    }


def validacao_cruzada(X, y, k=5, eta=0.5, epocas=5000, semente=42):
    """k-fold: treina em k-1 partes e testa na restante, k vezes.
    A normalização é refeita DENTRO de cada fold (evita vazamento de dados).
    Retorna média, desvio-padrão e o resultado de cada fold."""
    n = len(y)
    idx = np.random.default_rng(semente).permutation(n)
    folds = np.array_split(idx, k)
    resultados = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        m = RegressaoLinearGD(eta, epocas, semente).fit(X[tr], y[tr])
        r = metricas(y[te], m.predict(X[te]))
        r["fold"] = i + 1
        resultados.append(r)
    df = pd.DataFrame(resultados).set_index("fold")
    return {
        "media": df.mean().to_dict(),
        "desvio": df.std(ddof=1).to_dict(),
        "folds": df,
    }


def treinar(df_bruto: pd.DataFrame, k=5, eta=0.5, epocas=5000, semente=42):
    """Fluxo completo de treino: limpeza -> k-fold (expectativa) -> modelo final."""
    df = limpar(df_bruto)
    if ALVO not in df.columns:
        raise ValueError(f"A base de treino precisa da coluna '{ALVO}' (rótulo).")
    X, y, cols = separar(df)
    cv = validacao_cruzada(X, y, k, eta, epocas, semente)
    modelo = RegressaoLinearGD(eta, epocas, semente).fit(X, y)
    pacote = {
        "colunas_x": cols,
        "n_treino": int(len(y)),
        "k": k,
        "modelo": modelo.to_dict(),
        "expectativa": {"media": cv["media"], "desvio": cv["desvio"]},
        "metricas_treino": metricas(y, modelo.predict(X)),
        "folds": cv["folds"].reset_index().to_dict("records"),
        # curva de convergência do gradiente descendente (1 ponto a cada 50 épocas)
        "historico_mse": modelo.historico_mse[::50],
    }
    return pacote, modelo, cv


def aplicar(pacote: dict, df_bruto: pd.DataFrame):
    """Usa o modelo salvo em uma base de teste/aplicação.
    Retorna (df_com_previsoes, metricas_reais ou None se não houver rótulo)."""
    modelo = RegressaoLinearGD.from_dict(pacote["modelo"])
    df = limpar(df_bruto)
    X, y, _ = separar(df, pacote["colunas_x"])
    prev = modelo.predict(X)
    saida = df.copy()
    saida[f"{ALVO}_previsto"] = np.round(prev, 4)
    if y is None:
        return saida, None
    saida["erro"] = np.round(y - prev, 4)
    return saida, metricas(y, prev)


def comparar(expectativa: dict, real: dict) -> pd.DataFrame:
    """Tabela desempenho esperado (k-fold) x real (teste)."""
    linhas = []
    for m in ["MSE", "RMSE", "MAE", "R2"]:
        med, dp, r = expectativa["media"][m], expectativa["desvio"][m], real[m]
        linhas.append({
            "Métrica": m,
            "Esperado (média k-fold)": med,
            "Desvio k-fold": dp,
            "Real (teste)": r,
            "Diferença (real - esperado)": r - med,
            "Dentro de ±2 desvios?": "Sim" if abs(r - med) <= 2 * dp else "Não",
        })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Arquivos ponta-a-ponta: leitura de .csv / .zip / .enc e geração de .zip / .enc
# --------------------------------------------------------------------------
def _chave(senha: str, sal: bytes) -> bytes:
    import base64
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=sal, iterations=200_000)
    return base64.urlsafe_b64encode(kdf.derive(senha.encode()))


def criptografar(dados: bytes, senha: str) -> bytes:
    """AES (Fernet) com chave derivada da senha. Formato: 16 bytes de sal + token."""
    import os
    from cryptography.fernet import Fernet
    sal = os.urandom(16)
    return sal + Fernet(_chave(senha, sal)).encrypt(dados)


def descriptografar(dados: bytes, senha: str) -> bytes:
    from cryptography.fernet import Fernet, InvalidToken
    try:
        return Fernet(_chave(senha, dados[:16])).decrypt(dados[16:])
    except InvalidToken:
        raise ValueError("Senha incorreta ou arquivo corrompido.")


def compactar(nome_csv: str, csv_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(nome_csv, csv_bytes)
    return buf.getvalue()


def ler_arquivo(nome: str, dados: bytes, senha: str = "") -> pd.DataFrame:
    """Aceita .csv, .zip (com um .csv dentro) ou .enc (zip ou csv criptografado)."""
    nome = nome.lower()
    if nome.endswith(".enc"):
        if not senha:
            raise ValueError("Arquivo criptografado: informe a senha.")
        dados = descriptografar(dados, senha)
        nome = nome[:-4]
        if dados[:2] == b"PK":  # assinatura de arquivo zip
            nome = nome if nome.endswith(".zip") else nome + ".zip"
    if nome.endswith(".zip") or dados[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(dados)) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise ValueError("O .zip não contém nenhum .csv.")
            dados = z.read(csvs[0])
    return pd.read_csv(io.BytesIO(dados))


def empacotar_saida(df: pd.DataFrame, nome_base: str, senha: str = ""):
    """Gera o arquivo de download: .zip (e .enc se houver senha). Retorna (nome, bytes)."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    z = compactar(f"{nome_base}.csv", csv_bytes)
    if senha:
        return f"{nome_base}.zip.enc", criptografar(z, senha)
    return f"{nome_base}.zip", z


def salvar_pacote(pacote: dict, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(pacote, f, ensure_ascii=False, indent=2)


def carregar_pacote(caminho):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)
