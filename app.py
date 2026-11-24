"""
Trabalho - 12524 Aprendizado Supervisionado
Regressão linear treinada e executada em nuvem (Streamlit Community Cloud).

Rodar localmente:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

import ml_core as mc

PASTA_MODELO = Path(__file__).parent / "modelo"
ARQ_MODELO = PASTA_MODELO / "modelo.json"

st.set_page_config(page_title="Regressão Linear em Nuvem", page_icon="📈", layout="wide")
st.title("📈 Regressão Linear em Nuvem")
st.caption("12524 · Aprendizado Supervisionado · previsão da coluna 'time' a partir dos 5 momentos anteriores")


def modelo_salvo():
    return mc.carregar_pacote(ARQ_MODELO) if ARQ_MODELO.exists() else None


def ler_upload(arquivo, senha):
    return mc.ler_arquivo(arquivo.name, arquivo.getvalue(), senha)


def formatar(d):
    return {k: round(v, 4) for k, v in d.items()}


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Estado do modelo")
    pacote = modelo_salvo()
    if pacote:
        st.success(f"Modelo treinado com {pacote['n_treino']} amostras")
        st.write("Entradas:", ", ".join(pacote["colunas_x"]))
    else:
        st.warning("Nenhum modelo treinado")

    st.divider()
    st.subheader("Resetar modelo")
    st.caption("Apaga o modelo do servidor para treinar em outra base de dados.")
    confirma = st.checkbox("Confirmo que quero apagar o modelo")
    if st.button("🗑️ Resetar", disabled=not (pacote and confirma)):
        ARQ_MODELO.unlink(missing_ok=True)
        st.session_state.pop("resultado", None)
        st.toast("Modelo apagado.")
        st.rerun()

    st.divider()
    st.subheader("Segurança dos arquivos")
    senha = st.text_input(
        "Senha (opcional)", type="password",
        help="Se preenchida: arquivos .enc enviados são descriptografados com ela "
             "e os resultados para download saem criptografados (.zip.enc).",
    )

aba_treino, aba_teste, aba_sobre = st.tabs(["1 · Treino", "2 · Teste / Aplicação", "Como funciona"])

# ---------------------------------------------------------------- 1. TREINO
with aba_treino:
    st.subheader("Upload da base de treino")
    st.caption("Formatos aceitos: .csv, .zip (com o .csv dentro) ou .enc (criptografado com a senha da barra lateral).")
    arq_treino = st.file_uploader("Base de treino (com a coluna 'time')", type=["csv", "zip", "enc"], key="treino")

    with st.expander("Hiperparâmetros do gradiente descendente"):
        c1, c2, c3 = st.columns(3)
        eta = c1.number_input("Taxa de aprendizado (η)", 0.001, 0.99, 0.5, 0.05, format="%.3f")
        epocas = c2.number_input("Épocas", 100, 50000, 5000, 500)
        k = c3.number_input("k (folds da validação cruzada)", 2, 20, 5, 1)

    if pacote and arq_treino:
        st.info("Já existe um modelo treinado. Treinar de novo vai substituí-lo.")

    if st.button("🚀 Treinar modelo", type="primary", disabled=arq_treino is None):
        try:
            df = ler_upload(arq_treino, senha)
            with st.spinner("Rodando validação cruzada e treinando o modelo final..."):
                pacote_novo, _, _ = mc.treinar(df, k=int(k), eta=float(eta), epocas=int(epocas))
            PASTA_MODELO.mkdir(exist_ok=True)
            mc.salvar_pacote(pacote_novo, ARQ_MODELO)
            st.session_state.pop("resultado", None)
            st.success("Modelo treinado e salvo no servidor.")
            st.rerun()
        except Exception as e:
            st.error(f"Erro no treino: {e}")

    if pacote:
        st.divider()
        st.subheader("Expectativa de desempenho")
        st.caption(
            f"Obtida por validação cruzada {pacote['k']}-fold: a média dos folds é o desempenho "
            "esperado em dados nunca vistos. A normalização min-max é refeita dentro de cada fold."
        )
        med, dp = pacote["expectativa"]["media"], pacote["expectativa"]["desvio"]
        cols = st.columns(4)
        for col, m in zip(cols, ["MSE", "RMSE", "MAE", "R2"]):
            col.metric(f"{m} esperado", f"{med[m]:.2f}", help=f"desvio-padrão entre folds: {dp[m]:.2f}")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Resultado de cada fold**")
            st.dataframe(pd.DataFrame(pacote["folds"]).set_index("fold").round(3), use_container_width=True)
            st.write("**Desempenho na própria base de treino** (só referência, não é a expectativa)")
            st.json(formatar(pacote["metricas_treino"]))
        with c2:
            st.write("**Convergência do gradiente descendente** (MSE normalizado por época)")
            hist = pd.DataFrame({"MSE": pacote["historico_mse"]},
                                index=[i * 50 for i in range(len(pacote["historico_mse"]))])
            hist.index.name = "época"
            st.line_chart(hist)
            st.write("**Pesos aprendidos** (sobre dados normalizados)")
            m = pacote["modelo"]
            st.dataframe(pd.DataFrame({"peso (w)": m["w"]}, index=pacote["colunas_x"]).round(4),
                         use_container_width=True)
            st.write(f"bias (b) = {m['b']:.4f}")

# ---------------------------------------------------------------- 2. TESTE / APLICAÇÃO
with aba_teste:
    if not pacote:
        st.warning("Treine um modelo primeiro na aba **Treino**.")
    else:
        st.subheader("Upload da base de teste ou de aplicação")
        st.caption("Se o arquivo tiver a coluna 'time', o modelo é avaliado (desempenho real). "
                   "Se não tiver, só são geradas as previsões.")
        arq_teste = st.file_uploader("Base de teste / aplicação", type=["csv", "zip", "enc"], key="teste")

        if arq_teste and st.button("▶️ Executar modelo", type="primary"):
            try:
                df = ler_upload(arq_teste, senha)
                saida, real = mc.aplicar(pacote, df)
                st.session_state["resultado"] = (arq_teste.name, saida, real)
            except Exception as e:
                st.error(f"Erro na execução: {e}")

        if "resultado" in st.session_state:
            nome, saida, real = st.session_state["resultado"]
            base = Path(nome).name.split(".")[0] + "_previsto"
            nome_dl, dados_dl = mc.empacotar_saida(saida, base, senha)

            st.write(f"**Arquivo:** {nome} · {len(saida)} linhas")
            st.download_button("⬇️ Baixar previsões (" + nome_dl + ")", dados_dl, file_name=nome_dl,
                               mime="application/octet-stream", type="primary")
            st.dataframe(saida.head(50), use_container_width=True)

            if real is None:
                st.info("Base sem rótulos: nenhuma avaliação feita, só as previsões para download.")
            else:
                st.divider()
                st.subheader("Desempenho real × esperado")
                med = pacote["expectativa"]["media"]
                cols = st.columns(4)
                for col, m in zip(cols, ["MSE", "RMSE", "MAE", "R2"]):
                    col.metric(f"{m} real", f"{real[m]:.2f}", delta=f"{real[m] - med[m]:+.2f} vs esperado",
                               delta_color="normal" if m == "R2" else "inverse")
                st.dataframe(mc.comparar(pacote["expectativa"], real).round(3), use_container_width=True,
                             hide_index=True)
                st.caption("'Dentro de ±2 desvios' indica se o resultado real ficou na faixa de variação "
                           "observada entre os folds da validação cruzada.")

                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Real × previsto**")
                    st.scatter_chart(saida, x=mc.ALVO, y=f"{mc.ALVO}_previsto")
                with c2:
                    st.write("**Erro por amostra**")
                    st.bar_chart(saida["erro"])

# ---------------------------------------------------------------- SOBRE
with aba_sobre:
    st.markdown("""
**Pipeline ponta-a-ponta**

1. **Entrada do arquivo**: `.csv`, `.zip` ou `.enc` (zip/csv criptografado com AES via senha).
2. **Limpeza**: nomes de colunas padronizados, valores convertidos para número, linhas inválidas removidas.
3. **Normalização min-max**: `x' = (x − min) / (max − min)`. O min e o max vêm **só do treino**
   e são reaplicados no teste (o teste nunca influencia o modelo).
4. **Modelo**: regressão linear como perceptron de ativação linear, `y = Σ wᵢ·xᵢ + b`,
   treinado por **gradiente descendente** minimizando o MSE:
   `wᵢ ← wᵢ − η·(Yp − Yr)·xᵢ` e `b ← b − η·(Yp − Yr)` (média sobre a base a cada época).
5. **Expectativa de desempenho**: validação cruzada k-fold (média dos folds).
6. **Teste**: previsões desnormalizadas para a escala original; se houver rótulo, calcula o desempenho real
   e compara com o esperado.
7. **Saída**: CSV compactado em `.zip` e, se houver senha, criptografado (`.zip.enc`).
8. **Reset**: apaga o modelo salvo no servidor para treinar com outra base.
""")
