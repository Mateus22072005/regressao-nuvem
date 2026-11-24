# Regressão Linear em Nuvem — 12524 Aprendizado Supervisionado

App web (Streamlit) que **treina e executa remotamente** uma regressão linear para prever a coluna `time`
a partir dos 5 momentos anteriores (`time-5` … `time-1`).

## Arquivos

| Arquivo | O que faz |
|---|---|
| `app.py` | Interface web: upload, treino, teste/aplicação, download, comparação, reset |
| `ml_core.py` | Todo o ML: limpeza, min-max, regressão por gradiente descendente, k-fold, métricas, zip/criptografia |
| `ferramenta_local.py` | Lado do usuário: compacta/criptografa antes de enviar e abre o resultado baixado |
| `requirements.txt` | Dependências instaladas pela nuvem |
| `dados/PI_train.csv` | Base de treino do professor |

---

## Publicar na nuvem (Streamlit Community Cloud, grátis, ~10 min)

1. Crie uma conta em **github.com** (se ainda não tiver).
2. Crie um repositório **público** novo (ex.: `regressao-nuvem`) e envie todos os arquivos desta pasta
   (botão *Add file → Upload files*, arraste tudo, *Commit*).
3. Entre em **share.streamlit.io** com a conta do GitHub.
4. *Create app → Deploy a public app from GitHub*: escolha o repositório, branch `main`, arquivo `app.py`.
5. Clique em **Deploy**. Em 1–3 minutos sai um link do tipo `https://regressao-nuvem.streamlit.app`.

> Para testar no próprio computador antes: `pip install -r requirements.txt` e `streamlit run app.py`.

**Atenção:** o modelo fica salvo no servidor (`modelo/modelo.json`). Se o app ficar dias sem uso,
o Streamlit "hiberna" e pode apagar o modelo: basta treinar de novo (leva poucos segundos).
**Abra o link uns minutos antes da arguição** para ele "acordar".

---

## Roteiro do dia da arguição

1. Abrir o link do app.
2. **(Opcional, mostra o ponta-a-ponta)** No computador:
   `python ferramenta_local.py enviar dados/PI_train.csv senha123` → gera `PI_train.zip.enc`.
   No app, digitar `senha123` na barra lateral.
3. **Treino** → upload do `PI_train.csv` (ou `.zip.enc`) → *Treinar*. Mostrar a **expectativa de desempenho**.
4. **Teste / Aplicação** → upload da **base de teste** (com rótulo) → *Executar* → baixar previsões
   e mostrar **real × esperado**.
5. Upload da **base de aplicação** (sem rótulo) → *Executar* → baixar previsões.
6. Se baixou `.zip.enc`: `python ferramenta_local.py abrir PI_xxx_previsto.zip.enc senha123`.
7. Mostrar o **Reset** (barra lateral) apagando o modelo.

---

## Como cada item da avaliação é atendido

**Treino do modelo em nuvem (2 pts):** o upload vai para o servidor do Streamlit Cloud, que roda a
validação cruzada e o treino e salva o modelo (pesos + min/max) em `modelo/modelo.json`.

**Execução em nuvem para teste e aplicação (4 pts):** a mesma tela aceita os dois casos. Se o arquivo tem a
coluna `time` → gera previsões **e** calcula o desempenho real. Se não tem → só gera as previsões.
Em ambos o CSV de previsões sai para download.

**Tratamento de dados ponta-a-ponta (2 pts):**
- Entrada aceita `.csv`, `.zip` ou `.enc` (AES/Fernet, chave derivada da senha com PBKDF2).
- Limpeza (conversão numérica, remoção de linhas inválidas).
- **Min-max** aprendido **só no treino** e reaplicado no teste (sem vazamento de dados).
- Previsões voltam para a escala original (desnormalização).
- Saída compactada em `.zip` e, com senha, criptografada (`.zip.enc`).

**Comparação real × esperado (2 pts):** tabela e cartões com MSE, RMSE, MAE e R² esperados
(média do k-fold) contra os reais, a diferença e se o real ficou dentro de ±2 desvios dos folds.

---

## Perguntas prováveis do professor (e respostas)

**Como o modelo é treinado?**
Como um perceptron com ativação linear: `y = Σ wᵢxᵢ + b`. Os pesos começam aleatórios e são ajustados
por gradiente descendente minimizando o MSE: `wᵢ ← wᵢ − η·(Yp − Yr)·xᵢ` e `b ← b − η·(Yp − Yr)`,
usando a média do erro de toda a base a cada época (gradiente descendente "batch"). η = 0,5, 5000 épocas.
Conferimos com a solução exata do scikit-learn: o MSE de treino é idêntico (7008,44).

**Por que normalizar?**
Como visto em aula, o gradiente descendente nem sempre converge. Com as entradas em [0, 1] e η < 1 ele
converge estável. A saída também é normalizada no treino e desnormalizada na previsão.

**Como vocês obtêm a expectativa de desempenho?**
Validação cruzada 5-fold. A base é dividida em 5 partes; treina em 4 e testa na restante, 5 vezes.
A média dos folds é o valor esperado no mundo real. Isso é melhor que um único treino-teste, que pode
sair "fácil" ou "difícil" por acaso. A normalização é refeita **dentro** de cada fold para não vazar
informação da parte de teste.

**Por que não usar o erro no próprio treino como expectativa?**
Porque o modelo já viu esses dados: o erro de treino é otimista. Mostramos só como referência
(MSE de treino 7008 contra MSE esperado de cerca de 7241).

**E se no teste aparecer um valor maior que o máximo do treino?**
O min-max gera um valor > 1, e a regressão extrapola a reta. Funciona, mas, como diz o slide, a
regressão é mais confiável dentro do intervalo de treino.

**Resultado da base de treino:** MSE esperado ≈ 7241 (RMSE ≈ 84,5), R² ≈ 0,42. A série tem bastante
ruído, então um modelo linear explica cerca de 40% da variação. Isso é esperado para esse tipo de dado.
