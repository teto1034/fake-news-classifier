# A assinatura linguística da fake news

**Projeto Final – Ciclo Básico 2026.2 · Insper Data**

Integrantes: Mateus Pratas Oliveira e **[NOME DO(A) INTEGRANTE 2]**

---

## 1. Problema e pergunta

Notícias falsas circulam com rapidez nas redes, e checá-las uma a uma é caro e lento. Se elas tiverem um **jeito próprio de serem escritas**, essas pistas de estilo podem ajudar na triagem automática, antes da checagem humana.

**Pergunta de pesquisa:** *notícias falsas em português têm uma assinatura linguística, isto é, características de escrita que as distinguem das verdadeiras, mesmo quando consideramos o tamanho do texto?*

A pergunta é **explicativa**. Os modelos servem para verificar se as diferenças vistas na exploração continuam de pé quando as variáveis são consideradas juntas e quando o modelo é avaliado em dados que não viu.

### Hipóteses (escritas antes da análise)

| | Hipótese |
|---|---|
| H1 | Notícias falsas são mais curtas. |
| H2 | Notícias falsas têm frases mais curtas e palavras mais simples. |
| H3 | Notícias falsas usam mais pronomes de 1ª/2ª pessoa (tom mais pessoal). |
| H4 | Notícias falsas são mais emotivas (mais adjetivos e advérbios). |
| H5 | Notícias falsas têm menos cuidado editorial (mais erros ortográficos e maiúsculas). |

---

## 2. Dados

**Fonte:** corpus [Fake.br](https://github.com/roneysco/Fake.br-Corpus) (NILC/USP), coletado entre 2016 e 2018.

- **7.200 notícias** em português: 3.600 falsas e 3.600 verdadeiras.
- As notícias vêm em **pares**: cada falsa tem uma verdadeira sobre o mesmo assunto (mesmo número de arquivo, `par_id`).
- Cada notícia tem um arquivo de **metadados** com 25 medidas linguísticas calculadas pelos autores do corpus (nº de palavras, verbos, adjetivos, pronomes, pausalidade, emotividade, diversidade lexical, erros ortográficos etc.).
- **Unidade de análise:** uma notícia. **Variável de interesse:** `rotulo` (1 = falsa, 0 = verdadeira).

A base bruta está em `data/raw/` (documentação original em [`data/raw/README_FakeBr.md`](data/raw/README_FakeBr.md)). A base processada (`data/processed/fakebr_base.csv`) não vai para o repositório: ela é gerada pelo notebook 01.

### Limpeza (notebook 01)

| Etapa | Decisão | Linhas |
|---|---|---|
| Leitura | textos + metadados unidos por (`par_id`, `rotulo`) | 7.200 |
| Duplicados | 1 notícia verdadeira repetida removida | 7.199 |
| Balanceamento | menor texto verdadeiro removido (3.599 x 3.599) | 7.198 |
| Balanceamento por categoria | 1 verdadeira (política) e 1 falsa (tv_celebridades) removidas | **7.196** |

- A base final é **balanceada** (50% falsas) e balanceada também **dentro de cada uma das 6 categorias**. Por isso a categoria não é usada como variável explicativa.
- `n_links` não foi coletada para 1.392 notícias verdadeiras (valor `None` no original). Como a falta depende da classe, essa variável ficou fora do modelo.
- Nenhum arquivo bruto foi editado à mão: tudo está no código.

---

## 3. Métodos

### Análise exploratória (notebook 02)

1. Distribuição do alvo e do tamanho dos textos.
2. Boxplots de cada variável de estilo por classe.
3. **Controle do tamanho do texto:** contagens brutas (verbos, pronomes, maiúsculas...) foram convertidas em **taxas por 100 palavras**. Para a diversidade lexical, comparamos as classes **dentro de faixas de tamanho parecido**.
4. Correlação de Spearman e **VIF** para identificar variáveis redundantes.
5. Teste de **Mann-Whitney** com **correlação rank-biserial** como tamanho de efeito.

### Modelagem (notebook 03)

- **Separação treino/teste por par** (`GroupShuffleSplit`, 80/20) e validação cruzada `GroupKFold` (5 dobras). Assim os dois textos de um par nunca ficam um no treino e outro no teste, o que vazaria informação sobre o assunto.
- **Baseline:** chutar sempre a mesma classe (`DummyClassifier`).
- **Regressão de referência:** regressão logística (`statsmodels`), com coeficientes lidos como razões de chance e também padronizados. Checagem de VIF, calibração e resíduos de deviance.
- **Machine learning:** Random Forest e XGBoost, com hiperparâmetros escolhidos por validação cruzada só no treino.
- Padronização feita dentro de `Pipeline` (ajustada só no treino). O conjunto de teste foi usado uma única vez, para a avaliação final.

**Variáveis do modelo final:** `n_palavras`, `diversidade`, `tam_medio_sentenca`, `tam_medio_palavra`, `pausalidade`, `erros_ortograficos`, `taxa_verbos`, `taxa_pronomes`, `taxa_pron_sing_1_2`, `taxa_pron_plur_1`, `taxa_maiusculas`.
A `emotividade` saiu do modelo por ter VIF > 100 (ela é calculada a partir de adjetivos e advérbios).

---

## 4. Principais resultados

### O tamanho do texto é a diferença mais marcante

| | Mediana de palavras |
|---|---|
| Verdadeiras | 922 |
| Falsas | 157 |

As classes quase não se sobrepõem em tamanho (rank-biserial = −0,96). Várias medidas dependem mecanicamente do tamanho, então **toda a análise precisou separar "estilo" de "tamanho"**.

### Tamanho de efeito das variáveis (EDA)

| Variável | Mediana falsa | Mediana verdadeira | Rank-biserial |
|---|---|---|---|
| n_palavras | 157 | 922,5 | −0,96 |
| diversidade | 0,68 | 0,47 | +0,95 |
| tam_medio_sentenca | 15,0 | 20,6 | −0,62 |
| pausalidade | 2,35 | 2,96 | −0,45 |
| taxa_verbos | 16,8 | 15,0 | +0,30 |
| taxa_maiusculas | 1,59 | 0,96 | +0,26 |

*(valores positivos = maior nas falsas; |r| ≈ 0,1 pequeno, 0,3 médio, ≥ 0,5 grande)*

- **Mesmo comparando textos de tamanho parecido**, as falsas têm **frases cerca de 3 palavras mais curtas** em todas as 8 faixas de tamanho, e **diversidade lexical maior** (+0,02 a +0,05) em todas as faixas.
- Depois da conversão para taxas, as falsas usam **mais maiúsculas** e **mais verbos** por 100 palavras.

### Balanço das hipóteses

| Hipótese | Resultado |
|---|---|
| H1 – mais curtas | **Apoiada** (efeito muito grande) |
| H2 – frases mais curtas / palavras simples | **Apoiada para frases**; o tamanho das palavras quase não difere e não é significativo na logística |
| H3 – mais pronomes de 1ª/2ª pessoa | **Não apoiada**: as falsas usam *menos* pronomes de 1ª/2ª pessoa do singular (rank-biserial −0,20; coeficiente negativo na logística), embora usem mais pronomes no geral |
| H4 – mais emotivas | **Não apoiada**: emotividade, adjetivos e advérbios têm médias parecidas |
| H5 – menos cuidado editorial | **Parcialmente apoiada**: mais maiúsculas nas falsas; os erros ortográficos são raros nas duas classes e o resultado é ambíguo (maiores nas verdadeiras na EDA, coeficiente positivo na logística) |
| Extra – maior diversidade lexical | **Apoiada**, mesmo comparando textos de tamanho parecido |

### Regressão logística (coeficientes padronizados, treino)

Efeito de +1 desvio padrão na log-chance de a notícia ser falsa, mantendo as demais variáveis constantes:

| Variável | Coef. padronizado | p-valor |
|---|---|---|
| n_palavras | −5,54 | < 0,001 |
| diversidade | +2,25 | < 0,001 |
| taxa_pronomes | +1,01 | < 0,001 |
| taxa_maiusculas | +0,78 | < 0,001 |
| erros_ortograficos | +0,37 | 0,006 |
| taxa_pron_sing_1_2 | −0,32 | 0,002 |
| tam_medio_sentenca | −0,30 | 0,002 |
| taxa_verbos | +0,22 | 0,017 |
| taxa_pron_plur_1, tam_medio_palavra, pausalidade | ≈ 0 | > 0,4 |

Todos os VIFs do modelo final ficaram abaixo de 3,5.

### Desempenho no teste (1.440 notícias nunca vistas)

| Modelo | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline | 0,500 | 0,000 | 0,000 | 0,000 | 0,500 |
| Regressão logística | 0,943 | 0,930 | 0,958 | 0,944 | 0,988 |
| Random Forest | 0,949 | 0,941 | 0,957 | 0,949 | 0,988 |
| XGBoost | 0,953 | 0,942 | 0,967 | 0,954 | 0,989 |

- Os três modelos superam o baseline com folga e têm desempenho praticamente igual. Modelos mais flexíveis não acrescentam muito ao que a logística já capta.
- **Só o tamanho do texto já dá AUC de 0,979.** Boa parte da capacidade de separar as classes vem do tamanho.

---

## 5. Conclusão

Os dados sugerem que, **no corpus Fake.br**, notícias falsas e verdadeiras têm estilos de escrita diferentes. A diferença mais forte é o tamanho: as falsas são muito mais curtas. Além do tamanho, as falsas estão associadas a **frases mais curtas, maior diversidade lexical e mais maiúsculas**, e essas diferenças aparecem mesmo comparando textos de tamanho parecido. As hipóteses de maior emotividade (H4) e de tom mais pessoal (H3) não foram confirmadas.

---

## 6. Limitações

- **Tamanho como confundidor:** as classes quase não se sobrepõem em tamanho. A comparação "a tamanho igual" se apoia em poucos textos nas faixas extremas, e o modelo pode estar aprendendo sobretudo "texto curto = falso".
- **Fontes concentradas:** as notícias falsas vêm de poucos sites. O modelo pode estar aprendendo o estilo desses sites (ou do formato "blog"), e não o de fake news em geral.
- **Período e representatividade:** corpus de 2016–2018, predominantemente sobre política. Não dá para generalizar para outras épocas, temas ou formatos (WhatsApp, vídeos, redes sociais).
- **Medidas prontas:** as métricas linguísticas foram calculadas pelos autores do corpus com ferramentas automáticas. Não as recalculamos, e erros dessas ferramentas passam para a análise.
- **Variáveis ausentes:** `n_links` não foi coletada para parte das verdadeiras e ficou de fora. Informações como fonte, autor e data também não entram no modelo de propósito, para que ele dependa só do texto.
- **Associação, não causalidade:** os coeficientes indicam associação entre estilo e veracidade, não que escrever de certa forma "torna" uma notícia falsa.
- Uma extensão natural é repetir a análise com a versão `size_normalized_texts` do corpus, em que os pares têm o mesmo tamanho.

---

## 7. Como reproduzir

Requisitos: Python 3.11+ e Jupyter.

```bash
git clone <url-do-repositorio>
cd fake-news-classifier
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Rode os notebooks **em ordem**, a partir da pasta `notebooks/`:

1. `01_montagem_base.ipynb`: lê `data/raw/` e gera `data/processed/fakebr_base.csv`
2. `02_analise_exploratoria_de_dados.ipynb`: análise exploratória e decisão de modelagem
3. `03_modelagem.ipynb`: baseline, regressão logística, Random Forest e XGBoost

Todos os modelos usam semente fixa (`SEED = 42`).

### Estrutura do repositório

```
fake-news-classifier/
├── data/
│   ├── raw/              # corpus Fake.br original (textos + metadados)
│   └── processed/        # base gerada pelo notebook 01 (não versionada)
├── notebooks/
│   ├── 01_montagem_base.ipynb
│   ├── 02_analise_exploratoria_de_dados.ipynb
│   └── 03_modelagem.ipynb
├── requirements.txt
└── README.md
```

---

## Referências

- Monteiro R.A., Santos R.L.S., Pardo T.A.S., de Almeida T.A., Ruiz E.E.S., Vale O.A. (2018). *Contributions to the Study of Fake News in Portuguese: New Corpus and Automatic Detection Results.* PROPOR 2018, LNCS vol. 11122. Springer.
- Silva R.M., Santos R.L.S., Almeida T.A., Pardo T.A.S. (2020). *Towards Automatically Filtering Fake News in Portuguese.* Expert Systems with Applications, 146, 113199.
