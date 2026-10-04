# Sistema Inteligente de Ingestão Documental, Busca Semântica e Detecção de Anomalias Textuais

## 1. Apresentação Executiva e Técnica do Projeto

Este repositório compreende a implementação de uma plataforma modular de processamento de linguagem natural (PLN), recuperação de informação densa (*Dense Information Retrieval*) e controle estatístico de qualidade textual, desenvolvida no âmbito do projeto acadêmico de Ciência de Dados e Aprendizado de Máquina (Semanas 1 a 4). O sistema foi concebido para operar em regime de soberania de dados e execução 100% local.

A solução consolida um pipeline de ponta a ponta composto por: ingestão e normalização de documentos não estruturados em formatos heterogêneos (`.txt` e `.pdf`); partição contextual mediante janelas deslizantes (*sliding window chunking*) com sobreposição; representação semântica vetorial densa em 384 dimensões no espaço latente via transformadores pré-treinados (`all-MiniLM-L6-v1`); indexação vetorial persistente com ChromaDB; motor de busca por similaridade de cosseno com interface de linha de comando (CLI); e módulos analíticos não supervisionados voltados à auditoria de qualidade heurística e isolamento de anomalias textuais via algoritmo *Isolation Forest*.

---

## 2. Declaração de Domínio e Acesso Aberto (*Open Access*)

### 2.1 Obras Utilizadas no Corpus
O corpus documental do projeto é composto estritamente por três obras literárias integrais, selecionadas para consolidar a base de conhecimento textual e viabilizar a avaliação de busca e detecção de anomalias:

1. **_Nineteen Eighty-Four_ (1984)**  
   * **Autor:** George Orwell (Eric Arthur Blair, 1903–1950).  
   * **Gênero/Temática:** Ficção distópica, controle social e regimes totalitários.  
   * **Arquivos no repositório:** `data/raw/txts/1984.txt` e `data/raw/pdfs/1984.pdf`.  
   * **Papel no sistema:** Documento central para consultas semânticas densas sobre vigilância, coerção institucional e linguagem doutrinária.

2. **_Animal Farm_ (A Revolução dos Bichos)**  
   * **Autor:** George Orwell (Eric Arthur Blair, 1903–1950).  
   * **Gênero/Temática:** Fábula alegórica e sátira política.  
   * **Arquivos no repositório:** `data/raw/txts/AnimalFarm.txt` e `data/raw/pdfs/AnimalFarm.pdf`.  
   * **Papel no sistema:** Análise comparativa intracorpus de retórica orwelliana, hierarquias de poder e fragmentação semântica.

3. **_Dieu D'Amour_**  
   * **Autora:** Edith Wharton (1862–1937).  
   * **Gênero/Temática:** Literatura dramática/poética e ficção de época em língua inglesa.  
   * **Arquivos no repositório:** `data/raw/txts/DieuDAmor.txt` e `data/raw/pdfs/DieuDAmor.pdf`.  
   * **Papel no sistema:** Atua como documento de contraste estilístico, temático e de distribuição léxica, servindo como elemento fundamental para avaliar a sensibilidade dos modelos de agrupamento (*clustering*) e detecção não supervisionada de anomalias (*outliers* semânticos no espaço latente).

### 2.2 Justificativa Jurídica e Conformidade com Direitos Autorais
A seleção das três obras atende com estrito rigor à diretriz ética e acadêmica de utilizar exclusivamente materiais livres de restrições proprietárias de direitos autorais:

* **Obras de George Orwell (falecido em 21 de janeiro de 1950):**  
  Em conformidade com a Convenção de Berna e com a legislação brasileira de direitos autorais (Lei nº 9.610/1998, Artigo 41), bem como pelas diretivas vigentes na União Europeia, o prazo de proteção patrimonial perdura por 70 anos contados a partir de 1º de janeiro do ano subsequente ao do falecimento do autor (*post mortem auctoris* - p.m.a.). Dessa forma, todas as obras originais de George Orwell em língua inglesa ingressaram legitimamente em **Domínio Público em 1º de janeiro de 2021**.

* **Obra de Edith Wharton (falecida em 11 de agosto de 1937):**  
  Com o encerramento do prazo de 70 anos p.m.a., a produção literária original da autora norte-americana ingressou em domínio público em **1º de janeiro de 2008**, estando plenamente liberada para reprodução, processamento e indexação científica há mais de uma década.

### 2.3 Origem e Proveniência dos Dados Brutos
Os arquivos originais foram obtidos a partir de repositórios digitais abertos dedicados à preservação e curadoria bibliográfica de domínio público:
* **Project Gutenberg** (`https://www.gutenberg.org` e `https://gutenberg.net.au`): Transcrições e digitalizações históricas abertas em formato de texto simples (.txt).
* **Standard Ebooks** (`https://standardebooks.org`): Edições digitais livres com revisão tipográfica criteriosa e diagramação preservada em PDF.

---

## 3. Arquitetura do Sistema e Fluxo de Dados

### 3.1 Diagrama Conceitual do Pipeline
A arquitetura do sistema segue uma abordagem modular baseada em responsabilidade única, desacoplando a ingestão do mecanismo de indexação e inferência estatística:

[Entrada de Dados: .txt e .pdf em data/raw/]  
       │  
       ▼  
[Módulo de Ingestão e Parsers: pypdf & io]  
       │  
       ▼  
[Segmentador Contextual: Chunker de 300 palavras com overlap de 50 palavras]  
       │  
       ▼  
[Inferência Local de Embeddings: all-MiniLM-L6-v1 (Sentence Transformers - 384d)]  
       │  
       ▼  
[Banco de Dados Vetorial Persistente: ChromaDB em chroma_db/]  
       │  
       ├─────────────────────────────────────────┐  
       ▼                                         ▼  
[Interface de Busca CLI]                 [Módulo de Qualidade e ML]  
• Consulta em linguagem natural          • Filtragem heurística de ruído  
• Vetorização em tempo real              • Detecção de anomalias (Isolation Forest)  
• Similaridade de cosseno top-N          • Agrupamento e redução t-SNE / KMeans  

### 3.2 Descrição dos Componentes Técnicos Implementados

* **Extratores e Normalizadores Textuais (`src/ingestion/parsers.py`):**  
  Implementa funções dedicadas para extração agnóstica de texto. Para arquivos `.txt`, realiza a leitura contínua com tratamento resiliente de codificação UTF-8 (`errors="ignore"`). Para arquivos `.pdf`, utiliza a biblioteca `pypdf` para percorrer iterativamente as páginas do documento, extrair fluxos textuais limpos e consolidá-los com metadados estruturados de rastreabilidade (nome do arquivo original e extensão).

* **Segmentação com Sobreposição Contextual (`src/ingestion/chunker.py`):**  
  Responsável pela decomposição de textos extensos em blocos discretos com tamanho parametrizado em 300 palavras e sobreposição móvel (*overlap*) de 50 palavras. Essa parametrização assegura que a extensão textual respeite a capacidade nominal do transformador (evitando truncamento arbitrário) e que sentenças situadas nas fronteiras entre chunks consecutivos mantenham coesão semântica e contextual. Cada bloco recebe um identificador determinístico único e metadados contextuais (fonte, formato e índice sequencial).

* **Serviço Local de Embeddings (`src/embeddings/model.py`):**  
  Encapsula o modelo de linguagem pré-treinado `all-MiniLM-L6-v1` por meio da biblioteca `sentence-transformers`. O modelo projeta cada sequência textual em um vetor denso unitário de 384 dimensões no espaço latente. A execução é realizada integralmente no processador local (CPU), dispensando aceleração externa obrigatória ou envio de dados pela rede.

* **Armazenamento Vetorial Persistente (`src/storage/vector_store.py`):**  
  Implementa a classe `LocalVectorStore`, baseada no `ChromaDB` em modo persistente (`PersistentClient`) armazenado no diretório local `chroma_db/`. Gerencia a criação e conexão com coleções de conhecimento, operações de escrita idempotente (*upsert*) em lotes parametrizáveis e recuperação indexada por vizinhos mais próximos estruturada sobre distância de cosseno.

* **Auditoria de Qualidade e Detecção de Outliers (`src/ml/quality.py` e `src/ml/clustering.py`):**  
  Conjunto de métodos de aprendizado de máquina não supervisionado e inspeção estatística. Analisa a distribuição de comprimento dos blocos para segregação de ruído (e.g., índices e cabeçalhos residuais do Gutenberg) e aplica o classificador `IsolationForest` diretamente sobre a matriz de embeddings para assinalar desvios de distribuição semântica, formatação aberrante ou fragmentos em outros idiomas. Inclui ainda pipelines de redução de dimensionalidade (`t-SNE`) e particionamento (`KMeans`) para exploração visual das densidades semânticas.

* **Interface de Linha de Comando (`cli.py`):**  
  Interface utilitária construída sobre `argparse`, provendo comandos operacionais para automatizar a ingestão em lote (`python cli.py ingest`) e a realização de consultas semânticas configuráveis (`python cli.py search "<termo>" -n <quantidade>`).

---

## 4. Stack Tecnológica e Padrões de Conformidade

### 4.1 Ambiente de Linguagem e Dependências Centrais
* **Linguagem:** Python 3.10 ou superior.
* **sentence-transformers:** Orquestração e inferência local do modelo `all-MiniLM-L6-v1`.
* **chromadb:** Banco de dados vetorial embarcado de alto desempenho para armazenamento local de representações densas.
* **pypdf:** Biblioteca nativa para parsing e extração de sequências textuais de documentos PDF.
* **scikit-learn:** Algoritmos de aprendizado de máquina não supervisionado (`IsolationForest`, `KMeans`, `TSNE`).
* **numpy:** Computação científica e manipulação de arrays multidimensionais para cálculo vetorial.
* **pandas:** Estruturação tabular de metadados, contagem de tokens e geração de métricas estatísticas.
* **matplotlib:** Geração de gráficos diagnósticos de dispersão latente e histogramas de frequência textual.
* **pytest:** Framework robusto para testes unitários, asserções e testes de integração de pipeline.

### 4.2 Ferramentas de Engenharia de Software e Qualidade de Código
* **Black:** Formatador estrito de código-fonte Python aderente à especificação PEP 8, configurado para limite de 100 caracteres por linha.
* **Mypy:** Checador estático de tipos, garantindo anotações formais de tipo (`typing`) em todas as assinaturas de funções e classes dos módulos de produção.

---

## 5. Estrutura do Repositório

A organização dos arquivos e módulos do projeto adota uma arquitetura limpa e hierárquica, separando dados brutos, artefatos gerados, lógica de domínio, testes automatizados e cadernos de exploração:

* **data/**: Diretório reservado para armazenamento e separação dos conjuntos de dados.
  * **raw/**: Armazenamento dos arquivos de entrada sem modificações.
    * **pdfs/**: Documentos integrais no formato PDF (e.g., `1984.pdf`, `AnimalFarm.pdf`, `DieuDAmor.pdf`).
    * **txts/**: Documentos integrais no formato de texto simples (e.g., `1984.txt`, `AnimalFarm.txt`, `DieuDAmor.txt`).
  * **processed/**: Artefatos de dados resultantes de etapas intermediárias de curadoria e metadados.
* **chroma_db/**: Banco de dados vetorial embarcado, gerado e persistido automaticamente após a execução da ingestão.
* **notebooks/**: Cadernos de pesquisa analítica, prototipagem metodológica e documentação científica.
  * `01_pipeline_inicial.ipynb`: Condução das Semanas 1 e 2 — ingestão, chunking com overlap, extração de embeddings locais e indexação.
  * `02_controle_qualidade_e_anomalias.ipynb`: Condução das Semanas 3 e 4 — auditoria estatística, detecção de outliers com Isolation Forest e avaliação de relevância na busca.
* **src/**: Pacote principal contendo os módulos de código de produção.
  * `__init__.py`: Inicializador do pacote Python.
  * **ingestion/**:
    * `parsers.py`: Funções para extração resiliente de dados de arquivos `.txt` e `.pdf`.
    * `chunker.py`: Algoritmo de particionamento deslizante com preservação de overlap e geração de metadados.
  * **embeddings/**:
    * `model.py`: Classe de encapsulamento e carregamento local do modelo `all-MiniLM-L6-v1`.
  * **storage/**:
    * `vector_store.py`: Camada de persistência, upsert em lote e busca por similaridade no ChromaDB.
  * **ml/**:
    * `quality.py`: Implementação do detector não supervisionado de anomalias baseado em Isolation Forest.
    * `clustering.py`: Pipeline de redução dimensional t-SNE e agrupamento KMeans para diagnóstico espacial.
* **tests/**: Suíte de testes automatizados.
  * `test_ingestion.py`: Testes unitários para parsers (.txt e .pdf), integridade do chunking e validação dimensional dos embeddings.
* `cli.py`: Ponto de entrada executável via linha de comando para ingestão e consulta.
* `requirements.txt`: Relação declarativa das dependências do ecossistema e bibliotecas de suporte.
* `pyproject.toml`: Metadados de empacotamento, diretrizes de build e configurações do Black e Mypy.
* `README.md`: Documentação técnica canônica do projeto.

---

## 6. Guia de Instalação e Execução

### Passo 1: Obtenção do Código-Fonte
Clone o repositório a partir do controle de versão para o seu ambiente local e acesse o diretório raiz do projeto:  
`git clone <URL_DO_REPOSITORIO>`  
`cd Capstone`

### Passo 2: Criação e Ativação do Ambiente Virtual
Recomenda-se a utilização de um ambiente virtual isolado com Python 3.10 ou superior.

* **Criação do ambiente virtual:**  
  Execute: `python -m venv venv`

* **Ativação no Windows (PowerShell):**  
  Execute: `.\venv\Scripts\Activate.ps1`  
  *Observação:* Se houver bloqueio de execução de scripts pelo sistema operacional, libere a sessão corrente executando: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`

* **Ativação no Linux ou macOS (Bash/Zsh):**  
  Execute: `source venv/bin/activate`

### Passo 3: Instalação das Dependências
Com o ambiente ativado, atualize o gerenciador de pacotes e instale as dependências listadas no arquivo de manifesto:  
`python -m pip install --upgrade pip`  
`pip install -r requirements.txt`

### Passo 4: Execução do Pipeline de Ingestão e Indexação Vetorial
Para carregar os documentos brutos contidos em `data/raw/`, dividi-los em blocos semânticos com sobreposição, computar as representações vetoriais de 384 dimensões e persistir os registros no banco vetorial local ChromaDB, execute:  
`python cli.py ingest`

O processo exibirá no console a contagem de documentos localizados, o total de blocos gerados, o progresso do cálculo de embeddings e a confirmação de persistência no diretório `chroma_db/`.

### Passo 5: Realização de Buscas Semânticas via CLI
Após a conclusão da ingestão, o motor de busca semântica está apto para receber consultas em linguagem natural. A consulta é convertida dinamicamente no espaço vetorial e comparada contra os chunks armazenados por similaridade de cosseno.

* **Consulta padrão (retorna os 3 trechos mais relevantes):**  
  Execute: `python cli.py search "Big Brother and totalitarian control"`

* **Consulta com parametrização de número de resultados (`-n` ou `--num`):**  
  Execute: `python cli.py search "All animals are equal, but some animals are more equal than others" -n 5`

* **Consulta sobre temas econômicos ou sociais:**  
  Execute: `python cli.py search "poverty, dishwashing and living conditions in Paris" -n 2`

O terminal retornará a identificação do documento de origem de cada resultado, o índice relativo do bloco e a prévia textual recuperada.

### Passo 6: Execução dos Testes Automatizados
O projeto conta com uma suíte de testes unitários que valida a leitura de arquivos, a consistência lógica do fatiamento com sobreposição e a dimensionalidade estrita dos tensores de embedding:

* **Execução via pytest:**  
  Execute: `pytest tests/test_ingestion.py -v`

* **Execução alternativa via módulo nativo unittest:**  
  Execute: `python -m unittest tests/test_ingestion.py`

### Passo 7: Verificação de Qualidade e Conformidade de Código
Para auditar a conformidade de formatação e tipagem estática no repositório, utilize:

* **Auditoria de tipos estáticos com Mypy:**  
  Execute: `mypy src/`

* **Auditoria de estilo e formatação com Black:**  
  Execute: `black --check src/ cli.py`
