# Arquitetura da Data Backend Factory

## Visao geral

A Data Backend Factory do SimpleTuner implementa uma arquitetura modular seguindo principios SOLID para manutenibilidade, testabilidade e desempenho.

## Diagrama de arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Factory Registry                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Performance       │  │     Backend         │  │    Configuration    │  │
│  │    Tracking         │  │    Management       │  │     Loading         │  │
│  │                     │  │                     │  │                     │  │
│  │ • Memory usage      │  │ • text_embed_backends│ │ • JSON parsing      │  │
│  │ • Initialization    │  │ • image_embed_backends│ │ • Variable filling  │  │
│  │ • Build times       │  │ • data_backends     │  │ • Dependency sort   │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Configuration Classes                                  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │  BaseBackendConfig  │  │  ImageBackendConfig │  │ TextEmbedBackendConfig│ │
│  │                     │  │                     │  │                     │  │
│  │ • Common fields     │  │ • Image datasets    │  │ • Text embedding    │  │
│  │ • Shared validation │  │ • Video datasets    │  │   datasets          │  │
│  │ • Base defaults     │  │ • Conditioning      │  │ • Simple validation │  │
│  │                     │  │ • Extensive config  │  │                     │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                       │                                      │
│                        ┌─────────────────────┐                             │
│                        │ImageEmbedBackendConfig│                             │
│                        │                     │                             │
│                        │ • Image embedding   │                             │
│                        │   datasets          │                             │
│                        │ • VAE cache config  │                             │
│                        └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Builder Classes                                     │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │  BaseBackendBuilder │  │  LocalBackendBuilder│  │  AwsBackendBuilder  │  │
│  │                     │  │                     │  │                     │  │
│  │ • Metadata creation │  │ • LocalDataBackend  │  │ • S3DataBackend     │  │
│  │ • Cache management  │  │ • File system       │  │ • AWS S3 storage    │  │
│  │ • Common build flow │  │   operations        │  │ • Connection pools  │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐                          │
│  │  CsvBackendBuilder  │  │HuggingfaceBackendBuilder│                        │
│  │                     │  │                     │                          │
│  │ • CSVDataBackend    │  │ • HuggingfaceDatasetsBackend│                    │
│  │ • CSV file handling │  │ • Streaming datasets│                          │
│  │ • URL downloads     │  │ • Filter functions  │                          │
│  └─────────────────────┘  └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Runtime Components                                   │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐                          │
│  │   BatchFetcher      │  │ DataLoader Iterator │                          │
│  │                     │  │                     │                          │
│  │ • Background        │  │ • Multi-backend     │                          │
│  │   prefetching       │  │   sampling          │                          │
│  │ • Queue management  │  │ • Weighted selection│                          │
│  │ • Thread safety     │  │ • Exhaustion handling│                         │
│  └─────────────────────┘  └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Descricao dos componentes

### 1. Factory Registry (`factory.py`)

O orquestrador central que coordena toda a criacao e gerenciamento de backends.

**Responsabilidades-chave:**
- Rastreamento de performance com metricas de memoria e tempo
- Gerenciamento do ciclo de vida do backend
- Processamento e validacao de configuracao
- Coordenacao de backends de embedding de texto/imagem

**Recursos de desempenho:**
- Rastreamento de uso de memoria em tempo real
- Metricas de tempo por estagio
- Monitoramento de pico de memoria
- Logs para analise

### 2. Classes de configuracao (`config/`)

Gerenciamento de configuracao com tipagem usando dataclasses com heranca.

#### BaseBackendConfig (`config/base.py`)
- Base abstrata com campos comuns e validacao
- Validacao compartilhada de resolucao, estrategia de caption e tamanho de imagem
- Fornece template para configuracoes especializadas

#### ImageBackendConfig (`config/image.py`)
- Configuracao para datasets de imagem, video, condicionamento e avaliacao
- Lida com recorte complexo, proporcao e configuracoes especificas do backend
- Validacao abrangente para todos os casos de uso

#### TextEmbedBackendConfig (`config/text_embed.py`)
- Configuracao simplificada para datasets de embedding de texto
- Foco em gerenciamento de cache e conectividade do backend

#### ImageEmbedBackendConfig (`config/image_embed.py`)
- Configuracao para datasets de embedding de imagem VAE
- Lida com diretorios de cache e parametros de processamento

### 3. Classes Builder (`builders/`)

Implementacao do padrao factory para criar instancias de backend.

#### BaseBackendBuilder (`builders/base.py`)
- Builder abstrato com logica compartilhada de criacao de backend de metadados
- Gerenciamento de cache e tratamento de exclusao
- Padrao template method para fluxo consistente de build

#### Builders especializados
- **LocalBackendBuilder**: Cria instancias de `LocalDataBackend` para acesso ao sistema de arquivos
- **AwsBackendBuilder**: Cria instancias de `S3DataBackend` com pool de conexoes
- **CsvBackendBuilder**: Cria instancias de `CSVDataBackend` para datasets baseados em CSV
- **HuggingfaceBackendBuilder**: Cria instancias de `HuggingfaceDatasetsBackend` com suporte a streaming

### 4. Componentes de runtime (`runtime/`)

Componentes otimizados para operacoes em tempo de treinamento.

#### BatchFetcher (`runtime/batch_fetcher.py`)
- Prefetch em segundo plano de batches de treinamento
- Gerenciamento de fila thread-safe
- Tamanho de fila configuravel e monitoramento de performance

#### DataLoader Iterator (`runtime/dataloader_iterator.py`)
- Amostragem ponderada entre multiplos backends
- Tratamento automatico de esgotamento e repeats de dataset
- Suporte a estrategias de amostragem uniforme e auto-ponderacao

## Fluxo de dados

### Fluxo de inicializacao
```
1. Criacao do Factory Registry
   ├── Inicializacao do rastreamento de performance
   ├── Configuracao do registro de componentes
   └── Inicio do rastreamento de memoria

2. Carregamento de configuracao
   ├── Parsing e validacao de JSON
   ├── Substituicao de variaveis
   └── Ordenacao de dependencias

3. Configuracao de backends
   ├── Backends de text embed
   ├── Backends de image embed
   └── Backends principais de dados

4. Setup de runtime
   ├── Criacao de backend de metadados
   ├── Configuracao de dataset e sampler
   └── Inicializacao de cache
```

### Fluxo de treinamento
```
1. Requisicao de batch
   ├── Prefetch do BatchFetcher
   ├── Selecao ponderada de backend
   └── Gerenciamento de fila

2. Carregamento de dados
   ├── Acesso a dados especificos do backend
   ├── Lookup de metadados
   └── Composicao do batch

3. Processamento
   ├── Lookup de embedding de texto
   ├── Acesso ao cache do VAE
   └── Preparacao final do batch
```

## Decisoes de design e justificativa

### 1. Arquitetura modular
**Decisao**: Dividir a factory monolitica em componentes especializados
**Justificativa**:
- Melhora a testabilidade ao isolar responsabilidades
- Permite desenvolvimento independente de funcionalidades
- Reduz a carga cognitiva ao trabalhar em funcionalidades especificas
- Segue o principio da responsabilidade unica

### 2. Classes de configuracao com heranca
**Decisao**: Usar dataclasses com hierarquia de heranca
**Justificativa**:
- Seguranca de tipos e suporte de IDE
- Logica de validacao compartilhada sem duplicacao
- Separacao clara entre diferentes tipos de backend
- Deteccao de erros em tempo de compilacao

### 3. Padrao factory para builders
**Decisao**: Usar o padrao factory para criacao de backend
**Justificativa**:
- Logica de criacao centralizada
- Facil extensao para novos tipos de backend
- Fluxo de criacao consistente em todos os backends
- Separacao entre configuracao e instanciacao

### 4. Design orientado a performance
**Decisao**: Rastreamento e monitoramento de performance embutidos
**Justificativa**:
- Critico para workloads de treinamento em grande escala
- Permite identificar otimizacoes
- Fornece dados para analise de performance
- Capacidade de monitoramento em tempo real

## Melhorias de desempenho alcancadas

### Otimizacao de memoria
- Uso minimo de memoria no pico
- Carregamento preguiçoso de componentes
- Gerenciamento eficiente de cache
- Padroes otimizados de coleta de lixo

### Velocidade de inicializacao
- Inicializacao rapida de backends
- Processamento paralelo de configuracao
- Resolucao de dependencias otimizada
- Minimo de operacoes redundantes

### Manutenibilidade do codigo
- Baixa complexidade de funcao
- Tamanho maximo de funcao: 50 linhas
- Separacao clara de responsabilidades
- Cobertura de tipos

## Abordagem de desenvolvimento

### Fases de implementacao
- **Fase 1**: Design de arquitetura e implementacao de componentes principais
- **Fase 2**: Testes e validacao
- **Fase 3**: Otimizacao e monitoramento de performance
- **Fase 4**: Deploy e manutencao em producao

### Estrategia de testes
- Testes end-to-end com workloads reais
- Benchmarking e monitoramento de performance
- Validacao de casos limite
- Testes de integracao continua

## Metricas de qualidade de codigo

### Organizacao de arquivos
- Tamanho maximo de arquivo: 500 linhas
- Tamanho medio de funcao: 25 linhas
- Limites claros de modulo
- Convencoes de nome consistentes

### Cobertura de tipos
- 100% de type hints em APIs publicas
- Cobertura de docstring
- Conformidade de analise estatica
- Design amigavel para IDE

### Estrategia de testes
- Testes unitarios para cada componente
- Testes de integracao para workflows
- Testes de regressao de performance
- Validacao de casos limite

## Melhorias futuras

### Recursos planejados
1. **Suporte async**: Operacoes de backend nao bloqueantes
2. **Arquitetura de plugins**: Extensoes de backend de terceiros
3. **Cache avancado**: Hierarquia de cache multi-nivel
4. **Dashboard de monitoramento**: Visualizacao de performance em tempo real

### Pontos de extensao
- Novos tipos de backend via registro de builder
- Classes de configuracao customizadas
- Estrategias de amostragem plugaveis
- Metricas de performance customizadas

## Conclusao

A Data Backend Factory fornece uma base para gerenciamento escalavel e maintenable de backends de dados no SimpleTuner. A arquitetura modular permite melhorias futuras enquanto oferece bom desempenho, qualidade de codigo e experiencia de desenvolvimento.

A arquitetura aborda a complexidade de gerenciar multiplas fontes de dados enquanto fornece funcionalidade robusta e padroes claros para extensao e customizacao.
