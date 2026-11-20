# Arquitetura da fábrica de backends de dados

## Construção do conjunto de dados

A fábrica valida identificadores, expande conjuntos gerados e coloca as fontes de metadados antes dos seus dependentes. Os construtores abrem os provedores de origem e de metadados. O `DatasetRuntime` vincula os provedores, a configuração resolvida, os conjuntos relacionados e os caches de artefatos. O `StateTracker` registra os objetos pelo ID do conjunto.

```text
FactoryRegistry → builders → DatasetRuntime
                              ├─ BaseDataBackend
                              ├─ MetadataBackend
                              ├─ TextEmbeddingCache / VAECache
                              └─ training dataloader
```

## Contratos dos provedores

O provedor de origem lê arquivos ou objetos. O provedor de metadados publica a seleção aceita e a geometria de corte. `describe()` informa contagens e revisões disponíveis, `read_page()` retorna uma página limitada e `read_samples()` lê os campos solicitados na ordem das referências, com duplicatas. As referências seguem a identificação do provedor.

Use `iter_preparation_batches()` e `iter_preparation_text_records()` no pré-processamento. Os provedores dividem as amostras aceitas entre workers de forma independente das repetições e do preenchimento do treino. Os artefatos podem usar um armazenamento separado. A codificação online de texto usa o lote atual.

A descoberta Local, AWS e CSV compartilham os metadados e o agendamento legados. Hugging Face pode particionar transformações preparadas por shard Arrow. Parquet nativo no Webshart usa uma visão indexada, transformações por shard para cortes compatíveis e um cursor de época que pode ser reconstruído.

## Treinamento e retomada

O trainer solicita os lotes agendados ao runtime. O seletor escolhe entre os conjuntos e o prefetch pode antecipar lotes. O trainer confirma a continuação depois de consumir um lote. O checkpoint grava essa posição e o estado do seletor. A retomada nativa exige a mesma visão aceita, opções de amostragem e configuração distribuída.

## Verificação

Para um provedor novo, teste no runtime a estabilidade das referências, a projeção ordenada, a cobertura da preparação, o cancelamento e a reabertura. Se houver agendamento nativo, verifique a reprodução do cursor. Publique uma revisão enquanto outro leitor usa a anterior. Para artefatos novos, verifique que o preparo e o treino usam as mesmas chaves.

```bash
.venv/bin/python -m unittest -v -f tests.test_dataset_runtime tests.test_webshart_parquet_backend tests.test_vae_cache_traversal
```

Em um checkout, use o executor de integração da fábrica para testar os provedores. O benchmark de inicialização local cria JPEGs reais em uma área temporária e mede o preparo inicial e a reabertura no mesmo processo. Execute-o em cada checkout para comparar revisões. A verificação do catálogo Webshart mede leituras projetadas da origem e informa a versão instalada do leitor e as operações de transporte. O treino e a retomada usam o verificador do trainer.

```bash
.venv/bin/python tests/run_factory_tests.py
.venv/bin/python -m scripts.benchmark_dataset_startup --samples 128
.venv/bin/python -m scripts.benchmark_webshart_catalogue --source ./parquet-fixture --shards 3
```
