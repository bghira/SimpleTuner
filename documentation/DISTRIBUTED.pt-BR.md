# Treinamento distribuido (multi-node)

Este documento contem notas* sobre como configurar um cluster 4-way 8xH100 para uso com o SimpleTuner.

> *Este guia nao contem instrucoes completas de instalacao de ponta a ponta. Em vez disso, ele serve como consideracoes a tomar ao seguir o documento [INSTALL](INSTALL.md) ou um dos [guias de inicio rapido](QUICKSTART.md).

## Backend de armazenamento

O treinamento multi-node exige, por padrao, o uso de armazenamento compartilhado entre nodes para o `output_dir`


### Exemplo de NFS no Ubuntu

Apenas um exemplo basico de armazenamento que ja te coloca para rodar.

#### No node "master" que vai gravar os checkpoints

**1. Instalar pacotes do servidor NFS**

```bash
sudo apt update
sudo apt install nfs-kernel-server
```

**2. Configurar o export do NFS**

Edite o arquivo de exports do NFS para compartilhar o diretorio:

```bash
sudo nano /etc/exports
```

Adicione a seguinte linha ao final do arquivo (substitua `slave_ip` pelo IP real da sua maquina slave):

```
/home/ubuntu/simpletuner/output slave_ip(rw,sync,no_subtree_check)
```

*Se voce quiser permitir varios slaves ou uma sub-rede inteira, use:*

```
/home/ubuntu/simpletuner/output subnet_ip/24(rw,sync,no_subtree_check)
```

**3. Exportar o diretorio compartilhado**

```bash
sudo exportfs -a
```

**4. Reiniciar o servidor NFS**

```bash
sudo systemctl restart nfs-kernel-server
```

**5. Verificar o status do servidor NFS**

```bash
sudo systemctl status nfs-kernel-server
```

---

#### Nos nodes slave que enviam o otimizador e outros estados

**1. Instalar pacotes do cliente NFS**

```bash
sudo apt update
sudo apt install nfs-common
```

**2. Criar o diretorio de montagem**

Garanta que o diretorio exista (ele ja deve existir conforme sua configuracao):

```bash
sudo mkdir -p /home/ubuntu/simpletuner/output
```

*Nota:* Se o diretorio contiver dados, faca backup, pois a montagem ocultara os conteudos existentes.

**3. Montar o compartilhamento NFS**

Monte o diretorio compartilhado do master no diretorio local do slave (substitua `master_ip` pelo IP do master):

```bash
sudo mount master_ip:/home/ubuntu/simpletuner/output /home/ubuntu/simpletuner/output
```

**4. Verificar a montagem**

Verifique se a montagem foi bem-sucedida:

```bash
mount | grep /home/ubuntu/simpletuner/output
```

**5. Testar acesso de escrita**

Crie um arquivo de teste para garantir que voce tem permissoes de escrita:

```bash
touch /home/ubuntu/simpletuner/output/test_file_from_slave.txt
```

Em seguida, verifique na maquina master se o arquivo aparece em `/home/ubuntu/simpletuner/output`.

**6. Tornar a montagem persistente**

Para garantir que a montagem persista apos reinicios, adicione-a ao arquivo `/etc/fstab`:

```bash
sudo nano /etc/fstab
```

Adicione a seguinte linha ao final:

```
master_ip:/home/ubuntu/simpletuner/output /home/ubuntu/simpletuner/output nfs defaults 0 0
```

---

#### **Consideracoes adicionais:**

- **Permissoes de usuario:** Garanta que o usuario `ubuntu` tenha o mesmo UID e GID em ambas as maquinas para que as permissoes de arquivo sejam consistentes. Voce pode verificar UIDs com `id ubuntu`.

- **Configuracoes de firewall:** Se voce tiver um firewall habilitado, garanta que o trafego NFS esteja liberado. Na maquina master:

  ```bash
  sudo ufw allow from slave_ip to any port nfs
  ```

- **Sincronizar relogios:** E uma boa pratica manter os relogios dos dois sistemas sincronizados, especialmente em setups distribuidos. Use `ntp` ou `systemd-timesyncd`.

- **Testar checkpoints do DeepSpeed:** Rode um pequeno job DeepSpeed para confirmar que os checkpoints estao sendo gravados corretamente no diretorio do master.


## Configuracao do dataloader

Datasets muito grandes podem ser um desafio para gerenciar com eficiencia. O SimpleTuner shard as bases automaticamente em cada node e distribui o pre-processamento por todas as GPUs disponiveis no cluster, enquanto usa filas e threads assincronas para manter o throughput.

### Dimensionamento do dataset para treinamento multi-GPU

Ao treinar em varias GPUs ou nodes, seu dataset precisa conter amostras suficientes para atender o **batch size efetivo**:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**Calculos de exemplo:**

| Configuracao | Calculo | Batch size efetivo |
|--------------|---------|--------------------|
| 1 node, 8 GPUs, batch_size=4, grad_accum=1 | 4 × 8 × 1 | 32 amostras |
| 2 nodes, 16 GPUs, batch_size=8, grad_accum=2 | 8 × 16 × 2 | 256 amostras |
| 4 nodes, 32 GPUs, batch_size=8, grad_accum=1 | 8 × 32 × 1 | 256 amostras |

Cada bucket de proporcao no seu dataset deve conter pelo menos esse numero de amostras (considerando `repeats`) ou o treinamento falhara com uma mensagem de erro detalhada.

#### Solucoes para datasets pequenos

Se seu dataset for menor que o batch size efetivo:

1. **Reduza o batch size** - Reduza `train_batch_size` para diminuir os requisitos de memoria
2. **Reduza o numero de GPUs** - Treine com menos GPUs (embora isso torne o treinamento mais lento)
3. **Aumente repeats** - Defina `repeats` na sua [configuracao de dataloader](DATALOADER.md#repeats)
4. **Habilite oversubscription automatica** - Use `--allow_dataset_oversubscription` para ajustar os repeats automaticamente

A flag `--allow_dataset_oversubscription` (documentada em [OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription)) calcula e aplica automaticamente o minimo de repeats exigidos para sua configuracao, sendo ideal para prototipagem ou experimentos com datasets pequenos.

### Varredura/descoberta lenta de imagens

O backend **discovery** atualmente restringe a coleta de dados dos buckets de proporcao a um unico node. Isso pode levar um tempo **extremamente** longo com datasets muito grandes, pois cada imagem precisa ser lida do armazenamento para recuperar sua geometria.

Para contornar esse problema, o [metadata_backend parquet](DATALOADER.md#parquet-caption-strategy-json-lines-datasets) deve ser usado, permitindo preprocessar seus dados da forma que for mais acessivel. Como descrito na secao linkada, a tabela parquet contem as colunas `filename`, `width`, `height` e `caption` para ajudar a ordenar os dados de forma rapida e eficiente em seus respectivos buckets.


### Espaco de armazenamento

Datasets enormes, especialmente ao usar o codificador de texto T5-XXL, consumirao quantidades enormes de espaco para os dados originais, os embeddings de imagem e os embeddings de texto.

#### Armazenamento em nuvem

Usando um provedor como Cloudflare R2, e possivel gerar datasets enormes com taxas de armazenamento muito baixas.

Veja o [guia de configuracao do dataloader](DATALOADER.md#local-cache-with-cloud-dataset) para um exemplo de como configurar o tipo `aws` no `multidatabackend.json`

- Dados de imagem podem ser armazenados localmente ou via S3
  - Se as imagens estiverem no S3, a velocidade de pre-processamento diminui conforme a largura de banda de rede
  - Se as imagens forem armazenadas localmente, isso nao aproveita o throughput do NVMe durante o **treinamento**
- Embeddings de imagem e texto podem ser armazenados separadamente em armazenamento local ou em nuvem
  - Colocar embeddings no armazenamento em nuvem reduz pouco a taxa de treinamento, pois sao buscados em paralelo

Idealmente, todas as imagens e todos os embeddings sao mantidos em um bucket de armazenamento em nuvem. Isso simplifica muito o risco de problemas durante o pre-processamento e ao retomar o treinamento.

#### Codificacao VAE sob demanda

Para datasets grandes em que armazenar latentes VAE em cache e inviavel por restricoes de armazenamento ou acesso lento a armazenamento compartilhado, voce pode usar `--vae_cache_disable`. Isso desativa o cache de VAE completamente, forçando o VAE a codificar as imagens sob demanda durante o treinamento.

Isso aumenta a carga de computacao da GPU, mas reduz significativamente os requisitos de armazenamento e o I/O de rede para latentes em cache.

#### Preservando caches de varredura do filesystem

Se seus datasets forem tao grandes que a varredura por novas imagens se torna um gargalo, adicionar `preserve_data_backend_cache=true` a cada entrada de configuracao do dataloader impedira que o backend seja escaneado para novas imagens.

**Note** que voce deve usar o tipo de backend de dados `image_embeds` ([mais informacoes aqui](DATALOADER.md#local-cache-with-cloud-dataset)) para permitir que essas listas de cache vivam separadamente caso seu job de pre-processamento seja interrompido. Isso evitará que a **lista de imagens** seja revarrida na inicializacao.

#### Compressao de dados

A compressao de dados deve ser habilitada adicionando o seguinte ao `config.json`:

```json
{
    ...
    "--compress_disk_cache": true,
    ...
}
```

Isso usara gzip embutido para reduzir a quantidade de espaco em disco redundante consumida pelos embeddings de texto e imagem, que sao bem grandes.

## Configurando via 🤗 Accelerate

Ao usar `accelerate config` (`/home/user/.cache/huggingface/accelerate/default_config.yaml`) para implantar o SimpleTuner, essas opcoes terao prioridade sobre o conteudo de `config/config.env`

Um exemplo de default_config para Accelerate que nao inclui DeepSpeed:

```yaml
# this should be updated on EACH node.
machine_rank: 0
# Everything below here is the same on EACH node.
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: NO
enable_cpu_affinity: false
main_process_ip: 10.0.0.100
main_process_port: 8080
main_training_function: main
mixed_precision: bf16
num_machines: 4
num_processes: 32
rdzv_backend: static
same_network: false
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### DeepSpeed

Este documento nao entra em tanto detalhe quanto a [pagina dedicada](DEEPSPEED.md).

Ao otimizar o treinamento com DeepSpeed em multi-node, usar o menor nivel de ZeRO possivel e **essencial**.

Por exemplo, uma GPU NVIDIA de 80G pode treinar Flux com ZeRO nivel 1 offload, minimizando o overhead substancialmente.

Adicione as seguintes linhas

```yaml
# Update this from MULTI_GPU to DEEPSPEED
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 0.01
  zero3_init_flag: false
  zero_stage: 1
```

### Otimizacao do torch compile

Para desempenho extra (com o inconveniente de problemas de compatibilidade), voce pode habilitar o torch compile adicionando as seguintes linhas no yaml de cada node:

```yaml
dynamo_config:
  # Update this from NO to INDUCTOR
  dynamo_backend: INDUCTOR
  dynamo_mode: max-autotune
  dynamo_use_dynamic: false
  dynamo_use_fullgraph: false
```

## Performance esperada

- 4 nodes H100 SXM5 conectados via rede local
- 1TB de memoria por node
- Streaming de cache de treinamento a partir de backend de dados compartilhado compativel com S3 (Cloudflare R2) na mesma regiao
- Batch size de **8** por acelerador, e **nenhum** passo de acumulacao de gradiente
  - O batch size efetivo total e **256**
- Resolucao em 1024px com bucketing de dados habilitado
- **Velocidade**: 15 segundos por passo com dados 1024x1024 ao treinar Flux.1-dev em full-rank (12B)

Batch sizes menores, resolucao menor e habilitar torch compile podem elevar a velocidade para **iteracoes por segundo**:

- Reduza a resolucao para 512px e desabilite o data bucketing (apenas recortes quadrados)
- Troque o DeepSpeed de AdamW para o otimizador Lion fused
- Habilite torch compile com max-autotune
- **Velocidade**: 2 iteracoes por segundo

## Monitoramento de saude da GPU

SimpleTuner inclui monitoramento automatico de saude da GPU para detectar falhas de hardware precocemente, o que e especialmente importante em treinamento distribuido onde uma unica falha de GPU pode desperdicar tempo de computacao e dinheiro em todo o cluster.

### Circuit Breaker de GPU

O **circuit breaker de GPU** esta sempre habilitado e monitora:

- **Erros ECC** - Detecta erros de memoria incorrigiveis (importante para GPUs A100/H100)
- **Temperatura** - Alerta ao se aproximar do limite de desligamento termico
- **Throttling** - Detecta slowdown de hardware por problemas termicos ou de energia
- **Erros CUDA** - Captura erros de runtime durante o treinamento

Quando uma falha de GPU e detectada:

1. Um webhook `gpu.fault` e emitido (se webhooks estiverem configurados)
2. O circuito abre para evitar mais treinamento em hardware com falha
3. O treinamento termina de forma limpa para que orquestradores possam encerrar a instancia

### Configuracao de webhooks

Configure webhooks no seu `config.json` para receber alertas de falha de GPU:

```json
{
  "--webhook_config": "config/webhooks.json"
}
```

Exemplo de `webhooks.json` para alertas do Discord:

```json
{
  "webhook_url": "https://discord.com/api/webhooks/...",
  "webhook_type": "discord"
}
```

### Consideracoes multi-node

Em treinamento multi-node:

- Cada node executa seu proprio monitor de saude de GPU
- Uma falha de GPU em qualquer node disparara um webhook daquele node
- O job de treinamento falhara em todos os nodes devido a falha de comunicacao distribuida
- Orquestradores devem monitorar falhas de qualquer node no cluster

Veja [Infraestrutura de Resiliencia](experimental/cloud/RESILIENCE.md#circuit-breaker-de-gpu) para formato detalhado do payload do webhook e acesso programatico.

## Observacoes do treinamento distribuido

- Cada node deve ter o mesmo numero de aceleradores disponiveis
- Apenas LoRA/LyCORIS pode ser quantizado, entao o treinamento distribuido completo do modelo requer DeepSpeed
- Esta e uma operacao de custo muito alto, e batch sizes grandes podem te desacelerar mais do que voce deseja, exigindo aumentar o numero de GPUs no cluster. Um equilibrio cuidadoso de custos deve ser considerado.
- (DeepSpeed) Validacoes podem precisar ser desabilitadas ao treinar com DeepSpeed ZeRO 3
- (DeepSpeed) O salvamento do modelo acaba criando copias shardadas estranhas ao salvar com ZeRO nivel 3, mas os niveis 1 e 2 funcionam como esperado
- (DeepSpeed) O uso dos otimizadores baseados em CPU do DeepSpeed se torna necessario, pois ele lida com sharding e offload dos estados do otimizador.
