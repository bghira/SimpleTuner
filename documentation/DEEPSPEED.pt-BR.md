# Offload do DeepSpeed / treinamento multi-GPU

O SimpleTuner v0.7 introduziu suporte preliminar para treinar SDXL usando os estagios 1 a 3 do DeepSpeed ZeRO.
Na v3.0, esse suporte foi muito melhorado, com um construtor de configuracao na WebUI, suporte melhorado a otimizadores e gerenciamento de offload aprimorado.

> ⚠️ DeepSpeed nao esta disponivel em macOS (MPS) ou sistemas ROCm.

**Treinando SDXL 1.0 com 9237MiB de VRAM**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:08:00.0 Off |                  Off |
|  0%   43C    P2   100W / 450W |   9237MiB / 24564MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     11500      C   ...uner/.venv/bin/python3.13     9232MiB |
+-----------------------------------------------------------------------------+
```

Essas economias de memoria foram obtidas com o uso do DeepSpeed ZeRO Stage 2 offload. Sem isso, o U-net do SDXL consumira mais de 24G de VRAM, causando a temida excecao CUDA Out of Memory.

## O que e DeepSpeed?

ZeRO significa **Zero Redundancy Optimizer**. Essa tecnica reduz o consumo de memoria de cada GPU ao particionar os diversos estados de treinamento do modelo (pesos, gradientes e estados do otimizador) entre os dispositivos disponiveis (GPUs e CPUs).

O ZeRO e implementado como estagios incrementais de otimizacoes, onde as otimizacoes dos estagios anteriores estao disponiveis nos estagios posteriores. Para um mergulho profundo no ZeRO, veja o [paper](https://arxiv.org/abs/1910.02054v3) original (1910.02054v3).

## Problemas conhecidos

### Suporte a LoRA

Devido a forma como o DeepSpeed altera as rotinas de salvamento do modelo, atualmente nao e suportado treinar LoRA com DeepSpeed.

Isso pode mudar em uma versao futura.

### Habilitar / desabilitar DeepSpeed em checkpoints existentes

Atualmente no SimpleTuner, o DeepSpeed nao pode ser **habilitado** ao retomar de um checkpoint que **nao** usava DeepSpeed anteriormente.

Da mesma forma, o DeepSpeed nao pode ser **desabilitado** ao tentar retomar o treinamento de um checkpoint treinado com DeepSpeed.

Para contornar esse problema, exporte o pipeline de treinamento para um conjunto completo de pesos do modelo antes de tentar habilitar/desabilitar o DeepSpeed em uma sessao de treinamento em andamento.

E improvavel que esse suporte venha a existir, ja que o otimizador do DeepSpeed e muito diferente das escolhas usuais.

## Estagios do DeepSpeed

O DeepSpeed oferece tres niveis de otimizacao para treinar um modelo, com cada aumento trazendo cada vez mais overhead.

Especialmente para treinamento multi-GPU, as transferencias de CPU atualmente nao sao altamente otimizadas dentro do DeepSpeed.

Por causa desse overhead, e recomendado que o nivel **mais baixo** de DeepSpeed que funcione seja o que voce selecione.

### Stage 1

Os estados do otimizador (por exemplo, para o otimizador Adam, pesos de 32 bits e as estimativas de primeiro e segundo momento) sao particionados entre os processos, de modo que cada processo atualize apenas sua particao.

### Stage 2

Os gradientes reduzidos de 32 bits para atualizar os pesos do modelo tambem sao particionados, de modo que cada processo mantenha apenas os gradientes correspondentes a sua parte dos estados do otimizador.

### Stage 3

Os parametros do modelo de 16 bits sao particionados entre os processos. O ZeRO-3 coleta e particiona automaticamente durante os passes forward e backward.

## Habilitando o DeepSpeed

O [tutorial oficial](https://www.deepspeed.ai/tutorials/zero/) e muito bem estruturado e inclui varios cenarios nao descritos aqui.

### Metodo 1: Construtor de configuracao na WebUI (recomendado)

O SimpleTuner agora fornece uma WebUI amigavel para configuracao do DeepSpeed:

1. Navegue ate a WebUI do SimpleTuner
2. Mude para a aba **Hardware** e abra a secao **Accelerate & Distributed**
3. Clique no botao **DeepSpeed Builder** ao lado do campo `DeepSpeed Config (JSON)`
4. Use a interface interativa para:
   - Selecionar o estagio de otimizacao ZeRO (1, 2 ou 3)
   - Configurar opcoes de offload (CPU, NVMe)
   - Escolher otimizadores e schedulers
   - Definir parametros de acumulacao e clipping de gradientes
5. Previsualize a configuracao JSON gerada
6. Salve e aplique a configuracao

O builder mantem a estrutura JSON consistente e troca otimizadores nao suportados por valores seguros quando necessario, ajudando a evitar erros comuns de configuracao.

### Metodo 2: Configuracao manual em JSON

Para usuarios que preferem editar a configuracao diretamente, voce pode adicionar a configuracao do DeepSpeed diretamente no seu arquivo `config.json`:

```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 2,
      "offload_param": {
        "device": "cpu",
        "pin_memory": true
      },
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
      }
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 1e-4,
        "warmup_num_steps": 500
      }
    },
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 2
  }
}
```

**Opcoes-chave de configuracao:**

- `zero_optimization.stage`: Defina como 1, 2 ou 3 para diferentes niveis de otimizacao ZeRO
- `offload_param.device`: Use "cpu" ou "nvme" para offload de parametros
- `offload_optimizer.device`: Use "cpu" ou "nvme" para offload do otimizador
- `optimizer.type`: Escolha entre os otimizadores suportados (AdamW, Adam, Adagrad, Lamb, etc.)
- `gradient_accumulation_steps`: Numero de passos para acumular gradientes

**Exemplo de offload NVMe:**
```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 3,
      "offload_param": {
        "device": "nvme",
        "nvme_path": "/path/to/nvme/storage",
        "buffer_size": 100000000.0,
        "pin_memory": true
      }
    }
  }
}
```

### Metodo 3: Configuracao manual via accelerate config

Para usuarios avancados, o DeepSpeed ainda pode ser habilitado atraves do `accelerate config`:

```
----------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------
What should be your DeepSpeed's ZeRO optimization stage?
1
How many gradient accumulation steps you're passing in your script? [1]: 4
Do you want to use gradient clipping? [yes/NO]:
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:
----------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?bf16
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
```

Isso resulta no seguinte arquivo yaml:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 4
  zero3_init_flag: false
  zero_stage: 1
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Configurando o SimpleTuner

O SimpleTuner nao requer configuracao especial para o uso do DeepSpeed.

Se estiver usando ZeRO stage 2 ou 3 com offload NVMe, `--offload_param_path=/path/to/offload` pode ser fornecido para armazenar os arquivos de offload de parametros/otimizador em uma particao dedicada. Esse armazenamento idealmente deve ser um dispositivo NVMe, mas qualquer armazenamento serve.

### Melhorias recentes (v0.7+)

#### Construtor de configuracao na WebUI
O SimpleTuner agora inclui um construtor completo de configuracao DeepSpeed na WebUI, permitindo:
- Criar configuracoes JSON personalizadas do DeepSpeed por uma interface intuitiva
- Auto-descobrir parametros disponiveis
- Visualizar impacto da configuracao antes de aplicar
- Salvar e reutilizar templates de configuracao

#### Suporte aprimorado de otimizadores
O sistema agora inclui normalizacao e validacao melhoradas de nomes de otimizadores:
- **Otimizadores suportados**: AdamW, Adam, Adagrad, Lamb, OneBitAdam, OneBitLamb, ZeroOneAdam, MuAdam, MuAdamW, MuSGD, Lion, Muon
- **Otimizadores nao suportados** (substituidos automaticamente por AdamW): cpuadam, fusedadam
- Avisos automáticos de fallback quando otimizadores nao suportados sao especificados

#### Gerenciamento de offload melhorado
- **Limpeza automatica**: diretorios de swap do offload DeepSpeed obsoletos sao removidos automaticamente para evitar estados corrompidos ao retomar
- **Suporte NVMe aprimorado**: melhor manuseio de caminhos de offload NVMe com alocacao automatica de buffer size
- **Deteccao de plataforma**: DeepSpeed e desabilitado automaticamente em plataformas incompativeis (macOS/ROCm)

#### Validacao de configuracao
- Normalizacao automatica de nomes de otimizadores e estrutura de configuracao ao aplicar mudancas
- Protecoes de seguranca para selecoes de otimizadores nao suportados e JSON malformado
- Melhor tratamento de erros e logs para troubleshooting

### Otimizador DeepSpeed / scheduler de taxa de aprendizado

O DeepSpeed usa seu proprio scheduler de taxa de aprendizado e, por padrao, uma versao fortemente otimizada do AdamW - porem nao 8bit. Isso parece menos importante para o DeepSpeed, ja que as coisas tendem a ficar mais proximas da CPU.

Se um `scheduler` ou `optimizer` estiver configurado no seu `default_config.yaml`, eles serao usados. Se nenhum `scheduler` ou `optimizer` estiver definido, as opcoes padrao `AdamW` e `WarmUp` serao usadas como otimizador e scheduler, respectivamente.

## Alguns resultados rapidos de teste

Usando uma GPU 4090 24G:

* Agora podemos treinar o U-net completo a 1 megapixel (area de pixel 1024^2) usando apenas **13102MiB de VRAM para batch size 8**
  * Isso operou a 8 segundos por iteracao. Isso significa que 1000 passos de treinamento podem ser feitos em um pouco menos de 2 horas e meia.
  * Como indicado no tutorial do DeepSpeed, pode ser vantajoso tentar ajustar o batch size para um valor menor, de modo que a VRAM disponivel seja usada para parametros e estados do otimizador.
    * No entanto, o SDXL e um modelo relativamente pequeno, e podemos potencialmente evitar algumas das recomendacoes se o impacto de desempenho for aceitavel.
* Com tamanho de imagem **128x128** e batch size 8, o treinamento consome apenas **9237MiB de VRAM**. Esse e um caso de uso potencialmente nichado para treinamento de pixel art, que requer um mapeamento 1:1 com o espaco latente.

Dentro desses parametros, voce encontrara diferentes niveis de sucesso e pode possivelmente ate mesmo encaixar o treinamento completo do u-net em apenas 8GiB de VRAM a 1024x1024 com batch size 1 (nao testado).

Como o SDXL foi treinado por muitos passos em uma grande distribuicao de resolucoes e proporcoes de imagem, voce pode reduzir a area de pixel para 0.75 megapixels, aproximadamente 768x768, e otimizar ainda mais o uso de memoria.

# Suporte a dispositivos AMD

Nao tenho GPUs AMD de consumo ou workstation, no entanto, ha relatos de que a MI50 (agora fora de suporte) e outras placas Instinct de classe superior **funcionam** com DeepSpeed. A AMD mantem um repositorio para sua implementacao.

## Solucao de problemas

### Problemas comuns e solucoes

#### "DeepSpeed crash ao retomar"
**Problema**: O treinamento trava ao retomar de um checkpoint com offload DeepSpeed habilitado.

**Solucao**: O SimpleTuner agora limpa automaticamente diretorios de swap do offload DeepSpeed obsoletos para evitar estados corrompidos ao retomar. Esse problema foi resolvido nas atualizacoes mais recentes.

#### "Erro de otimizador nao suportado"
**Problema**: A configuracao do DeepSpeed contem nomes de otimizadores nao suportados.

**Solucao**: O sistema agora normaliza automaticamente nomes de otimizadores e substitui otimizadores nao suportados (cpuadam, fusedadam) por AdamW. Avisos sao registrados quando ocorrem fallbacks.

#### "DeepSpeed nao disponivel nesta plataforma"
**Problema**: Opcoes do DeepSpeed estao desabilitadas ou indisponiveis.

**Solucao**: O DeepSpeed so e suportado em sistemas CUDA. Ele e desabilitado automaticamente em macOS (MPS) e ROCm. Isso e por design para evitar problemas de compatibilidade.

#### "Problemas de caminho de offload NVMe"
**Problema**: Erros relacionados a configuracao do caminho de offload NVMe.

**Solucao**: Garanta que `--offload_param_path` aponte para um diretorio valido com espaco suficiente. O sistema agora lida automaticamente com alocacao de buffer size e validacao de caminho.

#### "Erros de validacao de configuracao"
**Problema**: Parametros invalidos de configuracao do DeepSpeed.

**Solucao**: Use o construtor de configuracao da WebUI para gerar o JSON; ele normaliza selecoes de otimizadores e estrutura antes de aplicar a configuracao.

### Informacoes de debug

Para solucionar problemas do DeepSpeed, verifique o seguinte:
- Compatibilidade de hardware via a aba Hardware da WebUI (Hardware → Accelerate) ou `nvidia-smi`
- Configuracao do DeepSpeed nos logs de treinamento
- Permissoes do caminho de offload e espaco disponivel
- Logs de deteccao de plataforma

# Treinamento com EMA (Exponential moving average)

Embora o EMA seja uma otima forma de suavizar gradientes e melhorar a capacidade de generalizacao dos pesos resultantes, e um processo muito pesado em memoria.

O EMA mantem uma copia sombra dos parametros do modelo na memoria, essencialmente dobrando a pegada do modelo. No SimpleTuner, o EMA nao passa pelo modulo Accelerator, o que significa que nao e impactado pelo DeepSpeed. Isso significa que a economia de memoria que vimos com o U-net base nao se aplica ao modelo EMA.

No entanto, por padrao, o modelo EMA e mantido na CPU.
