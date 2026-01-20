# Configuração

Para usuários que desejam usar Docker ou outra plataforma de orquestração de contêineres, veja [este documento](DOCKER.md) primeiro.

## Instalação

Para usuários no Windows 10 ou superior, há um guia de instalação baseado em Docker e WSL disponível [neste documento](DOCKER.md).

### Método de instalação via pip

Você pode instalar o SimpleTuner usando pip, o que é recomendado para a maioria dos usuários:

```bash
# for CUDA
pip install 'simpletuner[cuda]'
# for CUDA 13 / Blackwell (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
# for ROCm
pip install 'simpletuner[rocm]'
# for Apple Silicon
pip install 'simpletuner[apple]'
# for CPU-only (not recommended)
pip install 'simpletuner[cpu]'
# for JPEG XL support (optional)
pip install 'simpletuner[jxl]'

# development requirements (optional, only for submitting PRs or running tests)
pip install 'simpletuner[dev]'
```

### Método via repositório Git

Para desenvolvimento local ou testes, você pode clonar o repositório SimpleTuner e configurar o venv do Python:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# if python --version shows 3.11 or 3.12, you may want to upgrade to 3.13.
python3.13 -m venv .venv

source .venv/bin/activate
```

> ℹ️ Você pode usar seu próprio caminho de venv personalizado definindo `export VENV_PATH=/path/to/.venv` no arquivo `config/config.env`.

**Nota:** Estamos instalando a branch `release` aqui; a branch `main` pode conter recursos experimentais que talvez tenham melhores resultados ou menor uso de memória.

Instale o SimpleTuner com detecção automática de plataforma:

```bash
# Basic installation (auto-detects CUDA/ROCm/Apple)
pip install -e .

# With JPEG XL support
pip install -e .[jxl]
```

**Nota:** O setup.py detecta automaticamente sua plataforma (CUDA/ROCm/Apple) e instala as dependências apropriadas.

#### Etapas adicionais para NVIDIA Hopper / Blackwell

Opcionalmente, hardware Hopper (ou mais recente) pode usar o FlashAttention3 para melhorar o desempenho de inferência e treinamento ao usar `torch.compile`.

Você precisará executar a seguinte sequência de comandos no diretório do SimpleTuner, com seu venv ativo:

```bash
git clone https://github.com/Dao-AILab/flash-attention
pushd flash-attention
  pushd hopper
    python setup.py install
  popd
popd
```

> ⚠️ Gerenciar a build do flash_attn é pouco suportado no SimpleTuner no momento. Isso pode quebrar em atualizações, exigindo que você refaça esse procedimento manualmente de tempos em tempos.

#### Etapas adicionais para AMD ROCm

O seguinte deve ser executado para um AMD MI300X ser utilizável:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
  python3 -m pip install --upgrade pip
  python3 -m pip install .
popd
```

> ℹ️ **Padrões de aceleração ROCm**: Quando o SimpleTuner detecta um build do PyTorch com HIP habilitado ele exporta automaticamente `PYTORCH_TUNABLEOP_ENABLED=1` (a menos que você já tenha definido) para que os kernels TunableOp estejam disponíveis. Em dispositivos MI300/gfx94x também definimos `HIPBLASLT_ALLOW_TF32=1` por padrão, habilitando os caminhos TF32 do hipBLASLt sem exigir ajustes manuais de ambiente.

### Todas as plataformas

- 2a. **Opção Um (Recomendada)**: Execute `simpletuner configure`
- 2b. **Opção Dois**: Copie `config/config.json.example` para `config/config.json` e depois preencha os detalhes.

> ⚠️ Para usuários localizados em países onde o Hugging Face Hub não é facilmente acessível, você deve adicionar `HF_ENDPOINT=https://hf-mirror.com` ao seu `~/.bashrc` ou `~/.zshrc` dependendo de qual `$SHELL` seu sistema usa.

#### Treinamento com múltiplas GPUs {#multiple-gpu-training}

SimpleTuner agora inclui **detecção e configuração automática de GPUs** através da WebUI. Ao primeiro carregamento, você será guiado por uma etapa de onboarding que detecta suas GPUs e configura o Accelerate automaticamente.

##### Detecção automática da WebUI (Recomendada)

Quando você iniciar a WebUI pela primeira vez ou usar `simpletuner configure`, encontrará uma etapa de onboarding "Accelerate GPU Defaults" que:

1. **Detecta automaticamente** todas as GPUs disponíveis no seu sistema
2. **Mostra detalhes das GPUs** incluindo nome, memória e IDs de dispositivo
3. **Recomenda configurações ideais** para treinamento multi-GPU
4. **Oferece três modos de configuração:**

   - **Modo Automático** (Recomendado): Usa todas as GPUs detectadas com a contagem de processos ideal
   - **Modo Manual**: Selecione GPUs específicas ou defina uma contagem de processos personalizada
   - **Modo Desativado**: Treinamento apenas com uma GPU

**Como funciona:**
- O sistema detecta seu hardware de GPU via CUDA/ROCm
- Calcula `--num_processes` ideal com base nos dispositivos disponíveis
- Define `CUDA_VISIBLE_DEVICES` automaticamente quando GPUs específicas são selecionadas
- Salva suas preferências para futuras execuções de treinamento

##### Configuração manual

Se não estiver usando a WebUI, você pode controlar a visibilidade das GPUs diretamente no seu `config.json`:

```json
{
  "accelerate_visible_devices": [0, 1, 2],
  "num_processes": 3
}
```

Isso restringirá o treinamento às GPUs 0, 1 e 2, iniciando 3 processos.

3. Se você estiver usando `--report_to='wandb'` (o padrão), o seguinte ajudará você a reportar suas estatísticas:

```bash
wandb login
```

Siga as instruções exibidas para localizar sua chave de API e configurá-la.

Depois disso, todas as suas sessões de treinamento e dados de validação estarão disponíveis no Weights & Biases.

> ℹ️ Se você quiser desativar os relatórios do Weights & Biases ou do Tensorboard, use `--report-to=none`

4. Inicie o treinamento com simpletuner; os logs serão gravados em `debug.log`

```bash
simpletuner train
```

> ⚠️ Neste ponto, se você usou `simpletuner configure`, está pronto! Caso contrário, esses comandos funcionarão, mas é necessária configuração adicional. Veja [o tutorial](TUTORIAL.md) para mais informações.

### Executar testes unitários

Para executar testes unitários e garantir que a instalação foi concluída com sucesso:

```bash
python -m unittest discover tests/
```

## Avançado: múltiplos ambientes de configuração

Para usuários que treinam vários modelos ou precisam alternar rapidamente entre diferentes conjuntos de dados ou configurações, duas variáveis de ambiente são verificadas na inicialização.

Para usá-las:

```bash
simpletuner train env=default config_backend=env
```

- `env` terá como padrão `default`, que aponta para o diretório típico `SimpleTuner/config/` que este guia ajudou você a configurar
  - Usar `simpletuner train env=pixart` usará o diretório `SimpleTuner/config/pixart` para encontrar `config.env`
- `config_backend` terá como padrão `env`, que usa o arquivo `config.env` típico que este guia ajudou você a configurar
  - Opções suportadas: `env`, `json`, `toml` ou `cmd` se você depende de executar `train.py` manualmente
  - Usar `simpletuner train config_backend=json` procurará por `SimpleTuner/config/config.json` em vez de `config.env`
  - Da mesma forma, `config_backend=toml` usará `config.env`

Você pode criar `config/config.env` que contém um ou ambos estes valores:

```bash
ENV=default
CONFIG_BACKEND=json
```

Eles serão lembrados nas execuções subsequentes. Note que estes podem ser adicionados além das opções de multiGPU descritas [acima](#multiple-gpu-training).

## Dados de Treinamento

Um conjunto de dados publicamente disponível está disponível [no Hugging Face Hub](https://huggingface.co/datasets/bghira/pseudo-camera-10k) com aproximadamente 10k imagens com legendas como nomes de arquivo, prontas para uso com SimpleTuner.

Você pode organizar as imagens em uma única pasta ou organizá-las de forma ordenada em subdiretórios.

### Diretrizes de seleção de imagens

**Requisitos de qualidade:**
- Sem artefatos de JPEG ou imagens borradas - modelos modernos detectam isso
- Evite ruído granulado de sensor CMOS (aparecerá em todas as imagens geradas)
- Sem marcas d'água, selos ou assinaturas (serão aprendidos)
- Quadros de filme geralmente não funcionam devido à compressão (use stills de produção)

**Especificações técnicas:**
- Imagens idealmente divisíveis por 64 (permite reutilização sem redimensionar)
- Misture imagens quadradas e não quadradas para capacidades equilibradas
- Use conjuntos de dados variados e de alta qualidade para melhores resultados

### Geração de legendas

O SimpleTuner fornece [scripts de legendagem](/scripts/toolkit/README.md) para renomear arquivos em massa. Formatos de legenda suportados:
- Nome do arquivo como legenda (padrão)
- Arquivos de texto com `--caption_strategy=textfile`
- JSONL, CSV ou arquivos de metadados avançados

**Ferramentas de legendagem recomendadas:**
- **InternVL2**: Melhor qualidade, mas lento (conjuntos pequenos)
- **BLIP3**: Melhor opção leve com boa aderência às instruções
- **Florence2**: Mais rápido, mas alguns não gostam dos resultados

### Tamanho do lote de treinamento

Seu tamanho máximo de lote depende de VRAM e resolução:
```
vram use = batch size * resolution + base_requirements
```

**Princípios-chave:**
- Use o maior tamanho de lote possível sem problemas de VRAM
- Maior resolução = mais VRAM = menor tamanho de lote
- Se o tamanho de lote 1 a 128x128 não funcionar, o hardware é insuficiente

#### Requisitos de dataset para múltiplas GPUs

Ao treinar com múltiplas GPUs, seu dataset precisa ser grande o suficiente para o **tamanho de lote efetivo**:
```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**Exemplo:** Com 4 GPUs e `train_batch_size=4`, você precisa de pelo menos 16 amostras por bucket de proporção.

**Soluções para conjuntos de dados pequenos:**
- Use `--allow_dataset_oversubscription` para ajustar repetições automaticamente
- Defina `repeats` manualmente na configuração do dataloader
- Reduza o tamanho de lote ou a contagem de GPUs

Veja [DATALOADER.md](DATALOADER.md#multi-gpu-training-and-dataset-sizing) para detalhes completos.

## Publicando no Hugging Face Hub

Para enviar modelos ao Hub automaticamente ao finalizar, adicione ao `config/config.json`:

```json
{
  "push_to_hub": true,
  "hub_model_name": "your-model-name"
}
```

Faça login antes do treinamento:
```bash
huggingface-cli login
```

## Depuração

Ative logs detalhados adicionando ao `config/config.env`:

```bash
export SIMPLETUNER_LOG_LEVEL=DEBUG
export SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG
```

Um arquivo `debug.log` será criado na raiz do projeto com todas as entradas de log.
