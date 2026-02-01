# Opcoes do script de treinamento SimpleTuner

## Visao geral

Este guia fornece uma explicacao amigavel das opcoes de linha de comando disponiveis no script `train.py` do SimpleTuner. Essas opcoes oferecem alto grau de customizacao, permitindo treinar seu modelo conforme suas necessidades.

### Formato do arquivo de configuracao JSON

O nome esperado do arquivo JSON e `config.json` e os nomes das chaves sao os mesmos dos `--argumentos` abaixo. O prefixo `--` nao e necessario no arquivo JSON, mas pode ser mantido.

Procurando exemplos prontos? Veja os presets selecionados em [simpletuner/examples/README.md](/simpletuner/examples/README.md).

### Script de configuracao facil (***RECOMENDADO***)

O comando `simpletuner configure` pode ser usado para configurar um arquivo `config.json` com valores padrao quase ideais.

#### Modificando configuracoes existentes

O comando `configure` aceita um unico argumento, um `config.json` compativel, permitindo modificacao interativa do setup de treinamento:

```bash
simpletuner configure config/foo/config.json
```

Onde `foo` e seu ambiente de config — ou use `config/config.json` se nao estiver usando ambientes.

<img width="1484" height="560" alt="image" src="https://github.com/user-attachments/assets/67dec8d8-3e41-42df-96e6-f95892d2814c" />

> ⚠️ Para usuarios em paises onde o Hugging Face Hub nao esta facilmente acessivel, adicione `HF_ENDPOINT=https://hf-mirror.com` ao seu `~/.bashrc` ou `~/.zshrc` conforme o `$SHELL` do seu sistema.

---

## 🌟 Configuracao central do modelo

### `--model_type`

- **O que**: Seleciona se sera criado uma LoRA ou fine-tune completo.
- **Opcoes**: lora, full.
- **Padrao**: lora
  - Se lora for usado, `--lora_type` dita se PEFT ou LyCORIS estao em uso. Alguns modelos (PixArt) funcionam apenas com adaptadores LyCORIS.

### `--model_family`

- **O que**: Determina qual arquitetura de modelo sera treinada.
- **Opcoes**: pixart_sigma, flux, sd3, sdxl, kolors, legacy

### `--lora_format`

- **O que**: Seleciona o formato de chaves do checkpoint LoRA para load/save.
- **Opcoes**: `diffusers` (padrao), `comfyui`
- **Notas**:
  - `diffusers` e o layout padrao PEFT/Diffusers.
  - `comfyui` converte para/de chaves estilo ComfyUI (`diffusion_model.*` com tensores `lora_A/lora_B` e `.alpha`). Flux, Flux2, Lumina2 e Z-Image auto-detectam entradas ComfyUI mesmo se isso ficar em `diffusers`, mas defina `comfyui` para forcar saida ComfyUI ao salvar.

### `--fuse_qkv_projections`

- **O que**: Faz fusao das projecoes QKV nos blocos de atencao do modelo para usar hardware de forma mais eficiente.
- **Nota**: Disponivel apenas com NVIDIA H100 ou H200 com Flash Attention 3 instalado manualmente.

### `--offload_during_startup`

- **O que**: Faz offload dos pesos do text encoder para CPU enquanto o cache de VAE esta sendo criado.
- **Por que**: Isso e util para modelos grandes como HiDream e Wan 2.1, que podem dar OOM ao carregar o cache de VAE. Essa opcao nao afeta a qualidade do treinamento, mas para text encoders muito grandes ou CPUs lentas pode aumentar bastante o tempo de inicializacao com muitos datasets. Por isso, fica desabilitada por padrao.
- **Dica**: Complementa o recurso de offload em grupo abaixo para sistemas com memoria muito restrita.

### `--offload_during_save`

- **O que**: Move temporariamente todo o pipeline para a CPU enquanto `save_hooks.py` prepara checkpoints para que pesos FP8/quantizados sejam gravados fora do device.
- **Por que**: Salvar pesos fp8-quanto pode estourar o uso de VRAM (por exemplo, durante a serializacao de `state_dict()`). Esta opcao mantem o modelo no acelerador durante o treino, mas faz offload breve quando um save e disparado para evitar OOM CUDA.
- **Dica**: Habilite apenas quando o salvamento falhar por OOM; o loader move o modelo de volta e o treinamento continua sem interrupcao.

### `--delete_model_after_load`

- **O que**: Deleta arquivos do modelo do cache HuggingFace apos serem carregados na memoria.
- **Por que**: Reduz uso de disco em setups com orcamento que cobram por GB usado. Depois que os modelos sao carregados na VRAM/RAM, o cache em disco nao e mais necessario ate a proxima execucao. Isso troca storage por banda de rede em execucoes futuras.
- **Notas**:
  - O VAE **nao** e deletado se a validacao estiver habilitada, pois e necessario para gerar imagens de validacao.
  - Text encoders sao deletados apos o data backend factory completar o startup (apos o cache de embeddings).
  - Modelos Transformer/UNet sao deletados imediatamente apos o load.
  - Em setups multi-node, apenas local-rank 0 em cada node executa a delecao. Falhas sao ignoradas silenciosamente para lidar com race conditions em storage compartilhado.
  - Isso **nao** afeta checkpoints salvos — apenas o cache do modelo base pre-treinado.

### `--enable_group_offload`

- **O que**: Habilita offload de modulos em grupo do diffusers para que blocos do modelo possam ser estagiados na CPU (ou disco) entre forward passes.
- **Por que**: Reduz drasticamente o pico de VRAM em transformers grandes (Flux, Wan, Auraflow, LTXVideo, Cosmos2Image) com impacto minimo de performance quando usado com streams CUDA.
- **Notas**:
  - Mutuamente exclusivo com `--enable_model_cpu_offload` — escolha uma estrategia por execucao.
  - Requer diffusers **v0.33.0** ou superior.

### `--group_offload_type`

- **Opcoes**: `block_level` (padrao), `leaf_level`
- **O que**: Controla como as camadas sao agrupadas. `block_level` equilibra economia de VRAM com throughput, enquanto `leaf_level` maximiza a economia ao custo de mais transferencias CPU.

### `--group_offload_blocks_per_group`

- **O que**: Ao usar `block_level`, o numero de blocos transformer agrupados em um grupo de offload.
- **Padrao**: `1`
- **Por que**: Aumentar esse numero reduz a frequencia de transferencias (mais rapido) mas mantem mais parametros residentes no acelerador (mais VRAM).

### `--group_offload_use_stream`

- **O que**: Usa uma stream CUDA dedicada para sobrepor transferencias host/device com compute.
- **Padrao**: `False`
- **Notas**:
  - Faz fallback automatico para transferencias estilo CPU em backends nao-CUDA (Apple MPS, ROCm, CPU).
  - Recomendado ao treinar em GPUs NVIDIA com capacidade de copy engine sobrando.

### `--group_offload_to_disk_path`

- **O que**: Caminho de diretorio usado para despejar parametros em disco em vez de RAM.
- **Por que**: Util para orcamentos de RAM muito apertados (ex.: workstation com NVMe grande).
- **Dica**: Use um SSD local rapido; filesystems de rede desaceleram bastante o treino.

### `--musubi_blocks_to_swap`

- **O que**: Musubi block swap para LongCat-Video, Wan, LTXVideo, Kandinsky5-Video, Qwen-Image, Flux, Flux.2, Cosmos2Image e HunyuanVideo — mantem os ultimos N blocos transformer na CPU e faz streaming de pesos por bloco durante o forward.
- **Padrao**: `0` (desabilitado)
- **Notas**: Offload de pesos estilo Musubi; reduz VRAM com custo de throughput e e ignorado quando gradientes estao habilitados.

### `--musubi_block_swap_device`

- **O que**: String de dispositivo para armazenar blocos transformer trocados (ex.: `cpu`, `cuda:0`).
- **Padrao**: `cpu`
- **Notas**: Usado apenas quando `--musubi_blocks_to_swap` > 0.

### `--ramtorch`

- **O que**: Substitui camadas `nn.Linear` por implementacoes RamTorch com streaming de CPU.
- **Por que**: Compartilha pesos Linear na memoria da CPU e faz streaming para o acelerador para reduzir pressao de VRAM.
- **Notas**:
  - Requer CUDA ou ROCm (nao suportado no Apple/MPS).
  - Mutuamente exclusivo com `--enable_group_offload`.
  - Habilita automaticamente `--set_grads_to_none`.

### `--ramtorch_target_modules`

- **O que**: Padroes glob separados por virgula limitando quais modulos Linear sao convertidos para RamTorch.
- **Padrao**: Todas as camadas Linear sao convertidas quando nenhum padrao e fornecido.
- **Notas**: Casa nomes de modulo totalmente qualificados ou nomes de classe (wildcards permitidos).

### `--ramtorch_text_encoder`

- **O que**: Aplica substituicoes RamTorch a todas as camadas Linear do text encoder.
- **Padrao**: `False`

### `--ramtorch_vae`

- **O que**: Conversao RamTorch experimental apenas para as camadas Linear do mid-block do VAE.
- **Padrao**: `False`
- **Notas**: Blocos conv de up/down do VAE ficam inalterados.

### `--ramtorch_controlnet`

- **O que**: Aplica substituicoes RamTorch a camadas Linear do ControlNet ao treinar um ControlNet.
- **Padrao**: `False`

### `--ramtorch_transformer_percent`

- **O que**: Porcentagem (0-100) de camadas Linear do transformer a serem descarregadas com RamTorch.
- **Padrao**: `100` (todas as camadas elegiveis)
- **Por que**: Permite descarregamento parcial para equilibrar economia de VRAM com desempenho. Valores mais baixos mantem mais camadas na GPU para treinamento mais rapido, enquanto ainda reduz o uso de memoria.
- **Notas**: As camadas sao selecionadas desde o inicio da ordem de travessia do modulo. Pode ser combinado com `--ramtorch_target_modules`.

### `--ramtorch_text_encoder_percent`

- **O que**: Porcentagem (0-100) de camadas Linear do codificador de texto a serem descarregadas com RamTorch.
- **Padrao**: `100` (todas as camadas elegiveis)
- **Por que**: Permite descarregamento parcial de codificadores de texto quando `--ramtorch_text_encoder` esta habilitado.
- **Notas**: Aplica-se apenas quando `--ramtorch_text_encoder` esta habilitado.

### `--ramtorch_disable_sync_hooks`

- **O que**: Desativa os hooks de sincronizacao CUDA adicionados apos as camadas RamTorch.
- **Padrao**: `False` (hooks de sincronizacao habilitados)
- **Por que**: Os hooks de sincronizacao corrigem condicoes de corrida no sistema de buffering ping-pong do RamTorch que podem causar saidas nao deterministicas. Desativar pode melhorar o desempenho, mas arrisca resultados incorretos.
- **Notas**: Desative apenas se tiver problemas com os hooks de sincronizacao ou quiser testar sem eles.

### `--ramtorch_disable_extensions`

- **O que**: Aplica RamTorch apenas a camadas Linear, pula Embedding/RMSNorm/LayerNorm/Conv.
- **Padrao**: `True` (extensoes desabilitadas)
- **Por que**: O SimpleTuner estende o RamTorch alem das camadas Linear para incluir camadas Embedding, RMSNorm, LayerNorm e Conv. Use isso para desativar essas extensoes e descarregar apenas camadas Linear.
- **Notas**: Pode reduzir a economia de VRAM, mas pode ajudar a depurar problemas com os tipos de camadas estendidas.

### `--pretrained_model_name_or_path`

- **O que**: Caminho para o modelo pre-treinado ou seu identificador em <https://huggingface.co/models>.
- **Por que**: Para especificar o modelo base a partir do qual iniciar o treino. Use `--revision` e `--variant` para especificar versoes especificas de um repositorio. Isso tambem suporta caminhos `.safetensors` de arquivo unico para SDXL, Flux e SD3.x.

### `--pretrained_t5_model_name_or_path`

- **O que**: Caminho para o modelo T5 pre-treinado ou seu identificador em <https://huggingface.co/models>.
- **Por que**: Ao treinar PixArt, voce pode querer usar uma fonte especifica para seus pesos T5 para evitar baixar varias vezes ao trocar o modelo base de treinamento.

### `--pretrained_gemma_model_name_or_path`

- **O que**: Caminho para o modelo Gemma pre-treinado ou seu identificador em <https://huggingface.co/models>.
- **Por que**: Ao treinar modelos baseados em Gemma (por exemplo LTX-2, Sana ou Lumina2), voce pode apontar para um checkpoint Gemma compartilhado sem mudar o caminho do modelo base de difusao.

### `--custom_text_encoder_intermediary_layers`

- **O que**: Sobrescreve quais camadas de estado oculto extrair do encoder de texto para modelos FLUX.2.
- **Formato**: Array JSON de indices de camadas, ex: `[10, 20, 30]`
- **Padrao**: Valores padrao especificos do modelo sao usados quando nao definido:
  - FLUX.2-dev (Mistral-3): `[10, 20, 30]`
  - FLUX.2-klein (Qwen3): `[9, 18, 27]`
- **Por que**: Permite experimentacao com diferentes combinacoes de estados ocultos do encoder de texto para alinhamento personalizado ou propositos de pesquisa.
- **Nota**: Esta opcao e experimental e aplica-se apenas a modelos FLUX.2. Alterar indices de camadas invalidara embeddings de texto em cache e requer regeneracao. O numero de camadas deve corresponder a entrada esperada pelo modelo (3 camadas).

### `--gradient_checkpointing`

- **O que**: Durante o treinamento, os gradientes serao calculados por camada e acumulados para economizar VRAM de pico, ao custo de treinos mais lentos.

### `--gradient_checkpointing_interval`

- **O que**: Faz checkpoint apenas a cada *n* blocos, onde *n* e um valor maior que zero. Um valor 1 e efetivamente o mesmo que deixar `--gradient_checkpointing` habilitado, e 2 faz checkpoint a cada outro bloco.
- **Nota**: SDXL e Flux sao atualmente os unicos modelos que suportam essa opcao. SDXL usa uma implementacao meio hack.

### `--gradient_checkpointing_backend`

- **Opcoes**: `torch`, `unsloth`
- **O que**: Seleciona a implementacao para gradient checkpointing.
  - `torch` (padrao): Checkpointing PyTorch padrao que recalcula ativacoes durante o backward pass. ~20% de overhead de tempo.
  - `unsloth`: Descarrega ativacoes para CPU de forma assincrona em vez de recalcular. ~30% mais economia de memoria com apenas ~2% de overhead. Requer banda PCIe rapida.
- **Nota**: So funciona quando `--gradient_checkpointing` esta habilitado. O backend `unsloth` requer CUDA.

### `--refiner_training`

- **O que**: Habilita treinamento de uma serie de modelos mixture-of-experts customizada. Veja [Mixture-of-Experts](MIXTURE_OF_EXPERTS.md) para mais informacoes sobre essas opcoes.

## Precisao

### `--quantize_via`

- **Opcoes**: `cpu`, `accelerator`, `pipeline`
  - Em `accelerator`, pode funcionar um pouco mais rapido com risco de OOM em placas 24G para modelos grandes como Flux.
  - Em `cpu`, a quantizacao leva cerca de 30 segundos. (**Padrao**)
  - `pipeline` delega a quantizacao para o Diffusers usando `--quantization_config` ou presets compatíveis com pipeline (ex.: `nf4-bnb`, `int8-torchao`, `fp8-torchao`, `int8-quanto` ou checkpoints `.gguf`).

### `--base_model_precision`

- **O que**: Reduz a precisao do modelo e treina com menos memoria. Existem tres backends de quantizacao suportados: BitsAndBytes (pipeline), TorchAO (pipeline ou manual) e Optimum Quanto (pipeline ou manual).

#### Presets de pipeline Diffusers

- `nf4-bnb` carrega via Diffusers com config BitsAndBytes NF4 4-bit (apenas CUDA). Requer `bitsandbytes` e um build do diffusers com suporte BnB.
- `int4-torchao`, `int8-torchao` e `fp8-torchao` usam TorchAoConfig via Diffusers (CUDA). Requer `torchao` e diffusers/transformers recentes.
- `int8-quanto`, `int4-quanto`, `int2-quanto`, `fp8-quanto` e `fp8uz-quanto` usam QuantoConfig via Diffusers. O diffusers mapeia FP8-NUZ para pesos float8; use quantizacao manual do quanto se precisar da variante NUZ.
- Checkpoints `.gguf` sao auto-detectados e carregados com `GGUFQuantizationConfig` quando disponivel. Instale diffusers/transformers recentes para suporte GGUF.

#### Optimum Quanto

Fornecido pela Hugging Face, o optimum-quanto tem suporte robusto em todas as plataformas.

- `int8-quanto` e o mais compativel e provavelmente produz os melhores resultados
  - treinamento mais rapido em RTX4090 e provavelmente outras GPUs
  - usa matmul acelerado por hardware em CUDA para int8/int4
    - int4 ainda e bem lento
  - funciona com `TRAINING_DYNAMO_BACKEND=inductor` (`torch.compile()`)
- `fp8uz-quanto` e uma variante fp8 experimental para dispositivos CUDA e ROCm.
  - melhor suportado em silicio AMD como Instinct ou arquiteturas mais novas
  - pode ser um pouco mais rapido que `int8-quanto` em uma 4090 para treino, mas nao para inferencia (1 segundo mais lento)
  - funciona com `TRAINING_DYNAMO_BACKEND=inductor` (`torch.compile()`)
- `fp8-quanto` nao usa (por enquanto) matmul fp8 e nao funciona em Apple.
  - nao ha matmul fp8 por hardware em CUDA ou ROCm, entao pode ser visivelmente mais lento que int8
    - usa kernel MARLIN para GEMM fp8
  - incompatível com dynamo; desabilita automaticamente se a combinacao for tentada.

#### TorchAO

Uma biblioteca mais nova do Pytorch; AO permite substituir linears e convolucoes 2D (ex.: modelos tipo UNet) por equivalentes quantizados.

- `int8-torchao` reduz o consumo de memoria ao mesmo nivel das precisões do Quanto
  - no momento, roda um pouco mais lento (11s/iter) do que o Quanto (9s/iter) no Apple MPS
  - sem `torch.compile`, mesma velocidade e uso de memoria que `int8-quanto` em CUDA, perfil desconhecido em ROCm
  - com `torch.compile`, mais lento que `int8-quanto`
- `fp8-torchao` so esta disponivel para aceleradores Hopper (H100, H200) ou mais novos (Blackwell B200)

##### Otimizadores

O TorchAO inclui otimizadores 4bit e 8bit amplamente disponiveis: `ao-adamw8bit`, `ao-adamw4bit`

Tambem fornece dois otimizadores voltados para usuarios Hopper (H100 ou superior): `ao-adamfp8` e `ao-adamwfp8`

#### SDNQ (SD.Next Quantization Engine)

[SDNQ](https://github.com/disty0/sdnq) e uma biblioteca de quantizacao otimizada para treinamento que funciona em todas as plataformas: AMD (ROCm), Apple (MPS) e NVIDIA (CUDA). Ela fornece treinamento quantizado com arredondamento estocastico e estados de otimizador quantizados para eficiencia de memoria.

##### Niveis de precisao recomendados

**Para fine-tuning completo** (pesos do modelo sao atualizados):
- `uint8-sdnq` - Melhor equilibrio entre economia de memoria e qualidade
- `uint16-sdnq` - Maior precisao para qualidade maxima (ex.: Stable Cascade)
- `int16-sdnq` - Alternativa 16-bit com sinal
- `fp16-sdnq` - FP16 quantizado, maxima precisao com beneficios SDNQ

**Para treinamento LoRA** (pesos base congelados):
- `int8-sdnq` - 8-bit com sinal, boa escolha geral
- `int6-sdnq`, `int5-sdnq` - Menor precisao, menos memoria
- `uint5-sdnq`, `uint4-sdnq`, `uint3-sdnq`, `uint2-sdnq` - Compressao agressiva

**Nota:** `int7-sdnq` esta disponivel mas nao e recomendado (lento e pouco menor que int8).

**Importante:** Abaixo de 5-bit, o SDNQ habilita automaticamente SVD (Singular Value Decomposition) com 8 passos para manter qualidade. SVD leva mais tempo para quantizar e nao e deterministico, por isso Disty0 fornece modelos SVD pre-quantizados no HuggingFace. SVD adiciona overhead de compute durante o treinamento, entao evite para fine-tuning completo onde pesos sao atualizados ativamente.

**Recursos principais:**
- Multi-plataforma: funciona igual em AMD, Apple e NVIDIA
- Otimizado para treinamento: usa arredondamento estocastico para reduzir acumulacao de erro de quantizacao
- Eficiente em memoria: suporta buffers de estado do otimizador quantizados
- Matmul desacoplado: precisao de peso e precisao de matmul sao independentes (INT8/FP8/FP16 disponivel)

##### Otimizadores SDNQ

O SDNQ inclui otimizadores com buffers de estado quantizados opcionais para economia adicional de memoria:

- `sdnq-adamw` - AdamW com buffers de estado quantizados (uint8, group_size=32)
- `sdnq-adamw+no_quant` - AdamW sem estados quantizados (para comparacao)
- `sdnq-adafactor` - Adafactor com buffers quantizados
- `sdnq-came` - CAME com buffers quantizados
- `sdnq-lion` - Lion com buffers quantizados
- `sdnq-muon` - Muon com buffers quantizados
- `sdnq-muon+quantized_matmul` - Muon com matmul INT8 no zeropower

Todos os otimizadores SDNQ usam arredondamento estocastico por padrao e podem ser configurados com `--optimizer_config` para ajustes como `use_quantized_buffers=false` (desabilitar quantizacao de estado).

**Opcoes especificas do Muon:**
- `use_quantized_matmul` - Habilita matmul INT8/FP8/FP16 em zeropower_via_newtonschulz5
- `quantized_matmul_dtype` - Precisao de matmul: `int8` (GPUs consumer), `fp8` (datacenter), `fp16`
- `zeropower_dtype` - Precisao para zeropower (ignorada quando `use_quantized_matmul=True`)
- Prefixe args com `muon_` ou `adamw_` para valores diferentes em Muon vs fallback AdamW

**Modelos pre-quantizados:** Disty0 fornece modelos SVD uint4 pre-quantizados em [huggingface.co/collections/Disty0/sdnq](https://huggingface.co/collections/Disty0/sdnq). Carregue normalmente e depois converta com `convert_sdnq_model_to_training()` apos importar SDNQ (SDNQ deve ser importado antes do load para registrar no Diffusers).

**Nota sobre checkpointing:** Modelos SDNQ de treinamento sao salvos tanto em formato PyTorch nativo (`.pt`) para retomar treino quanto em formato safetensors para inferencia. O formato nativo e necessario para retomar corretamente porque a classe `SDNQTensor` usa serializacao customizada.

**Dica de espaco em disco:** Para economizar disco, voce pode manter apenas os pesos quantizados e usar o script [dequantize_sdnq_training.py](https://github.com/Disty0/sdnq/blob/main/scripts/dequantize_sdnq_training.py) para dequantizar quando necessario para inferencia.

### `--quantization_config`

- **O que**: Objeto JSON ou caminho de arquivo descrevendo overrides de `quantization_config` do Diffusers ao usar `--quantize_via=pipeline`.
- **Como**: Aceita JSON inline (ou arquivo) com entradas por componente. Chaves podem incluir `unet`, `transformer`, `text_encoder` ou `default`.
- **Exemplos**:

```json
{
  "unet": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"},
  "text_encoder": {"quant_type": {"group_size": 128}}
}
```

Este exemplo habilita 4-bit NF4 BnB no UNet e TorchAO int4 no text encoder.

#### Torch Dynamo

Habilite `torch.compile()` na WebUI em **Hardware → Accelerate (advanced)** e defina **Torch Dynamo Backend** para seu compilador preferido (ex.: *inductor*). Toggles adicionais permitem escolher o **mode**, habilitar **dynamic shape** ou **regional compilation** para acelerar cold starts em transformers muito profundos.

A mesma configuracao pode ser expressa em `config/config.env`:

```bash
TRAINING_DYNAMO_BACKEND=inductor
```

Opcionalmente, combine com `--dynamo_mode=max-autotune` ou outras flags Dynamo expostas na UI para controle fino.

Note que os primeiros passos do treinamento serao mais lentos devido a compilacao em background.

Para persistir as configuracoes em `config.json`, adicione as chaves equivalentes:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "max-autotune",
  "dynamo_fullgraph": false,
  "dynamo_dynamic": false,
  "dynamo_use_regional_compilation": true
}
```

Omitir entradas que voce quer herdar dos defaults do Accelerate (por exemplo, deixe `dynamo_mode` ausente para selecao automatica).

### `--attention_mechanism`

Mecanismos de atencao alternativos sao suportados, com diferentes niveis de compatibilidade e trade-offs:

- `diffusers` usa os kernels SDPA nativos do PyTorch e e o padrao.
- `xformers` habilita o kernel de atencao [xformers](https://github.com/facebookresearch/xformers) (treino + inferencia) quando o modelo expõe `enable_xformers_memory_efficient_attention`.
- `flash-attn`, `flash-attn-2`, `flash-attn-3` e `flash-attn-3-varlen` usam o helper `attention_backend` do Diffusers para rotear atencao para kernels FlashAttention v1/2/3. Instale os wheels `flash-attn` / `flash-attn-interface` correspondentes e note que FA3 exige GPUs Hopper.
- `flex` seleciona o backend FlexAttention do PyTorch 2.5 (FP16/BF16 em CUDA). Voce deve compilar/instalar os kernels Flex separadamente — veja [documentation/attention/FLEX.md](attention/FLEX.md).
- `cudnn`, `native-efficient`, `native-flash`, `native-math`, `native-npu` e `native-xla` selecionam o backend SDPA correspondente exposto por `torch.nn.attention.sdpa_kernel`. Esses sao úteis para determinismo (`native-math`), kernel SDPA do CuDNN ou aceleradores nativos (NPU/XLA).
- `sla` habilita [Sparse–Linear Attention (SLA)](https://github.com/thu-ml/SLA), fornecendo um kernel hibrido esparso/linear ajustavel para treino e validacao sem gating adicional.
  - Instale o pacote SLA (por exemplo via `pip install -e ~/src/SLA`) antes de selecionar esse backend.
  - O SimpleTuner salva os pesos de projecao aprendidos do SLA em `sla_attention.pt` dentro de cada checkpoint; mantenha esse arquivo junto ao checkpoint para que resumes e inferencia mantenham o estado treinado.
  - Como o backbone e ajustado em torno do comportamento hibrido do SLA, o SLA sera necessario na inferencia tambem. Veja `documentation/attention/SLA.md`.
  - Use `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'` (JSON ou sintaxe de dict Python) para override de defaults do SLA se precisar experimentar.
- `sageattention`, `sageattention-int8-fp16-triton`, `sageattention-int8-fp16-cuda` e `sageattention-int8-fp8-cuda` envolvem os kernels [SageAttention](https://github.com/thu-ml/SageAttention). Sao orientados a inferencia e devem ser usados com `--sageattention_usage` para evitar treinamento acidental.
  - Em termos simples, SageAttention reduz o custo de compute para inferencia

> ℹ️ Os seletores Flash/Flex/PyTorch dependem do dispatcher `attention_backend` do Diffusers, entao atualmente beneficiam modelos estilo transformer que ja usam esse caminho (Flux, Wan 2.x, LTXVideo, QwenImage, etc.). UNets SD/SDXL classicos ainda usam SDPA do PyTorch diretamente.

Usar `--sageattention_usage` para habilitar treino com SageAttention deve ser feito com cuidado, pois ele nao rastreia nem propaga gradientes das implementacoes CUDA customizadas para linears QKV.

- Isso resulta em camadas totalmente nao treinadas, o que pode causar colapso do modelo ou pequenas melhorias em treinos curtos.

---

## 📰 Publishing

### `--push_to_hub`

- **O que**: Se fornecido, seu modelo sera enviado para o [Huggingface Hub](https://huggingface.co) quando o treinamento terminar. Usar `--push_checkpoints_to_hub` tambem envia cada checkpoint intermediario.

### `--push_to_hub_background`

- **O que**: Faz upload para o Hugging Face Hub em um worker de background para que envios de checkpoint nao pausem o loop de treinamento.
- **Por que**: Mantem treino e validacao rodando enquanto os uploads ocorrem de forma assincrona. Os uploads finais ainda sao aguardados antes do encerramento para que falhas aparecam.

### `--webhook_config`

- **O que**: Configuracao para alvos de webhook (ex.: Discord, endpoints customizados) para receber eventos de treinamento em tempo real.
- **Por que**: Permite monitorar execucoes com ferramentas externas e dashboards, recebendo notificacoes em etapas-chave.
- **Notas**: O campo `job_id` nos payloads de webhook pode ser preenchido definindo a variavel de ambiente `SIMPLETUNER_JOB_ID` antes do treinamento:
  ```bash
  export SIMPLETUNER_JOB_ID="my-training-run-name"
  python train.py
  ```
Isso e util para ferramentas de monitoramento que recebem webhooks de varios treinos para identificar qual config enviou cada evento. Se SIMPLETUNER_JOB_ID nao estiver definido, job_id sera null nos payloads.

### `--publishing_config`

- **O que**: JSON/dict/caminho de arquivo opcional descrevendo destinos de publicacao fora do Hugging Face (S3 compativel, Backblaze B2, Azure Blob Storage, Dropbox).
- **Por que**: Espelha o parsing de `--webhook_config` para que voce envie artefatos alem do Hub. A publicacao roda no processo principal apos a validacao usando o `output_dir` atual.
- **Notas**: Provedores sao aditivos a `--push_to_hub`. Instale os SDKs (ex.: `boto3`, `azure-storage-blob`, `dropbox`) dentro da sua `.venv` quando habilitar. Veja `documentation/publishing/README.md` para exemplos completos.

### `--hub_model_id`

- **O que**: Nome do modelo no Huggingface Hub e diretorio local de resultados.
- **Por que**: Esse valor e usado como nome de diretorio sob `--output_dir`. Se `--push_to_hub` for fornecido, isso vira o nome do modelo no Huggingface Hub.

### `--modelspec_comment`

- **O que**: Texto incorporado nos metadados do arquivo safetensors como `modelspec.comment`
- **Padrao**: None (desabilitado)
- **Notas**:
  - Visivel em visualizadores de modelo externos (ComfyUI, ferramentas de info de modelo)
  - Aceita uma string ou array de strings (unidas com quebras de linha)
  - Suporta placeholders `{env:VAR_NAME}` para substituicao de variaveis de ambiente
  - Cada checkpoint usa o valor de configuracao atual no momento do salvamento

**Exemplo (string)**:
```json
"modelspec_comment": "Treinado no meu dataset customizado v2.1"
```

**Exemplo (array para multiplas linhas)**:
```json
"modelspec_comment": [
  "Execucao de treino: experiment-42",
  "Dataset: custom-portraits-v2",
  "Notas: {env:TRAINING_NOTES}"
]
```

### `--disable_benchmark`

- **O que**: Desabilita a validacao/benchmark inicial que ocorre no passo 0 do modelo base. Essas saidas sao costuradas no lado esquerdo das imagens de validacao do modelo treinado.

## 📂 Armazenamento e gerenciamento de dados

### `--data_backend_config`

- **O que**: Caminho para sua configuracao de dataset do SimpleTuner.
- **Por que**: Multiplos datasets em midias diferentes podem ser combinados em uma unica sessao de treinamento.
- **Exemplo**: Veja [multidatabackend.json.example](/multidatabackend.json.example) para um exemplo de configuracao e [este documento](DATALOADER.md) para mais informacoes sobre o data loader.

### `--override_dataset_config`

- **O que**: Quando fornecido, permite ao SimpleTuner ignorar diferencas entre a configuracao cacheada dentro do dataset e os valores atuais.
- **Por que**: Quando o SimpleTuner roda pela primeira vez em um dataset, ele cria um documento de cache contendo informacoes sobre tudo nesse dataset. Isso inclui a configuracao do dataset, incluindo valores de "crop" e "resolution". Mudar esses valores arbitrariamente ou por acidente pode fazer seus treinos falharem aleatoriamente, entao e altamente recomendado nao usar esse parametro e resolver as diferencas de outra forma.

### `--data_backend_sampling`

- **O que**: Ao usar multiplos data backends, a amostragem pode usar estrategias diferentes.
- **Opcoes**:
  - `uniform` - comportamento anterior do v0.9.8.1 e anteriores, onde o tamanho do dataset nao era considerado, apenas pesos manuais de probabilidade.
  - `auto-weighting` - comportamento padrao onde o tamanho do dataset e usado para amostrar todos igualmente, mantendo amostragem uniforme de toda a distribuicao.
    - Isso e necessario se voce tem datasets de tamanhos diferentes e quer que o modelo aprenda igualmente.
    - Mas ajustar `repeats` manualmente e **obrigatorio** para amostrar imagens Dreambooth contra seu conjunto de regularizacao corretamente.

### `--vae_cache_scan_behaviour`

- **O que**: Configura o comportamento da varredura de integridade.
- **Por que**: Um dataset pode ter configuracoes incorretas aplicadas em varios pontos do treinamento, por exemplo, se voce deletar acidentalmente os arquivos `.json` de cache do dataset e trocar a configuracao do data backend para usar imagens quadradas em vez de aspect-crops. Isso resulta em um cache de dados inconsistente, que pode ser corrigido definindo `scan_for_errors` como `true` no seu `multidatabackend.json`. Quando essa varredura roda, ela usa `--vae_cache_scan_behaviour` para decidir como resolver a inconsistencias: `recreate` (padrao) remove a entrada de cache ofensora para que seja recriada, e `sync` atualiza os metadados do bucket para refletir a realidade do sample de treino. Valor recomendado: `recreate`.

### `--dataloader_prefetch`

- **O que**: Busca lotes com antecedencia.
- **Por que**: Especialmente com batch sizes grandes, o treino vai "pausar" enquanto samples sao lidos do disco (mesmo NVMe), impactando metricas de utilizacao da GPU. Habilitar prefetch mantem um buffer cheio de lotes completos para que possam ser carregados instantaneamente.

> ⚠️ Isso e relevante apenas para H100 ou superior em baixa resolucao onde I/O vira o gargalo. Para a maioria dos casos, e uma complexidade desnecessaria.

### `--dataloader_prefetch_qlen`

- **O que**: Aumenta ou reduz o numero de lotes mantidos em memoria.
- **Por que**: Quando prefetch esta habilitado, o padrao e manter 10 entradas por GPU/processo em memoria. Esse valor pode ser ajustado para preparar mais ou menos lotes.

### `--compress_disk_cache`

- **O que**: Compacta em disco os caches de VAE e embeddings de texto.
- **Por que**: O encoder T5 usado por DeepFloyd, SD3 e PixArt gera embeddings de texto muito grandes que ficam quase vazios para legendas curtas ou redundantes. Habilitar `--compress_disk_cache` pode reduzir o espaco em ate 75%, com economia media de 40%.

> ⚠️ Voce precisara remover manualmente os diretorios de cache existentes para que sejam recriados com compactacao pelo trainer.

---

## 🌈 Processamento de imagem e texto

Muitas configuracoes sao definidas no [dataloader config](DATALOADER.md), mas estas se aplicam globalmente.

### `--resolution_type`

- **O que**: Diz ao SimpleTuner se deve usar calculos por `area` ou por aresta de pixel. Uma abordagem hibrida `pixel_area` tambem e suportada, permitindo usar pixel em vez de megapixel para medidas de `area`.
- **Opcoes**:
  - `resolution_type=pixel_area`
    - Um valor de `resolution` 1024 sera mapeado internamente para uma medida de area precisa para bucketing eficiente.
    - Exemplos para `1024`: 1024x1024, 1216x832, 832x1216
  - `resolution_type=pixel`
    - Todas as imagens no dataset terao sua menor aresta redimensionada para essa resolucao, o que pode resultar em alto uso de VRAM devido ao tamanho das imagens resultantes.
    - Exemplos para `1024`: 1024x1024, 1766x1024, 1024x1766
  - `resolution_type=area`
    - **Deprecated**. Use `pixel_area`.

### `--resolution`

- **O que**: Resolucao de entrada expressa em tamanho de aresta em pixels
- **Padrao**: 1024
- **Nota**: Este e o padrao global, se um dataset nao definir resolucao.

### `--validation_resolution`

- **O que**: Resolucao de saida, em pixels, ou no formato `larguraxaltura`, como `1024x1024`. Multiplas resolucoes podem ser definidas, separadas por virgulas.
- **Por que**: Todas as imagens geradas durante a validacao terao essa resolucao. Util se o modelo estiver sendo treinado com resolucao diferente.

### `--validation_method`

- **O que**: Escolhe como as validacoes sao executadas.
- **Opcoes**: `simpletuner-local` (padrao) roda o pipeline embutido; `external-script` roda um executavel fornecido pelo usuario.
- **Por que**: Permite delegar validacao a um sistema externo sem pausar o treino para pipeline local.

### `--validation_external_script`

- **O que**: Executavel para rodar quando `--validation_method=external-script`. Usa separacao estilo shell, entao coloque aspas no comando.
- **Placeholders**: Voce pode embutir estes tokens (formatados com `.format`) para passar contexto de treino. Valores ausentes sao substituidos por string vazia, salvo quando indicado:
  - `{local_checkpoint_path}` → diretorio do ultimo checkpoint dentro de `output_dir` (requer ao menos um checkpoint).
  - `{global_step}` → passo global atual.
  - `{tracker_run_name}` → valor de `--tracker_run_name`.
  - `{tracker_project_name}` → valor de `--tracker_project_name`.
  - `{model_family}` → valor de `--model_family`.
  - `{model_type}` / `{lora_type}` → tipo de modelo e variante LoRA.
  - `{huggingface_path}` → valor de `--hub_model_id` (se definido).
  - `{remote_checkpoint_path}` → URL remota do ultimo upload (vazio para hook de validacao).
  - Qualquer valor `validation_*` no config (ex.: `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`).
- **Exemplo**: `--validation_external_script="/opt/tools/validate.sh {local_checkpoint_path} {global_step}"`

### `--validation_external_background`

- **O que**: Quando definido, `--validation_external_script` e iniciado em background (fire-and-forget).
- **Por que**: Mantem o treinamento rodando sem esperar o script externo; codigos de saida nao sao verificados nesse modo.

### `--post_upload_script`

- **O que**: Executavel opcional rodado apos cada provedor de publicacao e upload no Hugging Face Hub terminar (modelo final e uploads de checkpoints). Roda de forma assincrona para nao bloquear o treinamento.
- **Placeholders**: Mesmas substituicoes de `--validation_external_script`, mais `{remote_checkpoint_path}` (URI retornada pelo provedor) para que voce encaminhe a URL publicada para sistemas downstream.
- **Notas**:
  - Scripts rodam por provedor/upload; erros sao logados, mas nao param o treino.
  - Scripts tambem sao chamados quando nao ha upload remoto, entao voce pode usa-los para automacao local (ex.: rodar inferencia em outra GPU).
  - O SimpleTuner nao consome resultados do seu script — registre diretamente no tracker se quiser metricas ou imagens.
- **Exemplo**:
  ```bash
  --post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
  ```
  Onde `/opt/hooks/notify.sh` pode postar no seu sistema de tracking:
  ```bash
  #!/usr/bin/env bash
  REMOTE="$1"
  PROJECT="$2"
  RUN="$3"
  curl -X POST "https://tracker.internal/api/runs/${PROJECT}/${RUN}/artifacts" \
       -H "Content-Type: application/json" \
       -d "{\"remote_uri\":\"${REMOTE}\"}"
  ```
- **Exemplos funcionais**:
  - `simpletuner/examples/external-validation/replicate_post_upload.py` mostra um hook Replicate que consome `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}` e `{huggingface_path}` para disparar inferencia apos uploads.
  - `simpletuner/examples/external-validation/wavespeed_post_upload.py` mostra um hook WaveSpeed usando os mesmos placeholders mais o polling assincrono do WaveSpeed.
  - `simpletuner/examples/external-validation/fal_post_upload.py` mostra um hook fal.ai Flux LoRA (requer `FAL_KEY`).
  - `simpletuner/examples/external-validation/use_second_gpu.py` roda inferencia Flux LoRA em uma GPU secundaria e funciona mesmo sem uploads remotos.

### `--post_checkpoint_script`

- **O que**: Executavel para rodar imediatamente apos cada diretorio de checkpoint ser gravado em disco (antes de qualquer upload). Roda de forma assincrona no processo principal.
- **Placeholders**: Mesmas substituicoes de `--validation_external_script`, incluindo `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{model_type}`, `{lora_type}`, `{huggingface_path}` e qualquer valor `validation_*` de config. `{remote_checkpoint_path}` resolve para vazio nesse hook.
- **Notas**:
  - Dispara para checkpoints agendados, manuais e rolling assim que terminam de salvar localmente.
  - Util para acionar automacoes locais (copiar para outro volume, rodar avaliacao) sem esperar uploads terminarem.
- **Exemplo**:
  ```bash
  --post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'
  ```

### `--validation_adapter_path`

- **O que**: Carrega temporariamente um unico adaptador LoRA durante validacoes agendadas.
- **Formatos**:
  - Repo Hugging Face: `org/repo` ou `org/repo:weight_name.safetensors` (padrao `pytorch_lora_weights.safetensors`).
  - Caminho local para arquivo ou diretorio apontando para um adaptador safetensors.
- **Notas**:
  - Mutuamente exclusivo com `--validation_adapter_config`; fornecer ambos gera erro.
  - O adaptador so e anexado durante validacoes (pesos base de treino permanecem intactos).

### `--validation_adapter_name`

- **O que**: Identificador opcional para aplicar ao adaptador temporario carregado via `--validation_adapter_path`.
- **Por que**: Controla como a execucao e rotulada nos logs/WebUI e garante nomes previsiveis quando varios adaptadores sao testados em sequencia.

### `--validation_adapter_strength`

- **O que**: Multiplicador de forca aplicado ao habilitar o adaptador temporario (padrao `1.0`).
- **Por que**: Permite varrer escalas LoRA mais leves/pesadas durante validacao sem alterar o estado do treino; aceita qualquer valor maior que zero.

### `--validation_adapter_mode`

- **Opcoes**: `adapter_only`, `comparison`, `none`
- **O que**:
  - `adapter_only`: roda validacoes apenas com o adaptador temporario anexado.
  - `comparison`: gera amostras do modelo base e com adaptador para comparacao lado a lado.
  - `none`: ignora anexar o adaptador (util para desativar o recurso sem apagar flags CLI).

### `--validation_adapter_config`

- **O que**: Arquivo JSON ou JSON inline que descreve varias combinacoes de adaptadores para iterar.
- **Formato**: Um array de entradas ou um objeto com um array `runs`. Cada entrada pode incluir:
  - `label`: Nome amigavel exibido nos logs/UI.
  - `path`: Repo Hugging Face ou caminho local (mesmos formatos de `--validation_adapter_path`).
  - `adapter_name`: Identificador opcional por adaptador.
  - `strength`: Override escalar opcional.
  - `adapters`/`paths`: Array de objetos/strings para carregar multiplos adaptadores em uma unica execucao.
- **Notas**:
  - Quando fornecido, as opcoes de adaptador unico (`--validation_adapter_path`, `--validation_adapter_name`, `--validation_adapter_strength`, `--validation_adapter_mode`) sao ignoradas/desabilitadas na UI.
  - Cada execucao e carregada uma por vez e totalmente removida antes da proxima.

### `--validation_preview`

- **O que**: Transmite previews intermediarios durante a amostragem de difusao usando Tiny AutoEncoders.
- **Padrao**: False
- **Por que**: Habilita preview em tempo real das imagens de validacao enquanto sao geradas, decodificando via Tiny AutoEncoder e enviando por webhook. Isso permite monitorar o progresso passo a passo em vez de esperar a geracao completa.
- **Notas**:
  - Disponivel apenas para familias de modelos com suporte a Tiny AutoEncoder (ex.: Flux, SDXL, SD3)
  - Requer configuracao de webhook para receber previews
  - Use `--validation_preview_steps` para controlar a frequencia de decodificacao

### `--validation_preview_steps`

- **O que**: Intervalo para decodificar e transmitir previews de validacao
- **Padrao**: 1
- **Por que**: Controla com que frequencia latentes intermediarios sao decodificados durante a amostragem. Um valor maior (ex.: 3) reduz o overhead ao decodificar apenas a cada N passos.
- **Exemplo**: Com `--validation_num_inference_steps=20` e `--validation_preview_steps=5`, voce recebera 4 previews durante a geracao (passos 5, 10, 15, 20).

### `--evaluation_type`

- **O que**: Habilita avaliacao CLIP das imagens geradas durante validacoes.
- **Por que**: Scores CLIP calculam a distancia entre features da imagem gerada e o prompt de validacao. Isso pode indicar se a aderencia ao prompt esta melhorando, mas exige muitos prompts para ter valor.
- **Opcoes**: "none" ou "clip"
- **Agendamento**: Use `--eval_steps_interval` para agendamento por passos ou `--eval_epoch_interval` para agendamento por epocas (fracoes como `0.5` rodam varias vezes por epoca). Se ambos estiverem definidos, o trainer avisa e executa ambos.

### `--eval_loss_disable`

- **O que**: Desabilita o calculo de eval loss durante validacao.
- **Por que**: Quando um dataset de avaliacao e configurado, a loss sera calculada automaticamente. Se a avaliacao CLIP estiver habilitada, ambos rodam. Este flag permite desabilitar a eval loss mantendo CLIP.

### `--validation_using_datasets`

- **O que**: Usa imagens dos datasets de treinamento para validacao ao inves de geracao pura texto-para-imagem.
- **Por que**: Habilita modo de validacao imagem-para-imagem (img2img) ou imagem-para-video (i2v) onde o modelo usa imagens de treinamento como entradas de conditioning. Util para:
  - Testar modelos de edicao/inpainting que requerem imagens de entrada
  - Avaliar quao bem o modelo preserva a estrutura da imagem
  - Modelos que suportam workflows duais texto-para-imagem E imagem-para-imagem (ex., Flux2, LTXVideo2)
  - **Modelos de video I2V** (HunyuanVideo, WAN, Kandinsky5Video): Usa imagens de um dataset de imagens como entrada de conditioning do primeiro frame para validacao de geracao de video
- **Notas**:
  - Requer que o modelo tenha um pipeline `IMG2IMG` ou `IMG2VIDEO` registrado
  - Pode ser combinado com `--eval_dataset_id` para obter imagens de um dataset especifico
  - Para modelos i2v, permite usar um dataset de imagens simples para validacao sem a configuracao complexa de pareamento de datasets de conditioning usada durante o treinamento
  - A intensidade do denoise e controlada pelas configuracoes normais de timestep de validacao

### `--eval_dataset_id`

- **O que**: ID especifico do dataset para usar no sourcing de imagens de avaliacao/validacao.
- **Por que**: Ao usar `--validation_using_datasets` ou validacao baseada em conditioning, controla qual dataset fornece as imagens de entrada:
  - Sem esta opcao, imagens sao selecionadas aleatoriamente de todos os datasets de treinamento
  - Com esta opcao, apenas o dataset especificado e usado para entradas de validacao
- **Notas**:
  - O ID do dataset deve corresponder a um dataset configurado no seu config do dataloader
  - Util para manter avaliacao consistente usando um dataset de eval dedicado
  - Para modelos de conditioning, os dados de conditioning do dataset (se houver) tambem serao usados

---

## Entendendo Modos de Conditioning e Validacao

SimpleTuner suporta tres paradigmas principais para modelos que usam entradas de conditioning (imagens de referencia, sinais de controle, etc.):

### 1. Modelos que REQUEREM Conditioning

Alguns modelos nao funcionam sem entradas de conditioning:

- **Flux Kontext**: Sempre precisa de imagens de referencia para treinamento estilo edicao
- **Treinamento ControlNet**: Requer imagens de sinal de controle

Para estes modelos, um dataset de conditioning e obrigatorio. A WebUI mostrara opcoes de conditioning como obrigatorias, e o treinamento falhara sem elas.

### 2. Modelos que SUPORTAM Conditioning Opcional

Alguns modelos podem operar em modos texto-para-imagem E imagem-para-imagem:

- **Flux2**: Suporta treinamento dual T2I/I2I com imagens de referencia opcionais
- **LTXVideo2**: Suporta T2V e I2V (imagem-para-video) com conditioning de primeiro frame opcional
- **LongCat-Video**: Suporta conditioning de frames opcional
- **HunyuanVideo i2v**: Suporta I2V com conditioning de primeiro frame (flavours: `i2v-480p`, `i2v-720p`, etc.)
- **WAN i2v**: Suporta I2V com conditioning de primeiro frame
- **Kandinsky5Video i2v**: Suporta I2V com conditioning de primeiro frame

Para estes modelos, voce PODE adicionar datasets de conditioning mas nao e obrigatorio. A WebUI mostrara opcoes de conditioning como opcionais.

**Atalho de Validacao I2V**: Para modelos de video i2v, voce pode usar `--validation_using_datasets` com um dataset de imagens (especificado via `--eval_dataset_id`) para obter imagens de conditioning de validacao diretamente, sem precisar configurar o pareamento completo de datasets de conditioning usado durante o treinamento.

### 3. Modos de Validacao

| Modo | Flag | Comportamento |
|------|------|---------------|
| **Texto-para-Imagem/Video** | (padrao) | Gera apenas de prompts de texto |
| **Baseado em Dataset (img2img)** | `--validation_using_datasets` | Denoise parcial de imagens de datasets |
| **Baseado em Dataset (i2v)** | `--validation_using_datasets` | Para modelos de video i2v, usa imagens como conditioning de primeiro frame |
| **Baseado em Conditioning** | (auto quando conditioning configurado) | Usa entradas de conditioning durante validacao |

**Combinando modos**: Quando um modelo suporta conditioning E `--validation_using_datasets` esta habilitado:
- O sistema de validacao obtem imagens de datasets
- Se esses datasets tem dados de conditioning, sao usados automaticamente
- Use `--eval_dataset_id` para controlar qual dataset fornece entradas

**Modelos I2V com `--validation_using_datasets`**: Para modelos de video i2v (HunyuanVideo, WAN, Kandinsky5Video), habilitar este flag permite usar um dataset de imagens simples para validacao. As imagens sao usadas como entradas de conditioning de primeiro frame para gerar videos de validacao, sem precisar da configuracao complexa de pareamento de datasets de conditioning.

### Tipos de Dados de Conditioning

Diferentes modelos esperam diferentes dados de conditioning:

| Tipo | Modelos | Configuracao do Dataset |
|------|---------|------------------------|
| `conditioning` | ControlNet, Control | `type: conditioning` no config do dataset |
| `image` | Flux Kontext | `type: image` (dataset de imagem padrao) |
| `latents` | Flux, Flux2 | Conditioning e VAE-encoded automaticamente |

---

### `--caption_strategy`

- **O que**: Estrategia para derivar captions. **Opcoes**: `textfile`, `filename`, `parquet`, `instanceprompt`
- **Por que**: Determina como as captions sao geradas para imagens de treino.
  - `textfile` usa o conteudo de um arquivo `.txt` com o mesmo nome da imagem
  - `filename` aplica limpeza ao nome do arquivo antes de usar como caption
  - `parquet` requer um arquivo parquet no dataset e usa a coluna `caption` a menos que `parquet_caption_column` seja fornecida. Todas as captions devem estar presentes a menos que `parquet_fallback_caption_column` seja fornecida.
  - `instanceprompt` usa o valor de `instance_prompt` na config do dataset como prompt para cada imagem.

### `--conditioning_multidataset_sampling` {#--conditioning_multidataset_sampling}

- **O que**: Como amostrar de multiplos datasets de condicionamento. **Opcoes**: `combined`, `random`
- **Por que**: Ao treinar com varios datasets de condicionamento (ex.: multiplas imagens de referencia ou sinais de controle), isso determina como sao usados:
  - `combined` junta os condicionamentos, mostrando-os simultaneamente durante o treino. Util para composicao com varias imagens.
  - `random` seleciona um dataset de condicionamento por sample, alternando entre condicoes durante o treino.
- **Nota**: Ao usar `combined`, voce nao pode definir `captions` separadas nos datasets de condicionamento; as captions do dataset de origem sao usadas.
- **Veja tambem**: [DATALOADER.md](DATALOADER.md#conditioning_data) para configurar multiplos datasets de condicionamento.

---

## 🎛 Parametros de treinamento

### `--num_train_epochs`

- **O que**: Numero de epocas de treinamento (quantas vezes todas as imagens sao vistas). Definir como 0 permite que `--max_train_steps` tenha prioridade.
- **Por que**: Determina o numero de repeticoes de imagens, impactando a duracao do treino. Mais epocas tendem a resultar em overfitting, mas podem ser necessarias para capturar os conceitos. Um valor razoavel pode ser 5 a 50.

### `--max_train_steps`

- **O que**: Numero de passos para encerrar o treinamento. Se 0, `--num_train_epochs` tem prioridade.
- **Por que**: Util para encurtar a duracao do treinamento.

### `--ignore_final_epochs`

- **O que**: Ignora as ultimas epocas contadas em favor de `--max_train_steps`.
- **Por que**: Ao mudar o tamanho do dataloader, o treino pode terminar antes do esperado porque o calculo de epocas muda. Esta opcao ignora as ultimas epocas e continua ate `--max_train_steps`.

### `--learning_rate`

- **O que**: Taxa de aprendizado inicial apos warmup.
- **Por que**: A taxa de aprendizado e como o "tamanho do passo" nas atualizacoes de gradiente. Alta demais causa overshoot; baixa demais nao chega na solucao ideal. Para `full`, pode ser tao baixa quanto `1e-7` a no maximo `1e-6`; para `lora`, minima `1e-5` e maxima ate `1e-3`. Em taxas maiores, e vantajoso usar EMA com warmup — veja `--use_ema`, `--lr_warmup_steps` e `--lr_scheduler`.

### `--lr_scheduler`

- **O que**: Como a taxa de aprendizado escala ao longo do tempo.
- **Opcoes**: constant, constant_with_warmup, cosine, cosine_with_restarts, **polynomial** (recomendado), linear
- **Por que**: Modelos se beneficiam de ajustes continuos para explorar melhor o loss landscape. O padrao e cosine; isso permite transicao suave entre extremos. Com taxa constante, e comum escolher um valor alto demais (divergencia) ou baixo demais (minimo local). Um agendamento polynomial funciona melhor com warmup, aproximando-se gradualmente de `learning_rate` e depois desacelerando ate `--lr_end` ao final.

### `--optimizer`

- **O que**: Otimizador usado no treinamento.
- **Opcoes**: adamw_bf16, ao-adamw8bit, ao-adamw4bit, ao-adamfp8, ao-adamwfp8, adamw_schedulefree, adamw_schedulefree+aggressive, adamw_schedulefree+no_kahan, optimi-stableadamw, optimi-adamw, optimi-lion, optimi-radam, optimi-ranger, optimi-adan, optimi-adam, optimi-sgd, soap, bnb-adagrad, bnb-adagrad8bit, bnb-adam, bnb-adam8bit, bnb-adamw, bnb-adamw8bit, bnb-adamw-paged, bnb-adamw8bit-paged, bnb-lion, bnb-lion8bit, bnb-lion-paged, bnb-lion8bit-paged, bnb-ademamix, bnb-ademamix8bit, bnb-ademamix-paged, bnb-ademamix8bit-paged, prodigy

> Nota: Alguns otimizadores podem nao estar disponiveis em hardware nao-NVIDIA.

### `--optimizer_config`

- **O que**: Ajusta configuracoes do otimizador.
- **Por que**: Como os otimizadores tem muitas configuracoes, nao e viavel ter um argumento para cada uma. Em vez disso, use uma lista separada por virgulas para sobrescrever defaults.
- **Exemplo**: Voce pode definir `d_coef` para o otimizador **prodigy**: `--optimizer_config=d_coef=0.1`

> Nota: Betas do otimizador sao sobrescritos com `--optimizer_beta1` e `--optimizer_beta2`.

### `--train_batch_size`

- **O que**: Batch size para o dataloader de treinamento.
- **Por que**: Afeta consumo de memoria, qualidade de convergencia e velocidade. Batch maior tende a melhorar resultados, mas pode causar overfitting ou instabilidade e aumentar a duracao do treino. Experimente, mas em geral tente maximizar VRAM sem reduzir a velocidade.

### `--gradient_accumulation_steps`

- **O que**: Numero de passos a acumular antes de fazer backward/update, dividindo o trabalho em varios batches para economizar memoria ao custo de runtime maior.
- **Por que**: Util para modelos ou datasets maiores.

> Nota: Nao habilite fused backward pass em otimizadores ao usar gradient accumulation.

### `--allow_dataset_oversubscription` {#--allow_dataset_oversubscription}

- **O que**: Ajusta automaticamente `repeats` quando o dataset e menor que o batch efetivo.
- **Por que**: Evita falhas quando o tamanho do dataset nao atende os requisitos minimos da configuracao multi-GPU.
- **Como funciona**:
  - Calcula o **batch efetivo**: `train_batch_size × num_gpus × gradient_accumulation_steps`
  - Se algum bucket de aspecto tiver menos samples que o batch efetivo, aumenta `repeats`
  - So aplica quando `repeats` nao esta configurado explicitamente no dataset
  - Registra um warning mostrando o ajuste e a justificativa
- **Casos de uso**:
  - Datasets pequenos (< 100 imagens) com varias GPUs
  - Experimentar batch sizes diferentes sem reconfigurar datasets
  - Prototipar antes de coletar um dataset completo
- **Exemplo**: Com 25 imagens, 8 GPUs e `train_batch_size=4`, o batch efetivo e 32. Esta flag definira `repeats=1` para fornecer 50 samples (25 × 2).
- **Nota**: Isso **nao** sobrescreve valores de `repeats` definidos manualmente no dataloader. Assim como `--disable_bucket_pruning`, esta flag oferece conveniencia sem comportamento surpreendente.

Veja o guia [DATALOADER.md](DATALOADER.md#automatic-dataset-oversubscription) para mais detalhes sobre tamanho de dataset em treino multi-GPU.

---

## 🛠 Otimizacoes avancadas

### `--use_ema`

- **O que**: Manter uma media movel exponencial (EMA) dos pesos e como reintegrar periodicamente o modelo em si mesmo.
- **Por que**: Pode melhorar estabilidade ao custo de mais recursos e um pequeno aumento no tempo de treino.

### `--ema_device`

- **Opcoes**: `cpu`, `accelerator`; padrao: `cpu`
- **O que**: Escolhe onde os pesos EMA ficam entre atualizacoes.
- **Por que**: Manter EMA no acelerador atualiza mais rapido, mas custa VRAM. Manter no CPU reduz memoria, mas exige transferencias, a menos que `--ema_cpu_only` esteja definido.

### `--ema_cpu_only`

- **O que**: Impede que os pesos EMA sejam movidos de volta para o acelerador ao atualizar quando `--ema_device=cpu`.
- **Por que**: Economiza tempo de transferencia e VRAM para EMAs grandes. Nao tem efeito se `--ema_device=accelerator` pois os pesos ja ficam no acelerador.

### `--ema_foreach_disable`

- **O que**: Desabilita o uso dos kernels `torch._foreach_*` para atualizacoes EMA.
- **Por que**: Alguns backends ou combinacoes de hardware tem problemas com foreach. Desabilitar volta para a implementacao escalar, com leve reducao de velocidade.

### `--ema_update_interval`

- **O que**: Reduz a frequencia de atualizacao dos parametros EMA.
- **Por que**: Atualizar a cada passo e desnecessario. Por exemplo, `--ema_update_interval=100` atualiza EMA a cada 100 passos, reduzindo overhead quando `--ema_device=cpu` ou `--ema_cpu_only` estao habilitados.

### `--ema_decay`

- **O que**: Controla o fator de suavizacao usado nas atualizacoes EMA.
- **Por que**: Valores maiores (ex.: `0.999`) fazem a EMA responder lentamente e produzir pesos mais estaveis. Valores menores (ex.: `0.99`) se adaptam mais rapido.

### `--snr_gamma`

- **O que**: Utiliza fator de perda ponderado por min-SNR.
- **Por que**: O gamma min-SNR pondera a perda de um timestep pela posicao no schedule. Timesteps muito ruidosos sao reduzidos, e timesteps menos ruidosos aumentados. O valor recomendado pelo paper original e **5**, mas voce pode usar **1** a **20**. Acima de 20, a matematica muda pouco. Um valor **1** e o mais forte.

### `--use_soft_min_snr`

- **O que**: Treina usando um peso mais gradual na loss landscape.
- **Por que**: Em pixel diffusion, modelos degradam sem um agendamento de loss especifico. Isso e o caso do DeepFloyd, onde soft-min-snr-gamma foi quase obrigatorio. Em difusao latente, pode funcionar, mas em experimentos pequenos pode produzir resultados borrados.

### `--diff2flow_enabled`

- **O que**: Habilita a ponte Diffusion-to-Flow para modelos epsilon ou v-prediction.
- **Por que**: Permite que modelos treinados com objetivos de difusao padrao usem alvos flow-matching (noise - latents) sem mudar a arquitetura.
- **Nota**: Recurso experimental.

### `--diff2flow_loss`

- **O que**: Treina com loss de Flow Matching em vez da loss de predicao nativa.
- **Por que**: Quando habilitado com `--diff2flow_enabled`, calcula a loss contra o alvo flow (noise - latents) em vez do alvo nativo (epsilon ou velocity).
- **Nota**: Requer `--diff2flow_enabled`.

### `--scheduled_sampling_max_step_offset`

- **O que**: Numero maximo de passos para "roll out" durante o treinamento.
- **Por que**: Habilita Scheduled Sampling (Rollout), onde o modelo gera suas proprias entradas por alguns passos. Isso ajuda o modelo a corrigir seus erros e reduz exposure bias.
- **Padrao**: 0 (desabilitado). Defina um inteiro positivo (ex.: 5 ou 10) para habilitar.

### `--scheduled_sampling_strategy`

- **O que**: Estrategia para escolher o offset do rollout.
- **Opcoes**: `uniform`, `biased_early`, `biased_late`.
- **Padrao**: `uniform`.
- **Por que**: Controla a distribuicao de comprimentos de rollout. `uniform` amostra igualmente; `biased_early` favorece rollouts curtos; `biased_late` favorece rollouts longos.

### `--scheduled_sampling_probability`

- **O que**: Probabilidade de aplicar um offset de rollout nao-zero para um sample.
- **Padrao**: 0.0.
- **Por que**: Controla com que frequencia scheduled sampling e aplicado. 0.0 desabilita mesmo se `max_step_offset` > 0. 1.0 aplica em todos os samples.

### `--scheduled_sampling_prob_start`

- **O que**: Probabilidade inicial de scheduled sampling no inicio do ramp.
- **Padrao**: 0.0.

### `--scheduled_sampling_prob_end`

- **O que**: Probabilidade final de scheduled sampling no fim do ramp.
- **Padrao**: 0.5.

### `--scheduled_sampling_ramp_steps`

- **O que**: Numero de passos para rampar a probabilidade de `prob_start` para `prob_end`.
- **Padrao**: 0 (sem ramp).

### `--scheduled_sampling_start_step`

- **O que**: Passo global para iniciar o ramp de scheduled sampling.
- **Padrao**: 0.0.

### `--scheduled_sampling_ramp_shape`

- **O que**: Forma do ramp de probabilidade.
- **Opcoes**: `linear`, `cosine`.
- **Padrao**: `linear`.

### `--scheduled_sampling_sampler`

- **O que**: Solver usado para os passos de rollout.
- **Opcoes**: `unipc`, `euler`, `dpm`, `rk4`.
- **Padrao**: `unipc`.

### `--scheduled_sampling_order`

- **O que**: Ordem do solver usado no rollout.
- **Padrao**: 2.

### `--scheduled_sampling_reflexflow`

- **O que**: Habilita melhorias estilo ReflexFlow (anti-drift + ponderacao compensada por frequencia) durante scheduled sampling para modelos flow-matching.
- **Por que**: Reduz exposure bias ao fazer rollout em modelos flow-matching ao adicionar regularizacao direcional e perda ponderada por viés.
- **Padrao**: Habilita automaticamente para modelos flow-matching quando `--scheduled_sampling_max_step_offset` > 0; sobrescreva com `--scheduled_sampling_reflexflow=false`.

### `--scheduled_sampling_reflexflow_alpha`

- **O que**: Fator de escala para o peso de compensacao por frequencia derivado do exposure bias.
- **Padrao**: 1.0.
- **Por que**: Valores maiores aumentam regioes com maior exposure bias durante o rollout em modelos flow-matching.

### `--scheduled_sampling_reflexflow_beta1`

- **O que**: Peso para o regularizador anti-drift (direcional) do ReflexFlow.
- **Padrao**: 10.0.
- **Por que**: Controla o quao forte o modelo e incentivado a alinhar sua direcao prevista com o sample alvo limpo ao usar scheduled sampling em modelos flow-matching.

### `--scheduled_sampling_reflexflow_beta2`

- **O que**: Peso para o termo de compensacao por frequencia (reponderacao de loss) do ReflexFlow.
- **Padrao**: 1.0.
- **Por que**: Escala a loss flow-matching reponderada, seguindo o knob β₂ descrito no paper ReflexFlow.

---

## 🎯 CREPA (Cross-frame Representation Alignment)

CREPA e uma tecnica de regularizacao para fine-tuning de modelos de difusao de video que melhora a consistencia temporal alinhando hidden states com features visuais pre-treinadas de frames adjacentes. Baseada no paper ["Cross-Frame Representation Alignment for Fine-Tuning Video Diffusion Models"](https://arxiv.org/abs/2506.09229).

### `--crepa_enabled`

- **O que**: Habilita regularizacao CREPA durante o treinamento.
- **Por que**: Melhora a consistencia semantica entre frames ao alinhar hidden states do DiT com features DINOv2 de frames vizinhos.
- **Padrao**: `false`
- **Nota**: So se aplica a modelos de video (Wan, LTXVideo, SanaVideo, Kandinsky5).

### `--crepa_block_index`

- **O que**: Qual bloco transformer usar para alinhamento.
- **Por que**: O paper recomenda o bloco 8 para CogVideoX e o bloco 10 para Hunyuan Video. Blocos anteriores tendem a funcionar melhor por atuarem como "encoder" do DiT.
- **Obrigatorio**: Sim, quando CREPA esta habilitado.

### `--crepa_lambda`

- **O que**: Peso da loss de alinhamento CREPA em relacao a loss principal.
- **Por que**: Controla o quanto a regularizacao influencia o treino. O paper usa 0.5 para CogVideoX e 1.0 para Hunyuan Video.
- **Padrao**: `0.5`

### `--crepa_adjacent_distance`

- **O que**: Distancia `d` para alinhamento de frames vizinhos.
- **Por que**: Pela Equacao 6 do paper, $K = \{f-d, f+d\}$ define quais frames alinhar. Com `d=1`, cada frame alinha com seus vizinhos imediatos.
- **Padrao**: `1`

### `--crepa_adjacent_tau`

- **O que**: Coeficiente de temperatura para o peso exponencial de distancia.
- **Por que**: Controla quao rapido o peso de alinhamento decai com a distancia via $e^{-|k-f|/\tau}$. Valores menores focam mais nos vizinhos imediatos.
- **Padrao**: `1.0`

### `--crepa_cumulative_neighbors`

- **O que**: Usa modo cumulativo em vez de modo adjacente.
- **Por que**:
  - **Modo adjacente (padrao)**: Alinha apenas com frames na distancia exata `d` (corresponde a $K = \{f-d, f+d\}$)
  - **Modo cumulativo**: Alinha com todos os frames de distancia 1 ate `d`, oferecendo gradientes mais suaves
- **Padrao**: `false`

### `--crepa_normalize_by_frames`

- **O que**: Normaliza a loss de alinhamento pelo numero de frames.
- **Por que**: Garante escala consistente da loss independentemente do comprimento do video. Desabilite para dar mais sinal a videos longos.
- **Padrao**: `true`

### `--crepa_spatial_align`

- **O que**: Usa interpolacao espacial quando o numero de tokens difere entre DiT e encoder.
- **Por que**: Hidden states do DiT e features DINOv2 podem ter resolucoes espaciais diferentes. Quando habilitado, interpolacao bilinear alinha espacialmente. Quando desabilitado, usa pooling global.
- **Padrao**: `true`

### `--crepa_model`

- **O que**: Qual encoder pre-treinado usar para extrair features.
- **Por que**: O paper usa DINOv2-g (ViT-Giant). Variantes menores como `dinov2_vitb14` usam menos memoria.
- **Padrao**: `dinov2_vitg14`
- **Opcoes**: `dinov2_vitg14`, `dinov2_vitb14`, `dinov2_vits14`

### `--crepa_encoder_frames_batch_size`

- **O que**: Quantos frames o encoder externo processa em paralelo. Zero ou negativo para todos os frames do batch de uma vez. Se nao for divisor, o restante sera processado como batch menor.
- **Por que**: Como encoders tipo DINO sao modelos de imagem, podem processar frames em batches fatiados para reduzir VRAM ao custo de velocidade.
- **Padrao**: `-1`

### `--crepa_use_backbone_features`

- **O que**: Pula o encoder externo e alinha um bloco estudante a um bloco professor dentro do modelo de difusao.
- **Por que**: Evita carregar DINOv2 quando o backbone ja tem uma camada semantica mais forte para supervisionar.
- **Padrao**: `false`

### `--crepa_teacher_block_index`

- **O que**: Indice do bloco professor ao usar features do backbone.
- **Por que**: Permite alinhar um bloco estudante mais cedo a um bloco professor mais profundo sem encoder externo. Usa o bloco estudante quando nao definido.
- **Padrao**: Usa `crepa_block_index` se nao for fornecido.

### `--crepa_encoder_image_size`

- **O que**: Resolucao de entrada para o encoder.
- **Por que**: Modelos DINOv2 funcionam melhor na resolucao de treino. O modelo giant usa 518x518.
- **Padrao**: `518`

### `--crepa_scheduler`

- **O que**: Agendamento para decaimento do coeficiente CREPA durante o treinamento.
- **Por que**: Permite reduzir a forca da regularizacao CREPA conforme o treinamento progride, prevenindo overfitting nas features profundas do encoder.
- **Opcoes**: `constant`, `linear`, `cosine`, `polynomial`
- **Padrao**: `constant`

### `--crepa_warmup_steps`

- **O que**: Numero de passos para aumentar linearmente o peso CREPA de 0 ate `crepa_lambda`.
- **Por que**: Aquecimento gradual pode ajudar a estabilizar o treinamento inicial antes da regularizacao CREPA entrar em acao.
- **Padrao**: `0`

### `--crepa_decay_steps`

- **O que**: Total de passos para decaimento (apos warmup). Defina como 0 para decair durante todo o treinamento.
- **Por que**: Controla a duracao da fase de decaimento. O decaimento comeca apos o warmup completar.
- **Padrao**: `0` (usa `max_train_steps`)

### `--crepa_lambda_end`

- **O que**: Peso CREPA final apos o decaimento completar.
- **Por que**: Definir como 0 efetivamente desabilita o CREPA no final do treinamento, util para text2video onde CREPA pode causar artefatos.
- **Padrao**: `0.0`

### `--crepa_power`

- **O que**: Fator de potencia para decaimento polinomial. 1.0 = linear, 2.0 = quadratico, etc.
- **Por que**: Valores maiores causam decaimento inicial mais rapido que desacelera no final.
- **Padrao**: `1.0`

### `--crepa_cutoff_step`

- **O que**: Passo de corte rigido apos o qual o CREPA e desabilitado.
- **Por que**: Util para desabilitar o CREPA apos o modelo convergir no alinhamento temporal.
- **Padrao**: `0` (sem corte baseado em passo)

### `--crepa_similarity_threshold`

- **O que**: Limiar de EMA de similaridade no qual o corte CREPA e acionado.
- **Por que**: Quando a media movel exponencial da similaridade atinge este valor, o CREPA e desabilitado para prevenir overfitting nas features profundas do encoder. Isto e particularmente util para treinamento text2video.
- **Padrao**: None (desabilitado)

### `--crepa_similarity_ema_decay`

- **O que**: Fator de decaimento da media movel exponencial para rastreamento de similaridade.
- **Por que**: Valores maiores fornecem rastreamento mais suave (0.99 ≈ janela de 100 passos), valores menores reagem mais rapido a mudancas.
- **Padrao**: `0.99`

### `--crepa_threshold_mode`

- **O que**: Comportamento quando o limiar de similaridade e atingido.
- **Opcoes**: `permanent` (CREPA permanece desligado apos atingir o limiar), `recoverable` (CREPA reabilita se a similaridade cair)
- **Padrao**: `permanent`

### Exemplo de configuracao

```toml
# Habilite CREPA para fine-tuning de video
crepa_enabled = true
crepa_block_index = 8          # Ajuste conforme seu modelo
crepa_lambda = 0.5
crepa_adjacent_distance = 1
crepa_adjacent_tau = 1.0
crepa_cumulative_neighbors = false
crepa_normalize_by_frames = true
crepa_spatial_align = true
crepa_model = "dinov2_vitg14"
crepa_encoder_frames_batch_size = -1
crepa_use_backbone_features = false
# crepa_teacher_block_index = 16
crepa_encoder_image_size = 518

# Agendamento CREPA (opcional)
# crepa_scheduler = "cosine"           # Tipo de decaimento: constant, linear, cosine, polynomial
# crepa_warmup_steps = 100             # Warmup antes do CREPA entrar em acao
# crepa_decay_steps = 1000             # Passos para decaimento (0 = treinamento inteiro)
# crepa_lambda_end = 0.0               # Peso final apos decaimento
# crepa_cutoff_step = 5000             # Passo de corte rigido (0 = desabilitado)
# crepa_similarity_threshold = 0.9    # Corte baseado em similaridade
# crepa_threshold_mode = "permanent"   # permanent ou recoverable
```

---

## 🔄 Checkpointing e retomada

### `--checkpoint_step_interval` (alias: `--checkpointing_steps`)

- **O que**: Intervalo em que checkpoints de estado de treino sao salvos (em passos).
- **Por que**: Util para retomar treinamento e para inferencia. A cada *n* iteracoes, um checkpoint parcial e salvo em `.safetensors`, no layout do Diffusers.

---

## 🔁 LayerSync (Hidden State Self-Alignment)

LayerSync incentiva uma camada "estudante" a combinar com uma camada "professora" mais forte dentro do mesmo transformer, usando similaridade de cosseno sobre tokens ocultos.

### `--layersync_enabled`

- **O que**: Habilita alinhamento LayerSync de hidden states entre dois blocos transformer do mesmo modelo.
- **Notas**: Aloca um buffer de hidden states; falha no startup se flags obrigatorias faltarem.
- **Padrao**: `false`

### `--layersync_student_block`

- **O que**: Indice do bloco transformer tratado como ancora estudante.
- **Indexacao**: Aceita profundidades 1-based do paper LayerSync ou IDs 0-based; a implementacao tenta `idx-1` primeiro, depois `idx`.
- **Obrigatorio**: Sim quando LayerSync esta habilitado.

### `--layersync_teacher_block`

- **O que**: Indice do bloco transformer tratado como alvo professor (pode ser mais profundo que o estudante).
- **Indexacao**: Mesma logica 1-based primeiro, depois fallback 0-based do bloco estudante.
- **Padrao**: Usa o bloco estudante quando omitido para que a loss vire auto-similaridade.

### `--layersync_lambda`

- **O que**: Peso da loss de alinhamento por cosseno entre hidden states do estudante e professor (cosseno negativo).
- **Efeito**: Escala o regularizador auxiliar adicionado a loss base; valores maiores empurram tokens a alinhar mais fortemente.
- **Nome upstream**: `--reg-weight` na base de codigo LayerSync original.
- **Obrigatorio**: Deve ser > 0 quando LayerSync esta habilitado (caso contrario o treino aborta).
- **Padrao**: `0.2` quando LayerSync esta habilitado (igual ao repo de referencia), `0.0` caso contrario.

Mapeamento de opcoes upstream (LayerSync → SimpleTuner):
- `--encoder-depth` → `--layersync_student_block` (aceita profundidade 1-based ou indice 0-based)
- `--gt-encoder-depth` → `--layersync_teacher_block` (1-based preferido; padrao estudante quando omitido)
- `--reg-weight` → `--layersync_lambda`

> Notas: LayerSync sempre faz detach do hidden state professor antes da similaridade, como na implementacao de referencia. Depende de modelos que exponham hidden states (a maioria dos backbones transformer no SimpleTuner) e adiciona memoria por passo para o buffer; desabilite se a VRAM estiver apertada.

### `--checkpoint_epoch_interval`

- **O que**: Roda checkpointing a cada N epocas completas.
- **Por que**: Complementa checkpoints por passo garantindo captura nas fronteiras de epoca, mesmo quando o numero de passos varia por amostragem multi-dataset.

### `--resume_from_checkpoint`

- **O que**: Especifica se e de onde retomar o treinamento. Aceita `latest`, um nome/caminho local de checkpoint ou um URI S3/R2.
- **Por que**: Permite continuar de um estado salvo, manualmente ou do mais recente.
- **Retomada remota**: Forneca um URI completo (`s3://bucket/jobs/.../checkpoint-100`) ou uma chave relativa ao bucket (`jobs/.../checkpoint-100`). `latest` so funciona com `output_dir` local.
- **Requisitos**: A retomada remota precisa de uma entrada S3 em publishing_config (bucket + credenciais) que consiga ler o checkpoint.
- **Notas**: Checkpoints remotos devem incluir `checkpoint_manifest.json` (gerado por execucoes recentes do SimpleTuner). Um checkpoint e composto de um subdiretorio `unet` e opcionalmente `unet_ema`. O `unet` pode ser colocado em qualquer layout Diffusers SDXL para usar como modelo normal.

> ℹ️ Modelos transformer como PixArt, SD3 ou Hunyuan usam os subdiretorios `transformer` e `transformer_ema`.

### `--disk_low_threshold`

- **O que**: Espaco minimo livre em disco necessario antes de salvar checkpoints.
- **Por que**: Previne falhas no treinamento por erros de disco cheio detectando espaco baixo antecipadamente e tomando uma acao configurada.
- **Formato**: String de tamanho como `100G`, `50M`, `1T`, `500K`, ou bytes simples.
- **Padrao**: Nenhum (funcionalidade desativada)

### `--disk_low_action`

- **O que**: Acao a tomar quando o espaco em disco esta abaixo do limite.
- **Opcoes**: `stop`, `wait`, `script`
- **Padrao**: `stop`
- **Comportamento**:
  - `stop`: Encerra o treinamento imediatamente com uma mensagem de erro.
  - `wait`: Faz loop a cada 30 segundos ate o espaco ficar disponivel. Pode esperar indefinidamente.
  - `script`: Executa o script especificado por `--disk_low_script` para liberar espaco.

### `--disk_low_script`

- **O que**: Caminho para um script de limpeza a executar quando o espaco em disco esta baixo.
- **Por que**: Permite limpeza automatizada (ex: remover checkpoints antigos, limpar cache) quando o espaco em disco esta baixo.
- **Notas**: Usado apenas quando `--disk_low_action=script`. O script deve ser executavel. Se o script falhar ou nao liberar espaco suficiente, o treinamento parara com um erro.
- **Padrao**: Nenhum

---

## 📊 Logging e monitoramento

### `--logging_dir`

- **O que**: Diretorio para logs do TensorBoard.
- **Por que**: Permite monitorar progresso e metricas de performance.

### `--report_to`

- **O que**: Especifica a plataforma para reportar resultados e logs.
- **Por que**: Habilita integracao com TensorBoard, wandb ou comet_ml para monitoramento. Use multiplos valores separados por virgula para reportar a varios trackers.
- **Opcoes**: wandb, tensorboard, comet_ml

## Variaveis de configuracao do ambiente

As opcoes acima se aplicam em grande parte ao `config.json` — mas algumas entradas devem ser definidas no `config.env`.

- `TRAINING_NUM_PROCESSES` deve ser definido para o numero de GPUs no sistema. Para a maioria dos casos, isso basta para habilitar treino DDP. Use `num_processes` no `config.json` se preferir nao usar `config.env`.
- `TRAINING_DYNAMO_BACKEND` padrao e `no`, mas pode ser definido para qualquer backend suportado do torch.compile (ex.: `inductor`, `aot_eager`, `cudagraphs`) e combinado com `--dynamo_mode`, `--dynamo_fullgraph` ou `--dynamo_use_regional_compilation` para ajuste fino.
- `SIMPLETUNER_LOG_LEVEL` padrao e `INFO`, mas pode ser definido para `DEBUG` para adicionar mais informacoes de issues no `debug.log`.
- `VENV_PATH` pode ser definido para o caminho do seu virtual env python, se nao estiver no local tipico `.venv`.
- `ACCELERATE_EXTRA_ARGS` pode ficar vazio ou conter argumentos extras como `--multi_gpu` ou flags especificas do FSDP.

---

Este e um panorama basico para ajudar voce a comecar. Para uma lista completa de opcoes e explicacoes mais detalhadas, consulte a especificacao completa:

```
usage: train.py [-h] --model_family
                {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                [--model_flavour MODEL_FLAVOUR] [--controlnet [CONTROLNET]]
                [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                --output_dir OUTPUT_DIR [--logging_dir LOGGING_DIR]
                --model_type {full,lora} [--seed SEED]
                [--resolution RESOLUTION]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--prediction_type {epsilon,v_prediction,sample,flow_matching}]
                [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH]
                [--vae_dtype {default,fp32,fp16,bf16}]
                [--vae_cache_ondemand [VAE_CACHE_ONDEMAND]]
                [--accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL]
                [--aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}]
                [--base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL]
                [--offload_during_startup [OFFLOAD_DURING_STARTUP]]
                [--quantize_via {cpu,accelerator,pipeline}]
                [--quantization_config QUANTIZATION_CONFIG]
                [--fuse_qkv_projections [FUSE_QKV_PROJECTIONS]]
                [--control [CONTROL]]
                [--controlnet_custom_config CONTROLNET_CUSTOM_CONFIG]
                [--controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH]
                [--tread_config TREAD_CONFIG]
                [--pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH]
                [--pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER]
                [--pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH]
                [--pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER]
                [--pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH]
                [--pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH]
                [--revision REVISION] [--variant VARIANT]
                [--base_model_default_dtype {bf16,fp32}]
                [--unet_attention_slice [UNET_ATTENTION_SLICE]]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--max_train_steps MAX_TRAIN_STEPS]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--learning_rate LEARNING_RATE] --optimizer
                {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                [--optimizer_config OPTIMIZER_CONFIG]
                [--lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--lr_warmup_steps LR_WARMUP_STEPS]
                [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                [--train_text_encoder [TRAIN_TEXT_ENCODER]]
                [--text_encoder_lr TEXT_ENCODER_LR]
                [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER]
                [--use_soft_min_snr [USE_SOFT_MIN_SNR]] [--use_ema [USE_EMA]]
                [--ema_device {accelerator,cpu}]
                [--ema_cpu_only [EMA_CPU_ONLY]]
                [--ema_update_interval EMA_UPDATE_INTERVAL]
                [--ema_foreach_disable [EMA_FOREACH_DISABLE]]
                [--ema_decay EMA_DECAY] [--lora_rank LORA_RANK]
                [--lora_alpha LORA_ALPHA] [--lora_type {standard,lycoris}]
                [--lora_dropout LORA_DROPOUT]
                [--lora_init_type {default,gaussian,loftq,olora,pissa}]
                [--peft_lora_mode {standard,singlora}]
                [--peft_lora_target_modules PEFT_LORA_TARGET_MODULES]
                [--singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS]
                [--init_lora INIT_LORA] [--lycoris_config LYCORIS_CONFIG]
                [--init_lokr_norm INIT_LOKR_NORM]
                [--flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}]
                [--use_dora [USE_DORA]]
                [--resolution_type {pixel,area,pixel_area}]
                --data_backend_config DATA_BACKEND_CONFIG
                [--caption_strategy {filename,textfile,instance_prompt,parquet}]
                [--conditioning_multidataset_sampling {combined,random}]
                [--instance_prompt INSTANCE_PROMPT]
                [--parquet_caption_column PARQUET_CAPTION_COLUMN]
                [--parquet_filename_column PARQUET_FILENAME_COLUMN]
                [--ignore_missing_files [IGNORE_MISSING_FILES]]
                [--vae_cache_scan_behaviour {recreate,sync}]
                [--vae_enable_slicing [VAE_ENABLE_SLICING]]
                [--vae_enable_tiling [VAE_ENABLE_TILING]]
                [--vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]]
                [--vae_batch_size VAE_BATCH_SIZE]
                [--caption_dropout_probability CAPTION_DROPOUT_PROBABILITY]
                [--tokenizer_max_length TOKENIZER_MAX_LENGTH]
                [--validation_step_interval VALIDATION_STEP_INTERVAL]
                [--validation_epoch_interval VALIDATION_EPOCH_INTERVAL]
                [--disable_benchmark [DISABLE_BENCHMARK]]
                [--validation_prompt VALIDATION_PROMPT]
                [--num_validation_images NUM_VALIDATION_IMAGES]
                [--num_eval_images NUM_EVAL_IMAGES]
                [--eval_steps_interval EVAL_STEPS_INTERVAL]
                [--eval_epoch_interval EVAL_EPOCH_INTERVAL]
                [--eval_timesteps EVAL_TIMESTEPS]
                [--eval_dataset_pooling [EVAL_DATASET_POOLING]]
                [--evaluation_type {none,clip}]
                [--pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH]
                [--validation_guidance VALIDATION_GUIDANCE]
                [--validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS]
                [--validation_on_startup [VALIDATION_ON_STARTUP]]
                [--validation_using_datasets [VALIDATION_USING_DATASETS]]
                [--validation_torch_compile [VALIDATION_TORCH_COMPILE]]
                [--validation_guidance_real VALIDATION_GUIDANCE_REAL]
                [--validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP]
                [--validation_negative_prompt VALIDATION_NEGATIVE_PROMPT]
                [--validation_randomize [VALIDATION_RANDOMIZE]]
                [--validation_seed VALIDATION_SEED]
                [--validation_disable [VALIDATION_DISABLE]]
                [--validation_prompt_library [VALIDATION_PROMPT_LIBRARY]]
                [--user_prompt_library USER_PROMPT_LIBRARY]
                [--eval_dataset_id EVAL_DATASET_ID]
                [--validation_stitch_input_location {left,right}]
                [--validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE]
                [--validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]]
                [--validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS]
                [--validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START]
                [--validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP]
                [--validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE]
                [--validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH]
                [--validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}]
                [--validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES]
                [--validation_audio_only [VALIDATION_AUDIO_ONLY]]
                [--validation_resolution VALIDATION_RESOLUTION]
                [--validation_seed_source {cpu,gpu}]
                [--i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]]
                [--flow_sigmoid_scale FLOW_SIGMOID_SCALE]
                [--flux_fast_schedule [FLUX_FAST_SCHEDULE]]
                [--flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]]
                [--flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]]
                [--flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA]
                [--flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA]
                [--flow_schedule_shift FLOW_SCHEDULE_SHIFT]
                [--flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]]
                [--flux_guidance_mode {constant,random-range}]
                [--flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]]
                [--flux_guidance_value FLUX_GUIDANCE_VALUE]
                [--flux_guidance_min FLUX_GUIDANCE_MIN]
                [--flux_guidance_max FLUX_GUIDANCE_MAX]
                [--t5_padding {zero,unmodified}]
                [--sd3_clip_uncond_behaviour {empty_string,zero}]
                [--sd3_t5_uncond_behaviour {empty_string,zero}]
                [--soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA]
                [--mixed_precision {no,fp16,bf16,fp8}]
                [--attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}]
                [--sageattention_usage {training,inference,training+inference}]
                [--disable_tf32 [DISABLE_TF32]]
                [--set_grads_to_none [SET_GRADS_TO_NONE]]
                [--noise_offset NOISE_OFFSET]
                [--noise_offset_probability NOISE_OFFSET_PROBABILITY]
                [--input_perturbation INPUT_PERTURBATION]
                [--input_perturbation_steps INPUT_PERTURBATION_STEPS]
                [--lr_end LR_END] [--lr_scale [LR_SCALE]]
                [--lr_scale_sqrt [LR_SCALE_SQRT]]
                [--ignore_final_epochs [IGNORE_FINAL_EPOCHS]]
                [--freeze_encoder_before FREEZE_ENCODER_BEFORE]
                [--freeze_encoder_after FREEZE_ENCODER_AFTER]
                [--freeze_encoder_strategy {before,between,after}]
                [--layer_freeze_strategy {none,bitfit}]
                [--fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]]
                [--save_text_encoder [SAVE_TEXT_ENCODER]]
                [--text_encoder_limit TEXT_ENCODER_LIMIT]
                [--prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]]
                [--only_instance_prompt [ONLY_INSTANCE_PROMPT]]
                [--data_aesthetic_score DATA_AESTHETIC_SCORE]
                [--delete_unwanted_images [DELETE_UNWANTED_IMAGES]]
                [--delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]]
                [--disable_bucket_pruning [DISABLE_BUCKET_PRUNING]]
                [--disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]]
                [--preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]]
                [--override_dataset_config [OVERRIDE_DATASET_CONFIG]]
                [--cache_dir CACHE_DIR] [--cache_dir_text CACHE_DIR_TEXT]
                [--cache_dir_vae CACHE_DIR_VAE]
                [--compress_disk_cache [COMPRESS_DISK_CACHE]]
                [--aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]]
                [--keep_vae_loaded [KEEP_VAE_LOADED]]
                [--skip_file_discovery SKIP_FILE_DISCOVERY]
                [--data_backend_sampling {uniform,auto-weighting}]
                [--image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE]
                [--write_batch_size WRITE_BATCH_SIZE]
                [--read_batch_size READ_BATCH_SIZE]
                [--enable_multiprocessing [ENABLE_MULTIPROCESSING]]
                [--max_workers MAX_WORKERS]
                [--aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS]
                [--torch_num_threads TORCH_NUM_THREADS]
                [--dataloader_prefetch [DATALOADER_PREFETCH]]
                [--dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN]
                [--aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT]
                [--aspect_bucket_alignment {8,16,24,32,64}]
                [--minimum_image_size MINIMUM_IMAGE_SIZE]
                [--maximum_image_size MAXIMUM_IMAGE_SIZE]
                [--target_downsample_size TARGET_DOWNSAMPLE_SIZE]
                [--max_upscale_threshold MAX_UPSCALE_THRESHOLD]
                [--metadata_update_interval METADATA_UPDATE_INTERVAL]
                [--debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]]
                [--debug_dataset_loader [DEBUG_DATASET_LOADER]]
                [--print_filenames [PRINT_FILENAMES]]
                [--print_sampler_statistics [PRINT_SAMPLER_STATISTICS]]
                [--timestep_bias_strategy {earlier,later,range,none}]
                [--timestep_bias_begin TIMESTEP_BIAS_BEGIN]
                [--timestep_bias_end TIMESTEP_BIAS_END]
                [--timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER]
                [--timestep_bias_portion TIMESTEP_BIAS_PORTION]
                [--training_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--inference_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--loss_type {l2,huber,smooth_l1}]
                [--huber_schedule {snr,exponential,constant}]
                [--huber_c HUBER_C] [--snr_gamma SNR_GAMMA]
                [--masked_loss_probability MASKED_LOSS_PROBABILITY]
                [--hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]]
                [--hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT]
                [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                [--optimizer_beta1 OPTIMIZER_BETA1]
                [--optimizer_beta2 OPTIMIZER_BETA2]
                [--optimizer_cpu_offload_method {none}]
                [--gradient_precision {unmodified,fp32}]
                [--adam_weight_decay ADAM_WEIGHT_DECAY]
                [--adam_epsilon ADAM_EPSILON] [--prodigy_steps PRODIGY_STEPS]
                [--max_grad_norm MAX_GRAD_NORM]
                [--grad_clip_method {value,norm}]
                [--optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]]
                [--fuse_optimizer [FUSE_OPTIMIZER]]
                [--optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]]
                [--push_to_hub [PUSH_TO_HUB]]
                [--push_to_hub_background [PUSH_TO_HUB_BACKGROUND]]
                [--push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]]
                [--publishing_config PUBLISHING_CONFIG]
                [--hub_model_id HUB_MODEL_ID]
                [--model_card_private [MODEL_CARD_PRIVATE]]
                [--model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]]
                [--model_card_note MODEL_CARD_NOTE]
                [--modelspec_comment MODELSPEC_COMMENT]
                [--report_to {tensorboard,wandb,comet_ml,all,none}]
                [--checkpoint_step_interval CHECKPOINT_STEP_INTERVAL]
                [--checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL]
                [--checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS]
                [--checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]]
                [--checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT]
                [--tracker_run_name TRACKER_RUN_NAME]
                [--tracker_project_name TRACKER_PROJECT_NAME]
                [--tracker_image_layout {gallery,table}]
                [--enable_watermark [ENABLE_WATERMARK]]
                [--framerate FRAMERATE]
                [--seed_for_each_device [SEED_FOR_EACH_DEVICE]]
                [--snr_weight SNR_WEIGHT]
                [--rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]]
                [--webhook_config WEBHOOK_CONFIG]
                [--webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL]
                [--distillation_method {lcm,dcm,dmd,perflow}]
                [--distillation_config DISTILLATION_CONFIG]
                [--ema_validation {none,ema_only,comparison}]
                [--local_rank LOCAL_RANK] [--ltx_train_mode {t2v,i2v}]
                [--ltx_i2v_prob LTX_I2V_PROB]
                [--ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION]
                [--ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]]
                [--offload_param_path OFFLOAD_PARAM_PATH]
                [--offset_noise [OFFSET_NOISE]]
                [--quantize_activations [QUANTIZE_ACTIVATIONS]]
                [--refiner_training [REFINER_TRAINING]]
                [--refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]]
                [--refiner_training_strength REFINER_TRAINING_STRENGTH]
                [--sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]]
                [--sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION]

The following SimpleTuner command-line options are available:

options:
  -h, --help            show this help message and exit
  --model_family {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                        The base model architecture family to train
  --model_flavour MODEL_FLAVOUR
                        Specific variant of the selected model family
  --controlnet [CONTROLNET]
                        Train ControlNet (full or LoRA) branches alongside the
                        primary network.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Optional override of the model checkpoint. Leave blank
                        to use the default path for the selected model
                        flavour.
  --output_dir OUTPUT_DIR
                        Directory where model checkpoints and logs will be
                        saved
  --logging_dir LOGGING_DIR
                        Directory for TensorBoard logs
  --model_type {full,lora}
                        Choose between full model training or LoRA adapter
                        training
  --seed SEED           Seed used for deterministic training behaviour
  --resolution RESOLUTION
                        Resolution for training images
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Select checkpoint to resume training from
  --prediction_type {epsilon,v_prediction,sample,flow_matching}
                        The parameterization type for the diffusion model
  --pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH
                        Path to pretrained VAE model
  --vae_dtype {default,fp32,fp16,bf16}
                        Precision for VAE encoding/decoding. Lower precision
                        saves memory.
  --vae_cache_ondemand [VAE_CACHE_ONDEMAND]
                        Process VAE latents during training instead of
                        precomputing them
  --vae_cache_disable [VAE_CACHE_DISABLE]
                        Implicitly enables on-demand caching and disables
                        writing embeddings to disk.
  --accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL
                        Clear the cache from VRAM every X steps to prevent
                        memory leaks
  --aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}
                        Number of decimal places to round aspect ratios to for
                        bucket creation
  --base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for loading the base model. Lower precision
                        saves memory.
  --text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL
                        Checkpoint every N transformer blocks
  --offload_during_startup [OFFLOAD_DURING_STARTUP]
                        Offload text encoders to CPU during VAE caching
  --quantize_via {cpu,accelerator,pipeline}
                        Where to perform model quantization
  --quantization_config QUANTIZATION_CONFIG
                        JSON or file path describing Diffusers quantization
                        config for pipeline quantization
  --fuse_qkv_projections [FUSE_QKV_PROJECTIONS]
                        Enables Flash Attention 3 when supported; otherwise
                        falls back to PyTorch SDPA.
  --control [CONTROL]   Enable channel-wise control style training
  --controlnet_custom_config CONTROLNET_CUSTOM_CONFIG
                        Custom configuration for ControlNet models
  --controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH
                        Path to ControlNet model weights to preload
  --tread_config TREAD_CONFIG
                        Configuration for TREAD training method
  --pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH
                        Path to pretrained transformer model
  --pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER
                        Subfolder containing transformer model weights
  --pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH
                        Path to pretrained UNet model
  --pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER
                        Subfolder containing UNet model weights
  --pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH
                        Path to pretrained T5 model
  --pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH
                        Path to pretrained Gemma model
  --revision REVISION   Git branch/tag/commit for model version
  --variant VARIANT     Model variant (e.g., fp16, bf16)
  --base_model_default_dtype {bf16,fp32}
                        Default precision for quantized base model weights
  --unet_attention_slice [UNET_ATTENTION_SLICE]
                        Enable attention slicing for SDXL UNet
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of times to iterate through the entire dataset
  --max_train_steps MAX_TRAIN_STEPS
                        Maximum number of training steps (0 = use epochs
                        instead)
  --train_batch_size TRAIN_BATCH_SIZE
                        Number of samples processed per forward/backward pass
                        (per device).
  --learning_rate LEARNING_RATE
                        Base learning rate for training
  --optimizer {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                        Optimization algorithm for training
  --optimizer_config OPTIMIZER_CONFIG
                        Comma-separated key=value pairs forwarded to the
                        selected optimizer
  --lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        How learning rate changes during training
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of steps to accumulate gradients
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps to gradually increase LR from 0
  --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                        Maximum number of checkpoints to keep on disk
  --gradient_checkpointing [GRADIENT_CHECKPOINTING]
                        Trade compute for memory during training
  --train_text_encoder [TRAIN_TEXT_ENCODER]
                        Also train the text encoder (CLIP) model
  --text_encoder_lr TEXT_ENCODER_LR
                        Separate learning rate for text encoder
  --lr_num_cycles LR_NUM_CYCLES
                        Number of cosine annealing cycles
  --lr_power LR_POWER   Power for polynomial decay scheduler
  --use_soft_min_snr [USE_SOFT_MIN_SNR]
                        Use soft clamping instead of hard clamping for Min-SNR
  --use_ema [USE_EMA]   Maintain an exponential moving average copy of the
                        model during training.
  --ema_device {accelerator,cpu}
                        Where to keep the EMA weights in-between updates.
  --ema_cpu_only [EMA_CPU_ONLY]
                        Keep EMA weights exclusively on CPU even when
                        ema_device would normally move them.
  --ema_update_interval EMA_UPDATE_INTERVAL
                        Update EMA weights every N optimizer steps
  --ema_foreach_disable [EMA_FOREACH_DISABLE]
                        Fallback to standard tensor ops instead of
                        torch.foreach updates.
  --ema_decay EMA_DECAY
                        Smoothing factor for EMA updates (closer to 1.0 =
                        slower drift).
  --lora_rank LORA_RANK
                        Dimension of LoRA update matrices
  --lora_alpha LORA_ALPHA
                        Scaling factor for LoRA updates
  --lora_type {standard,lycoris}
                        LoRA implementation type
  --lora_dropout LORA_DROPOUT
                        LoRA dropout randomly ignores neurons during training.
                        This can help prevent overfitting.
  --lora_init_type {default,gaussian,loftq,olora,pissa}
                        The initialization type for the LoRA model
  --peft_lora_mode {standard,singlora}
                        PEFT LoRA training mode
  --peft_lora_target_modules PEFT_LORA_TARGET_MODULES
                        JSON array (or path to a JSON file) listing PEFT
                        LoRA target module names. Overrides preset targets.
  --singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS
                        Number of ramp-up steps for SingLoRA
  --slider_lora_target [SLIDER_LORA_TARGET]
                        Route LoRA training to slider-friendly targets
                        (self-attn + conv/time embeddings). Only affects
                        standard PEFT LoRA.
  --init_lora INIT_LORA
                        Specify an existing LoRA or LyCORIS safetensors file
                        to initialize the adapter
  --lycoris_config LYCORIS_CONFIG
                        Path to LyCORIS configuration JSON file
  --init_lokr_norm INIT_LOKR_NORM
                        Perturbed normal initialization for LyCORIS LoKr
                        layers
  --flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}
                        Which layers to train in Flux models
  --use_dora [USE_DORA]
                        Enable DoRA (Weight-Decomposed LoRA)
  --resolution_type {pixel,area,pixel_area}
                        How to interpret the resolution value
  --data_backend_config DATA_BACKEND_CONFIG
                        Select a saved dataset configuration (managed in
                        Datasets & Environments tabs)
  --caption_strategy {filename,textfile,instance_prompt,parquet}
                        How to load captions for images
  --conditioning_multidataset_sampling {combined,random}
                        How to sample from multiple conditioning datasets
  --instance_prompt INSTANCE_PROMPT
                        Instance prompt for training
  --parquet_caption_column PARQUET_CAPTION_COLUMN
                        Column name containing captions in parquet files
  --parquet_filename_column PARQUET_FILENAME_COLUMN
                        Column name containing image paths in parquet files
  --ignore_missing_files [IGNORE_MISSING_FILES]
                        Continue training even if some files are missing
  --vae_cache_scan_behaviour {recreate,sync}
                        How to scan VAE cache for missing files
  --vae_enable_slicing [VAE_ENABLE_SLICING]
                        Enable VAE attention slicing for memory efficiency
  --vae_enable_tiling [VAE_ENABLE_TILING]
                        Enable VAE tiling for large images
  --vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]
                        Enable patch-based 3D conv for HunyuanVideo VAE to
                        reduce peak VRAM (slight slowdown)
  --vae_batch_size VAE_BATCH_SIZE
                        Batch size for VAE encoding during caching
  --caption_dropout_probability CAPTION_DROPOUT_PROBABILITY
                        Caption dropout will randomly drop captions and, for
                        SDXL, size conditioning inputs based on this
                        probability
  --tokenizer_max_length TOKENIZER_MAX_LENGTH
                        Override the tokenizer sequence length (advanced).
  --validation_step_interval VALIDATION_STEP_INTERVAL
                        Run validation every N training steps (deprecated alias: --validation_steps)
  --validation_epoch_interval VALIDATION_EPOCH_INTERVAL
                        Run validation every N training epochs
  --disable_benchmark [DISABLE_BENCHMARK]
                        Skip generating baseline comparison images before
                        training starts
  --validation_prompt VALIDATION_PROMPT
                        Prompt to use for validation images
  --num_validation_images NUM_VALIDATION_IMAGES
                        Number of images to generate per validation
  --num_eval_images NUM_EVAL_IMAGES
                        Number of images to generate for evaluation metrics
  --eval_steps_interval EVAL_STEPS_INTERVAL
                        Run evaluation every N training steps
  --eval_epoch_interval EVAL_EPOCH_INTERVAL
                        Run evaluation every N training epochs (decimals run
                        multiple times per epoch)
  --eval_timesteps EVAL_TIMESTEPS
                        Number of timesteps for evaluation
  --eval_dataset_pooling [EVAL_DATASET_POOLING]
                        Combine evaluation metrics from all datasets into a
                        single chart
  --evaluation_type {none,clip}
                        Type of evaluation metrics to compute
  --pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH
                        Path to pretrained model for evaluation metrics
  --validation_guidance VALIDATION_GUIDANCE
                        CFG guidance scale for validation images
  --validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS
                        Number of diffusion steps for validation renders
  --validation_on_startup [VALIDATION_ON_STARTUP]
                        Run validation on the base model before training
                        starts
  --validation_using_datasets [VALIDATION_USING_DATASETS]
                        Use random images from training datasets for
                        validation
  --validation_torch_compile [VALIDATION_TORCH_COMPILE]
                        Use torch.compile() on validation pipeline for speed
  --validation_guidance_real VALIDATION_GUIDANCE_REAL
                        CFG value for distilled models (e.g., FLUX schnell)
  --validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP
                        Skip CFG for initial timesteps (Flux only)
  --validation_negative_prompt VALIDATION_NEGATIVE_PROMPT
                        Negative prompt for validation images
  --validation_randomize [VALIDATION_RANDOMIZE]
                        Use random seeds for each validation
  --validation_seed VALIDATION_SEED
                        Fixed seed for reproducible validation images
  --validation_disable [VALIDATION_DISABLE]
                        Completely disable validation image generation
  --validation_prompt_library [VALIDATION_PROMPT_LIBRARY]
                        Use SimpleTuner's built-in prompt library
  --user_prompt_library USER_PROMPT_LIBRARY
                        Path to custom JSON prompt library
  --eval_dataset_id EVAL_DATASET_ID
                        Specific dataset to use for evaluation metrics
  --validation_stitch_input_location {left,right}
                        Where to place input image in img2img validations
  --validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE
                        CFG rescale value for validation
  --validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]
                        Disable unconditional image generation during
                        validation
  --validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS
                        JSON list of transformer layers to skip during
                        classifier-free guidance
  --validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START
                        Starting layer index to skip guidance
  --validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP
                        Ending layer index to skip guidance
  --validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE
                        Scale guidance strength when applying layer skipping
  --validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH
                        Strength multiplier for LyCORIS validation
  --validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}
                        Noise scheduler for validation
  --validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES
                        Number of frames for video validation
  --validation_audio_only [VALIDATION_AUDIO_ONLY]
                        Disable video generation during validation and emit
                        audio only
  --validation_resolution VALIDATION_RESOLUTION
                        Override resolution for validation images (pixels or
                        megapixels)
  --validation_seed_source {cpu,gpu}
                        Source device used to generate validation seeds
  --i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]
                        Unlock experimental overrides and bypass built-in
                        safety limits.
  --flow_sigmoid_scale FLOW_SIGMOID_SCALE
                        Scale factor for sigmoid timestep sampling for flow-
                        matching models.
  --flux_fast_schedule [FLUX_FAST_SCHEDULE]
                        Use experimental fast schedule for Flux training
  --flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]
                        Use uniform schedule instead of sigmoid for flow-
                        matching
  --flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]
                        Use beta schedule instead of sigmoid for flow-matching
  --flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA
                        Alpha value for beta schedule (default: 2.0)
  --flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA
                        Beta value for beta schedule (default: 2.0)
  --flow_schedule_shift FLOW_SCHEDULE_SHIFT
                        Shift the noise schedule for flow-matching models
  --flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]
                        Auto-adjust schedule shift based on image resolution
  --flux_guidance_mode {constant,random-range}
                        Guidance mode for Flux training
  --flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]
                        Enable attention masked training for Flux models
  --flux_guidance_value FLUX_GUIDANCE_VALUE
                        Guidance value for constant mode
  --flux_guidance_min FLUX_GUIDANCE_MIN
                        Minimum guidance value for random-range mode
  --flux_guidance_max FLUX_GUIDANCE_MAX
                        Maximum guidance value for random-range mode
  --t5_padding {zero,unmodified}
                        Padding behavior for T5 text encoder
  --sd3_clip_uncond_behaviour {empty_string,zero}
                        How SD3 handles unconditional prompts
  --sd3_t5_uncond_behaviour {empty_string,zero}
                        How SD3 T5 handles unconditional prompts
  --soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA
                        Sigma data for soft min SNR weighting
  --mixed_precision {no,fp16,bf16,fp8}
                        Precision for training computations
  --attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}
                        Attention computation backend
  --sageattention_usage {training,inference,training+inference}
                        When to use SageAttention
  --disable_tf32 [DISABLE_TF32]
                        Force IEEE FP32 precision (disables TF32) using
                        PyTorch's fp32_precision controls when available
  --set_grads_to_none [SET_GRADS_TO_NONE]
                        Set gradients to None instead of zero
  --noise_offset NOISE_OFFSET
                        Add noise offset to training
  --noise_offset_probability NOISE_OFFSET_PROBABILITY
                        Probability of applying noise offset
  --input_perturbation INPUT_PERTURBATION
                        Add additional noise only to the inputs fed to the
                        model during training
  --input_perturbation_steps INPUT_PERTURBATION_STEPS
                        Only apply input perturbation over the first N steps
                        with linear decay
  --lr_end LR_END       A polynomial learning rate will end up at this value
                        after the specified number of warmup steps
  --lr_scale [LR_SCALE]
                        Scale the learning rate by the number of GPUs,
                        gradient accumulation steps, and batch size
  --lr_scale_sqrt [LR_SCALE_SQRT]
                        If using --lr_scale, use the square root of (number of
                        GPUs * gradient accumulation steps * batch size)
  --ignore_final_epochs [IGNORE_FINAL_EPOCHS]
                        When provided, the max epoch counter will not
                        determine the end of the training run
  --freeze_encoder_before FREEZE_ENCODER_BEFORE
                        When using 'before' strategy, we will freeze layers
                        earlier than this
  --freeze_encoder_after FREEZE_ENCODER_AFTER
                        When using 'after' strategy, we will freeze layers
                        later than this
  --freeze_encoder_strategy {before,between,after}
                        When freezing the text encoder, we can use the
                        'before', 'between', or 'after' strategy
  --layer_freeze_strategy {none,bitfit}
                        When freezing parameters, we can use the 'none' or
                        'bitfit' strategy
  --fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]
                        If set, will fully unload the text_encoder from memory
                        when not in use
  --save_text_encoder [SAVE_TEXT_ENCODER]
                        If set, will save the text encoder after training
  --text_encoder_limit TEXT_ENCODER_LIMIT
                        When training the text encoder, we want to limit how
                        long it trains for to avoid catastrophic loss
  --prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]
                        When determining the captions from the filename,
                        prepend the instance prompt as an enforced keyword
  --only_instance_prompt [ONLY_INSTANCE_PROMPT]
                        Use the instance prompt instead of the caption from
                        filename
  --data_aesthetic_score DATA_AESTHETIC_SCORE
                        Since currently we do not calculate aesthetic scores
                        for data, we will statically set it to one value. This
                        is only used by the SDXL Refiner
  --delete_unwanted_images [DELETE_UNWANTED_IMAGES]
                        If set, will delete images that are not of a minimum
                        size to save on disk space for large training runs
  --delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]
                        If set, any images that error out during load will be
                        removed from the underlying storage medium
  --disable_bucket_pruning [DISABLE_BUCKET_PRUNING]
                        When training on very small datasets, you might not
                        care that the batch sizes will outpace your image
                        count. Setting this option will prevent SimpleTuner
                        from deleting your bucket lists that do not meet the
                        minimum image count requirements. Use at your own
                        risk, it may end up throwing off your statistics or
                        epoch tracking
  --disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]
                        By default, the timestep schedule is divided into
                        roughly `train_batch_size` number of segments, and
                        then each of those are sampled from separately. This
                        improves the selection distribution, but may not be
                        desired in certain training scenarios, eg. when
                        limiting the timestep selection range
  --preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]
                        For very large cloud storage buckets that will never
                        change, enabling this option will prevent the trainer
                        from scanning it at startup, by preserving the cache
                        files that we generate. Be careful when using this,
                        as, switching datasets can result in the preserved
                        cache being used, which would be problematic.
                        Currently, cache is not stored in the dataset itself
                        but rather, locally. This may change in a future
                        release
  --override_dataset_config [OVERRIDE_DATASET_CONFIG]
                        When provided, the dataset's config will not be
                        checked against the live backend config
  --cache_dir CACHE_DIR
                        The directory where the downloaded models and datasets
                        will be stored
  --cache_dir_text CACHE_DIR_TEXT
                        This is the path to a local directory that will
                        contain your text embed cache
  --cache_dir_vae CACHE_DIR_VAE
                        This is the path to a local directory that will
                        contain your VAE outputs
  --compress_disk_cache [COMPRESS_DISK_CACHE]
                        If set, will gzip-compress the disk cache for Pytorch
                        files. This will save substantial disk space, but may
                        slow down the training process
  --aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]
                        When using a randomised aspect bucket list, the VAE
                        and aspect cache are rebuilt on each epoch. With a
                        large and diverse enough dataset, rebuilding the
                        aspect list may take a long time, and this may be
                        undesirable. This option will not override
                        vae_cache_clear_each_epoch. If both options are
                        provided, only the VAE cache will be rebuilt
  --keep_vae_loaded [KEEP_VAE_LOADED]
                        If set, will keep the VAE loaded in memory. This can
                        reduce disk churn, but consumes VRAM during the
                        forward pass
  --skip_file_discovery SKIP_FILE_DISCOVERY
                        Comma-separated values of which stages to skip
                        discovery for. Skipping any stage will speed up
                        resumption, but will increase the risk of errors, as
                        missing images or incorrectly bucketed images may not
                        be caught. Valid options: aspect, vae, text, metadata
  --data_backend_sampling {uniform,auto-weighting}
                        When using multiple data backends, the sampling
                        weighting can be set to 'uniform' or 'auto-weighting'
  --image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE
                        When resizing and cropping images, we do it in
                        parallel using processes or threads. This defines how
                        many images will be read into the queue before they
                        are processed
  --write_batch_size WRITE_BATCH_SIZE
                        When using certain storage backends, it is better to
                        batch smaller writes rather than continuous
                        dispatching. In SimpleTuner, write batching is
                        currently applied during VAE caching, when many small
                        objects are written. This mostly applies to S3, but
                        some shared server filesystems may benefit as well.
                        Default: 64
  --read_batch_size READ_BATCH_SIZE
                        Used by the VAE cache to prefetch image data. This is
                        the number of images to read ahead
  --enable_multiprocessing [ENABLE_MULTIPROCESSING]
                        If set, will use processes instead of threads during
                        metadata caching operations
  --max_workers MAX_WORKERS
                        How many active threads or processes to run during VAE
                        caching
  --aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS
                        When using AWS backends, the maximum number of
                        connections to keep open to the S3 bucket at a single
                        time
  --torch_num_threads TORCH_NUM_THREADS
                        The number of threads to use for PyTorch operations.
                        This is not the same as the number of workers
  --dataloader_prefetch [DATALOADER_PREFETCH]
                        When provided, the dataloader will read-ahead and
                        attempt to retrieve latents, text embeds, and other
                        metadata ahead of the time when the batch is required,
                        so that it can be immediately available
  --dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN
                        Set the number of prefetched batches
  --aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT
                        The number of workers to use for aspect bucketing.
                        This is a CPU-bound task, so the number of workers
                        should be set to the number of CPU threads available.
                        If you use an I/O bound backend, an even higher value
                        may make sense. Default: 12
  --aspect_bucket_alignment {8,16,24,32,64}
                        When training diffusion models, the image sizes
                        generally must align to a 64 pixel interval
  --minimum_image_size MINIMUM_IMAGE_SIZE
                        The minimum resolution for both sides of input images
  --maximum_image_size MAXIMUM_IMAGE_SIZE
                        When cropping images that are excessively large, the
                        entire scene context may be lost, eg. the crop might
                        just end up being a portion of the background. To
                        avoid this, a maximum image size may be provided,
                        which will result in very-large images being
                        downsampled before cropping them. This value uses
                        --resolution_type to determine whether it is a pixel
                        edge or megapixel value
  --target_downsample_size TARGET_DOWNSAMPLE_SIZE
                        When using --maximum_image_size, very-large images
                        exceeding that value will be downsampled to this
                        target size before cropping
  --max_upscale_threshold MAX_UPSCALE_THRESHOLD
                        Limit upscaling of small images to prevent quality
                        degradation (opt-in). When set, filters out aspect
                        buckets requiring upscaling beyond this threshold.
                        For example, 0.2 allows up to 20% upscaling. Default
                        (None) allows unlimited upscaling. Must be between 0
                        and 1.
  --metadata_update_interval METADATA_UPDATE_INTERVAL
                        When generating the aspect bucket indicies, we want to
                        save it every X seconds
  --debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]
                        If set, will print excessive debugging for aspect
                        bucket operations
  --debug_dataset_loader [DEBUG_DATASET_LOADER]
                        If set, will print excessive debugging for data loader
                        operations
  --print_filenames [PRINT_FILENAMES]
                        If any image files are stopping the process eg. due to
                        corruption or truncation, this will help identify
                        which is at fault
  --print_sampler_statistics [PRINT_SAMPLER_STATISTICS]
                        If provided, will print statistics about the dataset
                        sampler. This is useful for debugging
  --timestep_bias_strategy {earlier,later,range,none}
                        Strategy for biasing timestep sampling
  --timestep_bias_begin TIMESTEP_BIAS_BEGIN
                        Beginning of timestep bias range
  --timestep_bias_end TIMESTEP_BIAS_END
                        End of timestep bias range
  --timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER
                        Multiplier for timestep bias probability
  --timestep_bias_portion TIMESTEP_BIAS_PORTION
                        Portion of training steps to apply timestep bias
  --training_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for training scheduler
  --inference_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for inference scheduler
  --loss_type {l2,huber,smooth_l1}
                        Loss function for training
  --huber_schedule {snr,exponential,constant}
                        Schedule for Huber loss transition threshold
  --huber_c HUBER_C     Transition point between L2 and L1 regions for Huber
                        loss
  --snr_gamma SNR_GAMMA
                        SNR weighting gamma value (0 = disabled)
  --masked_loss_probability MASKED_LOSS_PROBABILITY
                        Probability of applying masked loss weighting per
                        batch
  --hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]
                        Apply experimental load balancing loss when training
                        HiDream models.
  --hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT
                        Strength multiplier for HiDream load balancing loss.
  --adam_beta1 ADAM_BETA1
                        First moment decay rate for Adam optimizers
  --adam_beta2 ADAM_BETA2
                        Second moment decay rate for Adam optimizers
  --optimizer_beta1 OPTIMIZER_BETA1
                        First moment decay rate for optimizers
  --optimizer_beta2 OPTIMIZER_BETA2
                        Second moment decay rate for optimizers
  --optimizer_cpu_offload_method {none}
                        Method for CPU offloading optimizer states
  --gradient_precision {unmodified,fp32}
                        Precision for gradient computation
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        L2 regularisation strength for Adam-family optimizers.
  --adam_epsilon ADAM_EPSILON
                        Small constant added for numerical stability.
  --prodigy_steps PRODIGY_STEPS
                        Number of steps Prodigy should spend adapting its
                        learning rate.
  --max_grad_norm MAX_GRAD_NORM
                        Gradient clipping threshold to prevent exploding
                        gradients.
  --grad_clip_method {value,norm}
                        Strategy for applying max_grad_norm during clipping.
  --optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]
                        Move optimizer gradients to CPU to save GPU memory.
  --fuse_optimizer [FUSE_OPTIMIZER]
                        Enable fused kernels when offloading to reduce memory
                        overhead.
  --optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]
                        Free gradient tensors immediately after optimizer step
                        when using Optimi optimizers.
  --push_to_hub [PUSH_TO_HUB]
                        Automatically upload the trained model to your Hugging
                        Face Hub repository.
  --push_to_hub_background [PUSH_TO_HUB_BACKGROUND]
                        Run Hub uploads in a background worker so training is
                        not blocked while pushing.
  --push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]
                        Upload intermediate checkpoints to the same Hugging
                        Face repository during training.
  --publishing_config PUBLISHING_CONFIG
                        Optional JSON/file path describing additional
                        publishing targets (S3/Backblaze B2/Azure Blob/Dropbox).
  --hub_model_id HUB_MODEL_ID
                        If left blank, SimpleTuner derives a name from the
                        project settings when pushing to Hub.
  --model_card_private [MODEL_CARD_PRIVATE]
                        Create the Hugging Face repository as private instead
                        of public.
  --model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]
                        Remove the default NSFW warning from the generated
                        model card on Hugging Face Hub.
  --model_card_note MODEL_CARD_NOTE
                        Optional note that appears at the top of the generated
                        model card.
  --modelspec_comment MODELSPEC_COMMENT
                        Text embedded in safetensors file metadata as
                        modelspec.comment, visible in external model viewers.
  --report_to {tensorboard,wandb,comet_ml,all,none}
                        Where to log training metrics
  --checkpoint_step_interval CHECKPOINT_STEP_INTERVAL
                        Save model checkpoint every N steps (deprecated alias: --checkpointing_steps)
  --checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL
                        Save model checkpoint every N epochs
  --checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS
                        Rolling checkpoint window size for continuous
                        checkpointing
  --checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]
                        Use temporary directory for checkpoint files before
                        final save
  --checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT
                        Maximum number of rolling checkpoints to keep
  --tracker_run_name TRACKER_RUN_NAME
                        Name for this training run in tracking platforms
  --tracker_project_name TRACKER_PROJECT_NAME
                        Project name in tracking platforms
  --tracker_image_layout {gallery,table}
                        How validation images are displayed in trackers
  --enable_watermark [ENABLE_WATERMARK]
                        Add invisible watermark to generated images
  --framerate FRAMERATE
                        Framerate for video model training
  --seed_for_each_device [SEED_FOR_EACH_DEVICE]
                        Use a unique deterministic seed per GPU instead of
                        sharing one seed across devices.
  --snr_weight SNR_WEIGHT
                        Weight factor for SNR-based loss scaling
  --rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]
                        Rescale betas for zero terminal SNR
  --webhook_config WEBHOOK_CONFIG
                        Path to webhook configuration file
  --webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL
                        Interval for webhook reports (seconds)
  --distillation_method {lcm,dcm,dmd,perflow}
                        Method for model distillation
  --distillation_config DISTILLATION_CONFIG
                        Path to distillation configuration file
  --ema_validation {none,ema_only,comparison}
                        Control how EMA weights are used during validation
                        runs.
  --local_rank LOCAL_RANK
                        Local rank for distributed training
  --ltx_train_mode {t2v,i2v}
                        Training mode for LTX models
  --ltx_i2v_prob LTX_I2V_PROB
                        Probability of using image-to-video training for LTX
  --ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION
                        Fraction of noise to add for LTX partial training
  --ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]
                        Protect the first frame from noise in LTX training
  --offload_param_path OFFLOAD_PARAM_PATH
                        Path to offloaded parameter files
  --offset_noise [OFFSET_NOISE]
                        Enable offset-noise training
  --quantize_activations [QUANTIZE_ACTIVATIONS]
                        Quantize model activations during training
  --refiner_training [REFINER_TRAINING]
                        Enable refiner model training mode
  --refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]
                        Invert the noise schedule for refiner training
  --refiner_training_strength REFINER_TRAINING_STRENGTH
                        Strength of refiner training
  --sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]
                        Use full timestep range for SDXL refiner
  --sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION
                        Complex human instruction for Sana model training
```
