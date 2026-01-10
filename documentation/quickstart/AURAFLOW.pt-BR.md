## Guia de Início Rápido do Auraflow

Neste exemplo, vamos treinar um Lycoris LoKr para o Auraflow.

O fine-tuning completo deste modelo vai exigir muita VRAM devido aos 6B parâmetros, e você precisará usar [DeepSpeed](../DEEPSPEED.md) para isso funcionar.

### Requisitos de hardware

Auraflow v0.3 foi lançado como um MMDiT de 6B parâmetros que usa Pile T5 para sua representação de texto codificada e o VAE SDXL de 4 canais para sua representação latente de imagem.

Este modelo é um pouco lento para inferência, mas treina em uma velocidade razoável.

### Offload de memória (opcional)

Auraflow se beneficia muito do novo caminho de offload em grupo. Adicione o seguinte aos seus flags de treino se você estiver limitado a uma única GPU de 24G (ou menor):

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams são automaticamente desativados em backends não-CUDA, então o comando é seguro para reutilizar em ROCm e MPS.
- Não combine isso com `--enable_model_cpu_offload`.
- Offload para disco troca throughput por menor pressão de RAM no host; mantenha em um SSD local para melhores resultados.

### Pré-requisitos

Garanta que você tenha Python instalado; o SimpleTuner vai bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem o Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

```bash
apt -y install python3.12 python3.12-venv
```

#### Dependências da imagem de contêiner

Para Vast, RunPod e TensorDock (entre outros), o seguinte funciona em uma imagem CUDA 12.2-12.8 para habilitar a compilação de extensões CUDA:

```bash
apt -y install nvidia-cuda-toolkit
```

### Instalação

Instale o SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

### Configurando o ambiente

Para rodar o SimpleTuner, você precisa configurar um arquivo de configuração, os diretórios de dataset e modelos, e um arquivo de configuração do dataloader.

#### Arquivo de configuração

Um script experimental, `configure.py`, pode permitir que você pule esta seção inteiramente por meio de uma configuração interativa passo a passo. Ele contém alguns recursos de segurança que ajudam a evitar armadilhas comuns.

**Nota:** Isso não configura seu dataloader. Você ainda precisará fazer isso manualmente depois.

Para executá-lo:

```bash
simpletuner configure
```

> ⚠️ Para usuários localizados em países onde o Hugging Face Hub não é facilmente acessível, você deve adicionar `HF_ENDPOINT=https://hf-mirror.com` ao seu `~/.bashrc` ou `~/.zshrc` dependendo de qual `$SHELL` seu sistema usa.

Se você preferir configurar manualmente:

Copie `config/config.json.example` para `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Lá, você provavelmente precisará modificar as seguintes variáveis:

- `model_type` - Defina como `lora`.
- `lora_type` - Defina como `lycoris`.
- `model_family` - Defina como `auraflow`.
- `model_flavour` - Defina como `pony`, ou deixe sem definir para usar o modelo padrão.
- `output_dir` - Defina como o diretório onde você quer armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - 1 a 4 devem funcionar para uma placa de 24G.
- `validation_resolution` - Defina como `1024x1024` ou uma das outras resoluções suportadas pelo Auraflow.
  - Outras resoluções podem ser especificadas usando vírgulas: `1024x1024,1280x768,1536x1536`
  - Note que as embeddings posicionais do Auraflow são um pouco estranhas e treinar com imagens multi-escala (múltiplas resoluções base) tem resultado incerto.
- `validation_guidance` - Use o que você costuma selecionar na inferência para Auraflow; um valor mais baixo em torno de 3.5-4.0 gera resultados mais realistas
- `validation_num_inference_steps` - Use algo em torno de 30-50
- `use_ema` - definir como `true` vai ajudar bastante a obter um resultado mais suavizado junto do seu checkpoint principal treinado.

- `optimizer` - Você pode usar qualquer otimizador com o qual se sinta confortável e familiarizado, mas usaremos `optimi-lion` neste exemplo.
  - O autor do Pony Flow recomenda usar `adamw_bf16` para ter menos problemas e resultados de treino mais estáveis e confiáveis
  - Estamos usando Lion nesta demonstração para ajudar você a ver o modelo treinar mais rapidamente, mas para execuções longas, `adamw_bf16` é uma aposta segura.
- `learning_rate` - Para o otimizador Lion com Lycoris LoKr, um valor de `4e-5` é um bom ponto de partida.
  - Se você escolheu `adamw_bf16`, o LR deve ser cerca de 10x maior do que isso, ou `2.5e-4`
  - Ranks menores de Lycoris/LoRA exigem **taxas de aprendizado maiores** e Lycoris/LoRA maiores exigem **taxas menores**
- `mixed_precision` - É recomendado definir como `bf16` para a configuração de treino mais eficiente, ou `no` para melhores resultados (mas vai consumir mais memória e ser mais lento).
- `gradient_checkpointing` - Desativar isso será o mais rápido, mas limita seus tamanhos de lote. É obrigatório habilitar para obter o menor uso de VRAM.

O impacto dessas opções ainda é desconhecido.

Seu config.json vai parecer com algo assim no final:

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "validation_torch_compile": "false",
    "validation_step_interval": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 2.0,
    "validation_guidance_rescale": "0.0",
    "vae_cache_ondemand": true,
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-auraflow",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "auraflow",
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/auraflow/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

E um arquivo simples `config/lycoris_config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 8
            },
        }
    }
}
```
</details>

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

#### Prompts de validação

Dentro de `config/config.json` está o "prompt de validação principal", que normalmente é o instance_prompt que você está treinando para seu único sujeito ou estilo. Além disso, um arquivo JSON pode ser criado contendo prompts extras para rodar durante as validações.

O arquivo de exemplo `config/user_prompt_library.json.example` contém o seguinte formato:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

Os nicknames são o nome do arquivo para a validação, então mantenha-os curtos e compatíveis com seu sistema de arquivos.

Para apontar o trainer para essa biblioteca de prompts, adicione ao TRAINER_EXTRA_ARGS uma nova linha no final de `config.json`:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

Um conjunto de prompts diversos ajudará a determinar se o modelo está colapsando conforme treina. Neste exemplo, a palavra `<token>` deve ser substituída pelo nome do seu sujeito (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

> ℹ️ Auraflow usa 128 tokens por padrão e então trunca.

#### Rastreamento de score CLIP

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre como configurar e interpretar scores CLIP.

</details>

# Perda de avaliação estável

Se você quiser usar perda MSE estável para pontuar o desempenho do modelo, veja [este documento](../evaluation/EVAL_LOSS.md) para informações sobre como configurar e interpretar a perda de avaliação.

#### Prévias de validação

SimpleTuner suporta streaming de prévias intermediárias de validação durante a geração usando modelos Tiny AutoEncoder. Isso permite ver imagens de validação sendo geradas passo a passo em tempo real via callbacks de webhook.

Para habilitar:
<details>
<summary>Ver exemplo de config</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**Requisitos:**
- Configuração de webhook
- Validação habilitada

Defina `validation_preview_steps` para um valor maior (por exemplo, 3 ou 5) para reduzir o overhead do Tiny AutoEncoder. Com `validation_num_inference_steps=20` e `validation_preview_steps=5`, você receberá imagens de prévia nos steps 5, 10, 15 e 20.

#### Ajuste de schedule de flow

Modelos de flow matching como OmniGen, Sana, Flux e SD3 têm uma propriedade chamada "shift" que permite deslocar a parte treinada do schedule de timesteps usando um valor decimal simples.

##### Auto-shift

Uma abordagem geralmente recomendada é seguir vários trabalhos recentes e habilitar o shift de timesteps dependente de resolução, `--flow_schedule_auto_shift`, que usa valores de shift maiores para imagens maiores e menores para imagens menores. Isso resulta em treinamentos estáveis, mas potencialmente medianos.

##### Especificação manual

_Agradecimentos ao General Awareness no Discord pelos exemplos a seguir_

Ao usar um valor de `--flow_schedule_shift` de 0.1 (um valor muito baixo), apenas os detalhes mais finos da imagem são afetados:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Ao usar um valor de `--flow_schedule_shift` de 4.0 (um valor muito alto), os grandes recursos de composição e possivelmente o espaço de cor do modelo são impactados:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### Considerações sobre o dataset

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisa garantir que seu dataset seja grande o bastante para treinar de forma eficaz. Note que o tamanho mínimo de dataset é `train_batch_size * gradient_accumulation_steps` além de ser maior que `vae_batch_size`. O dataset não será utilizável se for pequeno demais.

> ℹ️ Com poucas imagens, você pode ver a mensagem **no images detected in dataset** - aumentar o valor de `repeats` vai superar essa limitação.

Dependendo do dataset que você tem, será necessário configurar seu diretório de dataset e o arquivo de configuração do dataloader de forma diferente. Neste exemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo isto:

<details>
<summary>Ver exemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-auraflow",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/auraflow/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/auraflow/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/auraflow",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> ℹ️ Use `caption_strategy=textfile` se você tiver arquivos `.txt` contendo legendas.
> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Depois, crie um diretório `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

Isso vai baixar cerca de 10k amostras de fotografias para o diretório `datasets/pseudo-camera-10k`, que será criado automaticamente.

Suas imagens de Dreambooth devem ir para o diretório `datasets/dreambooth-subject`.

#### Login no WandB e Huggingface Hub

Você vai querer fazer login no WandB e no HF Hub antes de iniciar o treinamento, especialmente se estiver usando `--push_to_hub` e `--report_to=wandb`.

Se você pretende enviar itens para um repositório Git LFS manualmente, também deve executar `git config --global credential.helper store`.

Execute os seguintes comandos:

```bash
wandb login
```

e

```bash
huggingface-cli login
```

Siga as instruções para fazer login em ambos os serviços.

### Executando o treinamento

A partir do diretório do SimpleTuner, você tem várias opções para iniciar o treinamento:

**Opção 1 (Recomendado - pip install):**
```bash
pip install 'simpletuner[cuda]'
simpletuner train
```

**Opção 2 (Método Git clone):**
```bash
simpletuner train
```

**Opção 3 (Método legado - ainda funciona):**
```bash
./train.sh
```

Isso vai iniciar o cache em disco das embeddings de texto e saídas do VAE.

Para mais informações, veja os documentos do [dataloader](../DATALOADER.md) e do [tutorial](../TUTORIAL.md).

### Rodando inferência no LoKr depois

Como é um modelo novo, o exemplo precisará de alguns ajustes para funcionar. Aqui vai um exemplo funcional:

<details>
<summary>Mostrar exemplo de inferência em Python</summary>

```py
import torch
from helpers.models.auraflow.pipeline import AuraFlowPipeline
from helpers.models.auraflow.transformer import AuraFlowTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

model_id = 'terminusresearch/auraflow-v0.3'
adapter_repo_id = 'bghira/auraflow-photo-1mp-Prodigy'
adapter_filename = 'pytorch_lora_weights.safetensors'

def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
transformer = AuraFlowTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = AuraFlowPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)
lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

prompt = "Place your test prompt here."
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
t5_embeds, negative_t5_embeds, attention_mask, negative_attention_mask = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# We'll nuke the text encoders to save memory.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
model_output = pipeline(
    prompt_embeds=t5_embeds,
    prompt_attention_mask=attention_mask,
    negative_prompt_embeds=negative_t5_embeds,
    negative_prompt_attention_mask=negative_attention_mask,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```
</details>

## Notas e dicas de troubleshooting

### Configuração com menor VRAM

A configuração de menor VRAM para Auraflow é cerca de 20-22G:

- SO: Ubuntu Linux 24
- GPU: um único dispositivo NVIDIA CUDA (10G, 12G)
- Memória do sistema: aproximadamente 50G (pode ser mais, pode ser menos)
- Precisão do modelo base:
  - Para sistemas Apple e AMD, `int8-quanto` (ou `fp8-torchao`, `int8-torchao` seguem perfis de uso de memória similares)
    - `int4-quanto` também funciona, mas pode ter menor precisão / piores resultados
  - Para sistemas NVIDIA, `nf4-bnb` funciona bem, mas será mais lento que `int8-quanto`
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolução: 1024px
- Tamanho de lote: 1, zero passos de acumulação de gradiente
- DeepSpeed: desativado / não configurado
- PyTorch: 2.7+
- Usar `--quantize_via=cpu` para evitar outOfMemory na inicialização em placas <=16G.
- Habilite `--gradient_checkpointing`
- Use uma configuração LoRA ou Lycoris bem pequena (ex.: LoRA rank 1 ou Lokr factor 25)

**NOTA**: O pré-cache de embeddings do VAE e saídas do text encoder pode usar mais memória e ainda dar OOM. VAE tiling e slicing são ativados por padrão. Se você ver OOM, talvez precise habilitar `offload_during_startup=true`; caso contrário, pode ser apenas falta de memória.

A velocidade foi aproximadamente 3 iterações por segundo em uma NVIDIA 4090 usando PyTorch 2.7 e CUDA 12.8.

### Perda com máscara

Se você está treinando um sujeito ou estilo e gostaria de mascarar um ou outro, veja a seção [treino com loss mascarada](../DREAMBOOTH.md#masked-loss) do guia de Dreambooth.

### Quantização

Auraflow tende a responder bem até o nível de precisão `int4`, embora `int8` seja o ponto ideal para qualidade e estabilidade se você não puder usar `bf16`.

### Taxas de aprendizado

#### LoRA (--lora_type=standard)

*Não suportado.*

#### LoKr (--lora_type=lycoris)
- Taxas de aprendizado mais suaves são melhores para LoKr (`1e-4` com AdamW, `2e-5` com Lion)
- Outros algoritmos precisam de mais exploração.
- Definir `is_regularisation_data` tem impacto/efeito desconhecido com Auraflow (não testado, mas deve ficar tudo bem?)

### Artefatos de imagem

Auraflow tem resposta desconhecida a artefatos de imagem, embora use o VAE do Flux e tenha limitações semelhantes em detalhes finos.

Se surgirem problemas de qualidade de imagem, por favor abra uma issue no GitHub.

### Bucketização de aspecto

Algumas limitações na implementação do patch embed do modelo significam que há certas resoluções que causarão erro.

Experimentação será útil, assim como reports de bug detalhados.

### Ajuste full-rank

DeepSpeed vai usar MUITA memória de sistema com Auraflow, e o ajuste completo pode não performar como você espera em termos de aprendizado de conceitos ou evitar colapso do modelo.

Lycoris LoKr é recomendado no lugar do ajuste full-rank, pois é mais estável e tem menor uso de memória.
