## Guia de Início Rápido do HiDream

Neste exemplo, vamos treinar um Lycoris LoKr para HiDream, na esperança de termos memória suficiente para isso.

Uma GPU de 24G provavelmente é o mínimo viável sem offload extensivo de blocos e fused backward pass. Um Lycoris LoKr funciona tão bem quanto!

### Requisitos de hardware

O HiDream tem 17B de parâmetros no total, com ~8B ativos a qualquer momento usando um gate MoE aprendido para distribuir o trabalho. Ele usa **quatro** text encoders e o VAE do Flux.

No geral, o modelo sofre com complexidade arquitetural e parece ser um derivado do Flux Dev, seja por destilação direta ou por fine-tuning contínuo, evidente em algumas amostras de validação que parecem compartilhar os mesmos pesos.

### Pré-requisitos

Certifique-se de que você tenha Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

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
pip install simpletuner[cuda]
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

### Configurando o ambiente

Para rodar o SimpleTuner, você precisará configurar um arquivo de configuração, os diretórios de dataset e modelo, e um arquivo de configuração do dataloader.

#### Arquivo de configuração

Um script experimental, `configure.py`, pode permitir que você pule esta seção inteiramente por meio de uma configuração interativa passo a passo. Ele contém alguns recursos de segurança que ajudam a evitar armadilhas comuns.

**Nota:** Isso não configura seu dataloader. Você ainda terá que fazer isso manualmente, mais tarde.

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

Lá, você possivelmente precisará modificar as seguintes variáveis:

- `model_type` - Defina como `lora`.
- `lora_type` - Defina como `lycoris`.
- `model_family` - Defina como `hidream`.
- `model_flavour` - Defina como `full`, porque `dev` foi destilado de um jeito que não é facilmente treinável diretamente, a menos que você queira ir até o fim e quebrar sua destilação.
  - Na verdade, o modelo `full` também é difícil de treinar, mas é o único que não foi destilado.
- `output_dir` - Defina o diretório onde deseja armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - 1, talvez?.
- `validation_resolution` - Defina como `1024x1024` ou uma das outras resoluções suportadas pelo HiDream.
  - Outras resoluções podem ser especificadas separando por vírgulas: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Use qualquer valor que você costuma selecionar em inferência para HiDream; um valor mais baixo em torno de 2.5-3.0 gera resultados mais realistas
- `validation_num_inference_steps` - Use algo em torno de 30
- `use_ema` - Definir como `true` ajuda muito a obter um resultado mais suavizado junto com o seu checkpoint principal.

- `optimizer` - Você pode usar qualquer otimizador que conheça e com o qual se sinta confortável, mas usaremos `optimi-lion` neste exemplo.
- `mixed_precision` - É recomendado definir como `bf16` para a configuração de treinamento mais eficiente, ou `no` (mas consumirá mais memória e será mais lento).
- `gradient_checkpointing` - Desativar isso será o mais rápido, mas limita seus tamanhos de batch. É necessário habilitar isso para obter o menor uso de VRAM.

Algumas opções avançadas do HiDream podem ser definidas para incluir loss auxiliar de MoE durante o treinamento. Ao adicionar a loss MoE, o valor será naturalmente muito mais alto do que o normal.

- `hidream_use_load_balancing_loss` - Defina como `true` para habilitar a loss de balanceamento de carga.
- `hidream_load_balancing_loss_weight` - Esta é a magnitude da loss auxiliar. Um valor de `0.01` é o padrão, mas você pode definir `0.1` ou `0.2` para um treinamento mais agressivo.

O impacto dessas opções é atualmente desconhecido.

Seu config.json ficará parecido com o meu no final:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
    "validation_torch_compile": "false",
    "validation_step_interval": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 3.0,
    "validation_guidance_rescale": "0.0",
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-hidream",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "hidream",
    "offload_during_startup": true,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/hidream/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "text_encoder_3_precision": "int8-quanto",
    "text_encoder_4_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

> ℹ️ Esta configuração define os níveis de precisão dos text encoders T5 (#3) e Llama (#4) para int8 para economizar memória em placas de 24G. Você pode remover essas opções ou defini-las como `no_change` se tiver mais memória disponível.

E um arquivo simples `config/lycoris_config.json` - note que o `FeedForward` pode ser removido para maior estabilidade de treinamento.

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 16
            },
        }
    }
}
```
</details>

Definir `"use_scalar": true` em `config/lycoris_config.json` ou definir `"init_lokr_norm": 1e-4` em `config/config.json` vai acelerar o treinamento consideravelmente. Habilitar ambos parece desacelerar o treinamento levemente. Observe que definir `init_lokr_norm` vai alterar levemente as imagens de validação no step 0.

Adicionar o módulo `FeedForward` ao `config/lycoris_config.json` treinará um número muito maior de parâmetros, incluindo todos os experts. Treinar os experts parece ser bem difícil, porém.

Uma opção mais fácil é treinar apenas os parâmetros de feed forward fora dos experts usando o seguinte arquivo `config/lycoris_config.json`.

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "name_algo_map": {
            "double_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_t*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            }
        },
        "use_fnmatch": true
    }
}
```
</details>

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


O SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz o viés de exposição e melhora a qualidade de saída ao deixar o modelo gerar suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam a sobrecarga computacional do treinamento.

#### Prompts de validação

Dentro de `config/config.json` está o "prompt de validação primário", que normalmente é o principal instance_prompt no qual você está treinando para seu único assunto ou estilo. Além disso, um arquivo JSON pode ser criado contendo prompts extras para executar durante as validações.

O arquivo de exemplo `config/user_prompt_library.json.example` contém o seguinte formato:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

Os nicknames são o nome do arquivo de validação, então mantenha-os curtos e compatíveis com seu sistema de arquivos.

Para apontar o treinador para essa biblioteca de prompts, adicione-a ao TRAINER_EXTRA_ARGS com uma nova linha no final do `config.json`:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

Um conjunto de prompts diversificados ajuda a determinar se o modelo está colapsando conforme treina. Neste exemplo, a palavra `<token>` deve ser substituída pelo nome do seu assunto (instance_prompt).

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

> ℹ️ O HiDream usa 128 tokens por padrão e depois trunca.

#### Acompanhamento de CLIP score

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre configuração e interpretação de CLIP scores.

</details>

# Perda de avaliação estável

Se você quiser usar perda MSE estável para pontuar o desempenho do modelo, veja [este documento](../evaluation/EVAL_LOSS.md) para informações sobre configuração e interpretação de perda de avaliação.

#### Prévias de validação

O SimpleTuner suporta streaming de prévias intermediárias de validação durante a geração usando modelos Tiny AutoEncoder. Isso permite ver imagens de validação sendo geradas passo a passo em tempo real via callbacks de webhook.

Para habilitar:
<details>
<summary>Ver exemplo de configuração</summary>

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

Defina `validation_preview_steps` para um valor maior (ex.: 3 ou 5) para reduzir a sobrecarga do Tiny AutoEncoder. Com `validation_num_inference_steps=20` e `validation_preview_steps=5`, você receberá imagens de prévia nos steps 5, 10, 15 e 20.

#### Shift do cronograma de flow

Modelos flow-matching como OmniGen, Sana, Flux e SD3 têm uma propriedade chamada "shift" que permite deslocar a porção treinada do cronograma de timesteps usando um valor decimal simples.

O modelo `full` foi treinado com um valor de `3.0` e o `dev` usou `6.0`.

Na prática, usar um valor de shift tão alto tende a destruir ambos os modelos. Um valor de `1.0` é um bom ponto de partida, mas pode mover o modelo pouco, e `3.0` pode ser alto demais.

##### Auto-shift

Uma abordagem comumente recomendada é seguir vários trabalhos recentes e habilitar shift de timestep dependente de resolução, `--flow_schedule_auto_shift`, que usa valores de shift maiores para imagens maiores e menores para imagens menores. Isso resulta em resultados de treinamento estáveis, porém potencialmente medíocres.

##### Especificação manual

_Agradecimentos a General Awareness do Discord pelos exemplos a seguir_

Ao usar um valor de `--flow_schedule_shift` de 0.1 (um valor muito baixo), apenas os detalhes mais finos da imagem são afetados:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Ao usar um valor de `--flow_schedule_shift` de 4.0 (um valor muito alto), as grandes características composicionais e potencialmente o espaço de cores do modelo são impactados:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Considerações sobre o dataset

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisará garantir que seu dataset seja grande o suficiente para treinar seu modelo efetivamente. Observe que o tamanho mínimo do dataset é `train_batch_size * gradient_accumulation_steps`, além de ser maior que `vae_batch_size`. O dataset não será utilizável se for muito pequeno.

> ℹ️ Com poucas imagens, você pode ver a mensagem **no images detected in dataset** - aumentar o valor de `repeats` supera essa limitação.

Dependendo do dataset que você tem, será necessário configurar o diretório do dataset e o arquivo de configuração do dataloader de forma diferente. Neste exemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo:

<details>
<summary>Ver exemplo de configuração</summary>

```json
[
  {
    "id": "pseudo-camera-10k-hidream",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/hidream/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/hidream/dreambooth-subject",
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
    "cache_dir": "cache/text/hidream",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> ℹ️ Use `caption_strategy=textfile` se você tiver arquivos `.txt` contendo captions.
> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Depois, crie um diretório `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # coloque suas imagens em dreambooth-subject/ agora
popd
```

Isso vai baixar cerca de 10k amostras de fotos para o seu diretório `datasets/pseudo-camera-10k`, que será criado automaticamente.

Suas imagens Dreambooth devem ir para o diretório `datasets/dreambooth-subject`.

#### Login no WandB e Huggingface Hub

Você vai querer fazer login no WandB e no HF Hub antes de iniciar o treinamento, especialmente se estiver usando `--push_to_hub` e `--report_to=wandb`.

Se você vai enviar itens para um repositório Git LFS manualmente, também deve executar `git config --global credential.helper store`

Execute os seguintes comandos:

```bash
wandb login
```

e

```bash
huggingface-cli login
```

Siga as instruções para fazer login nos dois serviços.

### Executando o treinamento

A partir do diretório do SimpleTuner, você tem várias opções para iniciar o treinamento:

**Opção 1 (Recomendado - pip install):**
```bash
pip install simpletuner[cuda]
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

Isso vai iniciar o cache de text embeds e saídas VAE em disco.

Para mais informações, veja os documentos [dataloader](../DATALOADER.md) e [tutorial](../TUTORIAL.md).

### Rodando inferência no LoKr depois

Como é um modelo novo, o exemplo precisará de alguns ajustes para funcionar. Aqui vai um exemplo funcional:

<details>
<summary>Mostrar exemplo de inferência em Python</summary>

```py
import torch
from helpers.models.hidream.pipeline import HiDreamImagePipeline
from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

llama_repo = "unsloth/Meta-Llama-3.1-8B-Instruct"
model_id = 'HiDream-ai/HiDream-I1-Dev'
adapter_repo_id = 'bghira/hidream5m-photo-1mp-Prodigy'
adapter_filename = 'pytorch_lora_weights.safetensors'

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    llama_repo,
)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_repo,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

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
transformer = HiDreamImageTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = HiDreamImagePipeline.from_pretrained(
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

## Opcional: quantize o modelo para economizar VRAM.
## Nota: o modelo foi quantizado durante o treinamento, e por isso é recomendado fazer o mesmo na inferência.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # o pipeline já está no nível de precisão alvo
t5_embeds, llama_embeds, negative_t5_embeds, negative_llama_embeds, pooled_embeds, negative_pooled_embeds = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# Vamos zerar os text encoders para economizar memória.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
pipeline.text_encoder_4.to("meta")
model_output = pipeline(
    t5_prompt_embeds=t5_embeds,
    llama_prompt_embeds=llama_embeds,
    pooled_prompt_embeds=pooled_embeds,
    negative_t5_prompt_embeds=negative_t5_embeds,
    negative_llama_prompt_embeds=negative_llama_embeds,
    negative_pooled_prompt_embeds=negative_pooled_embeds,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```
</details>

## Notas e dicas de solução de problemas

### Configuração de VRAM mais baixa

A configuração de VRAM mais baixa do HiDream é de cerca de 20-22G:

- SO: Ubuntu Linux 24
- GPU: Um único dispositivo NVIDIA CUDA (10G, 12G)
- Memória do sistema: aproximadamente 50G de memória do sistema (pode ser mais, pode ser menos)
- Precisão do modelo base:
  - Para sistemas Apple e AMD, `int8-quanto` (ou `fp8-torchao`, `int8-torchao` seguem perfis de uso de memória semelhantes)
    - `int4-quanto` também funciona, mas você pode ter menor precisão / piores resultados
  - Para sistemas NVIDIA, `nf4-bnb` é relatado como funcionando bem, mas será mais lento que `int8-quanto`
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolução: 1024px
- Batch size: 1, zero steps de gradient accumulation
- DeepSpeed: desativado / não configurado
- PyTorch: 2.7+
- Usando `--quantize_via=cpu` para evitar erro outOfMemory durante a inicialização em placas <=16G.
- Habilitar `--gradient_checkpointing`
- Use um LoRA ou Lycoris bem pequeno (ex.: rank 1 ou fator Lokr 25)
- Definir a variável de ambiente `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ajuda a minimizar o uso de VRAM ao treinar múltiplas proporções de aspecto.

**NOTA**: O pré-cache de embeddings VAE e saídas do text encoder pode usar mais memória e ainda dar OOM. VAE tiling e slicing são habilitados por padrão. Se você ver OOM, tente habilitar `offload_during_startup=true`; caso contrário, talvez você simplesmente esteja sem sorte.

A velocidade foi de aproximadamente 3 iterações por segundo em uma NVIDIA 4090 usando Pytorch 2.7 e CUDA 12.8

### Loss mascarada

Se você estiver treinando um assunto ou estilo e quiser mascarar um ou outro, veja a seção de [treinamento com loss mascarada](../DREAMBOOTH.md#masked-loss) no guia de Dreambooth.

### Quantização

Embora `int8` seja a melhor opção para trade-offs de velocidade/qualidade vs memória, `nf4` e `int4` também estão disponíveis. `int4` não é recomendado para HiDream, pois pode levar a piores resultados, mas com treinamento suficiente, você acabaria com um modelo `int4` razoavelmente capaz.

### Taxas de aprendizado

#### LoRA (--lora_type=standard)

- Taxas de aprendizado mais altas em torno de 4e-4 funcionam melhor para LoRAs menores (rank-1 até rank-8)
- Taxas de aprendizado mais baixas em torno de 6e-5 funcionam melhor para LoRAs maiores (rank-64 até rank-256)
- Definir `lora_alpha` diferente de `lora_rank` não é suportado devido às limitações do Diffusers, a menos que você saiba o que está fazendo em ferramentas de inferência depois.
  - Como usar isso depois para inferência está fora do escopo, mas definir `lora_alpha` como 1.0 permitiria manter a taxa de aprendizado igual em todos os ranks de LoRA.

#### LoKr (--lora_type=lycoris)

- Taxas de aprendizado moderadas são melhores para LoKr (`1e-4` com AdamW, `2e-5` com Lion)
- Outros algos precisam de mais exploração.
- Prodigy parece ser uma boa escolha para LoRA ou LoKr, mas pode superestimar a taxa de aprendizado necessária e suavizar a pele.

### Artefatos de imagem

O HiDream tem uma resposta desconhecida a artefatos de imagem, embora use o VAE do Flux e tenha limitações semelhantes de detalhes finos.

O problema mais prevalente é usar uma taxa de aprendizado alta demais e/ou batch size baixo demais. Isso pode fazer o modelo produzir imagens com artefatos, como pele lisa, borrado e pixelização.

### Bucketing de aspecto

Inicialmente, o modelo não respondia muito bem a buckets de aspecto, mas a implementação foi melhorada pela comunidade.

### Treinamento multi-resolução

O modelo pode ser inicialmente treinado em uma resolução mais baixa como 512px para acelerar o treinamento, mas não é certo se o modelo vai generalizar muito bem para resoluções mais altas. Treinar sequencialmente primeiro em 512px e depois em 1024px provavelmente é a melhor abordagem.

É uma boa ideia habilitar `--flow_schedule_auto_shift` ao treinar em resoluções diferentes de 1024px. Resoluções mais baixas usam menos VRAM, permitindo batch sizes maiores.

### Full-rank tuning

O DeepSpeed vai usar MUITA memória do sistema com HiDream, mas o fine-tuning completo parece funcionar bem em um sistema muito grande.

Lycoris LoKr é recomendado no lugar do fine-tuning full-rank, pois é mais estável e tem menor uso de memória.

PEFT LoRA é útil para estilos mais simples, mas é mais difícil manter detalhes finos com ele.
