# Guia de Início Rápido do Chroma 1

![image](https://github.com/user-attachments/assets/3c8a12c6-9d45-4dd4-9fc8-6b7cd3ed51dd)

Chroma 1 é uma variante reduzida de 8.9B parâmetros do Flux.1 Schnell, lançada pela Lodestone Labs. Este guia mostra como configurar o SimpleTuner para treino de LoRA.

## Requisitos de hardware

Apesar do menor número de parâmetros, o uso de memória é próximo ao Flux Schnell:

- A quantização do transformer base ainda pode usar **≈40–50 GB** de RAM do sistema.
- Treino LoRA com rank 16 normalmente consome:
  - ~28 GB de VRAM sem quantização da base
  - ~16 GB de VRAM com int8 + bf16
  - ~11 GB de VRAM com int4 + bf16
  - ~8 GB de VRAM com NF4 + bf16
- Mínimo realista de GPU: **RTX 3090 / RTX 4090 / L40S** ou melhor.
- Funciona bem em **Apple M-series (MPS)** para treino LoRA e em AMD ROCm.
- Aceleradores de 80 GB ou setups multi-GPU são recomendados para fine-tuning full-rank.

## Pré-requisitos

Chroma compartilha as mesmas expectativas de runtime do guia do Flux:

- Python **3.10 – 3.12**
- Um backend de aceleração suportado (CUDA, ROCm ou MPS)

Verifique sua versão do Python:

```bash
python3 --version
```

Instale o SimpleTuner (exemplo CUDA):

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Para detalhes de setup específicos do backend (CUDA, ROCm, Apple), consulte o [guia de instalação](../INSTALL.md).

## Iniciando a web UI

```bash
simpletuner server
```

A UI estará disponível em http://localhost:8001.

## Configuração via CLI

`simpletuner configure` guia você pelas configurações centrais. Os valores-chave para Chroma são:

- `model_type`: `lora`
- `model_family`: `chroma`
- `model_flavour`: um dos
  - `base` (padrão, qualidade equilibrada)
  - `hd` (maior fidelidade, mais pesado)
  - `flash` (rápido porém instável – não recomendado para produção)
- `pretrained_model_name_or_path`: deixe vazio para usar o mapeamento de flavour acima
- `model_precision`: mantenha o padrão `bf16`
- `flux_fast_schedule`: deixe **desativado**; Chroma tem amostragem adaptativa própria

### Exemplo de configuração manual

<details>
<summary>Ver exemplo de config</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "chroma",
  "model_flavour": "base",
  "output_dir": "/workspace/chroma-output",
  "network_rank": 16,
  "learning_rate": 2.0e-4,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "pretrained_model_name_or_path": null
}
```
</details>

> ⚠️ Se o acesso ao Hugging Face for lento na sua região, exporte `HF_ENDPOINT=https://hf-mirror.com` antes de iniciar.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

</details>

## Dataset e dataloader

Chroma usa o mesmo formato de dataloader do Flux. Consulte o [tutorial geral](../TUTORIAL.md) ou o [tutorial da web UI](../webui/TUTORIAL.md) para preparação de datasets e bibliotecas de prompts.

## Opções de treinamento específicas do Chroma

- `flux_lora_target`: controla quais módulos do transformer recebem adaptadores LoRA (`all`, `all+ffs`, `context`, `tiny`, etc.). Os padrões espelham o Flux e funcionam bem na maioria dos casos.
- `flux_guidance_mode`: `constant` funciona bem; Chroma não expõe um intervalo de guidance.
- Mascaramento de atenção é sempre habilitado – garanta que seu cache de embeddings de texto foi gerado com padding masks (comportamento padrão nas releases atuais do SimpleTuner).
- Opções de shift de schedule (`flow_schedule_shift` / `flow_schedule_auto_shift`) não são necessárias para Chroma — o helper já aumenta os timesteps finais automaticamente.
- `flux_t5_padding`: defina `zero` se você preferir zerar tokens de padding antes do masking.

## Amostragem automática de timesteps finais

Flux usava um schedule log-normal que subamostrava extremos de alto/baixo ruído. O helper de treino do Chroma aplica um remapeamento quadrático (`σ ↦ σ²` / `1-(1-σ)²`) nos sigmas amostrados para que as regiões de cauda sejam visitadas com mais frequência. Isso **não exige configuração extra** — está embutido na família de modelo `chroma`.

## Dicas de validação e amostragem

- `validation_guidance_real` mapeia diretamente para `guidance_scale` do pipeline. Deixe em `1.0` para amostragem de passagem única, ou suba para `2.0`–`3.0` se quiser classifier-free guidance nas renderizações de validação.
- Use 20 steps de inferência para prévias rápidas; 28–32 para maior qualidade.
- Prompts negativos continuam opcionais; o modelo base já é de-distilled.
- O modelo só suporta text-to-image no momento; suporte a img2img chegará em uma atualização futura.

## Solução de problemas

- **OOM no início**: ative `offload_during_startup` ou quantize o modelo base (`base_model_precision: int8-quanto`).
- **Treino diverge cedo**: garanta que o gradient checkpointing esteja ligado, reduza `learning_rate` para `1e-4` e verifique se as legendas são diversas.
- **Validação repete a mesma pose**: aumente o tamanho dos prompts; modelos de flow matching colapsam quando a variedade de prompts é baixa.

Para tópicos avançados—DeepSpeed, FSDP2, métricas de avaliação—consulte os guias compartilhados linkados no README.
