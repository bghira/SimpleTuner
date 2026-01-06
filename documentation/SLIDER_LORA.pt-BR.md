# Slider LoRA Targeting

Neste guia, vamos treinar um adapter no estilo slider no SimpleTuner. Vamos usar o Z-Image Turbo porque ele treina rapido, tem licenca Apache 2.0 e entrega otimos resultados para o seu tamanho, mesmo com pesos destilados.

Para a matriz completa de compatibilidade (LoRA, LyCORIS, full-rank), veja a coluna Sliders em [documentation/QUICKSTART.md](QUICKSTART.md); este guia se aplica a todas as arquiteturas.

Slider targeting funciona com LoRA padrao, LyCORIS (incluindo `full`) e ControlNet. O toggle esta disponivel tanto no CLI quanto na WebUI; tudo ja vem no SimpleTuner, sem installs extras.

## Passo 1 - Siga a configuracao base

- **CLI**: Siga `documentation/quickstart/ZIMAGE.md` para ambiente, instalacao, notas de hardware e o `config.json` inicial.
- **WebUI**: Use `documentation/webui/TUTORIAL.md` para rodar o assistente do trainer; escolha Z-Image Turbo como de costume.

Tudo desses guias pode ser seguido ate o ponto de configurar um dataset, porque sliders apenas mudam onde os adapters sao colocados e como os dados sao amostrados.

## Passo 2 - Habilite slider targets

- CLI: adicione `"slider_lora_target": true` (ou passe `--slider_lora_target true`).
- WebUI: Model -> LoRA Config -> Advanced -> marque “Use slider LoRA targets”.

Para LyCORIS, mantenha `lora_type: "lycoris"` e para `lycoris_config.json`, use os presets na secao de detalhes abaixo.

## Passo 3 - Construa datasets amigaveis a slider

Sliders de conceito aprendem com um dataset contrastivo de "opostos". Crie pares antes/depois pequenos (4-6 pares ja bastam para comecar, mais se voce tiver):

- **Bucket positivo**: “mais do conceito” (ex.: olhos mais brilhantes, sorriso mais forte, mais areia). Defina `"slider_strength": 0.5` (qualquer valor positivo).
- **Bucket negativo**: “menos do conceito” (ex.: olhos mais apagados, expressao neutra). Defina `"slider_strength": -0.5` (qualquer valor negativo).
- **Bucket neutro (opcional)**: exemplos regulares. Omita `slider_strength` ou defina como `0`.

Nao e necessario manter nomes de arquivos correspondentes entre pastas positivas/negativas - apenas garanta um numero igual de amostras em cada bucket.

## Passo 4 - Aponte o dataloader para seus buckets

- Use o mesmo padrao de JSON do dataloader do quickstart de Z-Image.
- Adicione `slider_strength` em cada entrada de backend. O SimpleTuner vai:
  - Alternar batches **positivo -> negativo -> neutro** para manter ambas direcoes frescas.
  - Ainda respeitar a probabilidade de cada backend, entao seus knobs de ponderacao continuam funcionando.

Voce nao precisa de flags extras - apenas os campos `slider_strength`.

## Passo 5 - Treine

Use o comando usual (`simpletuner train ...`) ou inicie pela WebUI. O slider targeting e automatico quando a flag esta ligada.

## Passo 6 - Valide (ajustes opcionais de slider)

Bibliotecas de prompt podem carregar escalas de adapter por prompt para testes A/B:

```json
{
  "plain": "regular prompt",
  "slider_plus": { "prompt": "same prompt", "adapter_strength": 1.2 },
  "slider_minus": { "prompt": "same prompt", "adapter_strength": 0.5 }
}
```

Se omitido, a validacao usa sua forca global.

---

## Referencias e detalhes

<details>
<summary>Por que esses targets? (tecnico)</summary>

O SimpleTuner direciona sliders LoRA para self-attention, conv/proj e camadas de time-embedding para imitar a regra de Concept Sliders de “nao mexer no texto”. Execucoes de ControlNet ainda respeitam o slider targeting. Adapters assistentes permanecem congelados.
</details>

<details>
<summary>Listas padrao de targets de slider (por arquitetura)</summary>

- Geral (SD1.x, SDXL, SD3, Lumina2, Wan, HiDream, LTXVideo, Qwen-Image, Cosmos, Stable Cascade, etc.):

  ```json
  [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn1.to_qkv", "to_qkv",
    "proj_in", "proj_out",
    "conv_in", "conv_out",
    "time_embedding.linear_1", "time_embedding.linear_2"
  ]
  ```

- Flux / Flux2 / Chroma / AuraFlow (apenas stream visual):

  ```json
  ["to_q", "to_k", "to_v", "to_out.0", "to_qkv"]
  ```

  Variantes Flux2 incluem `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`, `attn.to_qkv_mlp_proj`.

- Kandinsky 5 (imagem/video):

  ```json
  ["attn1.to_query", "attn1.to_key", "attn1.to_value", "conv_in", "conv_out", "time_embedding.linear_1", "time_embedding.linear_2"]
  ```

</details>

<details>
<summary>Presets LyCORIS (exemplo LoKr)</summary>

A maioria dos modelos:

```json
{
  "algo": "lokr",
  "multiplier": 1.0,
  "linear_dim": 4,
  "linear_alpha": 1,
  "apply_preset": {
    "target_module": [
      "attn1.to_q",
      "attn1.to_k",
      "attn1.to_v",
      "attn1.to_out.0",
      "conv_in",
      "conv_out",
      "time_embedding.linear_1",
      "time_embedding.linear_2"
    ]
  }
}
```

Flux/Chroma/AuraFlow: troque os targets para `["attn.to_q","attn.to_k","attn.to_v","attn.to_out.0","attn.to_qkv_mlp_proj"]` (remova `attn.` quando checkpoints o omitirem). Evite projecoes `add_*` para manter texto/contexto intocado.

Kandinsky 5: use `attn1.to_query/key/value` mais `conv_*` e `time_embedding.linear_*`.
</details>

<details>
<summary>Como a amostragem funciona (tecnico)</summary>

Backends marcados com `slider_strength` sao agrupados por sinal e amostrados em um ciclo fixo: positivo -> negativo -> neutro. Dentro de cada grupo, as probabilidades usuais do backend se aplicam. Backends esgotados sao removidos e o ciclo continua com o que restar.
</details>
