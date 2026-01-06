# LayerSync (SimpleTuner)

LayerSync e um empurrao “ensine a si mesmo” para modelos transformer: uma camada (o student) aprende a alinhar com uma camada mais forte (o teacher). E leve, autocontido e nao exige baixar modelos extras.

## Quando usar

- Voce esta treinando familias transformer que expoem hidden states (ex.: Flux/Flux Kontext/Flux.2, PixArt Sigma, SD3/SDXL, Sana, Wan, Qwen Image/Edit, Hunyuan Video, LTXVideo, Kandinsky5 Video, Chroma, ACE-Step, HiDream, Cosmos/LongCat/Z-Image/Auraflow).
- Voce quer um regularizador embutido sem enviar um checkpoint de teacher externo.
- Voce esta vendo drift no meio do treinamento ou heads instaveis e quer puxar uma mid-layer em direcao a um teacher mais profundo.
- Voce tem um pouco de VRAM extra para manter as ativacoes de student/teacher no step atual.

## Config rapida (WebUI)

1. Abra **Training -> Loss functions**.
2. Habilite **LayerSync**.
3. Defina **Student Block** como uma mid-layer e **Teacher Block** como uma camada mais profunda. Em modelos DiT de 24 camadas (Flux, PixArt, SD3), comece com `8` -> `16`; em pilhas menores, mantenha o teacher alguns blocos acima do student.
4. Deixe **Weight** em `0.2` (padrao quando LayerSync esta habilitado).
5. Treine normalmente; os logs incluem `layersync_loss` e `layersync_similarity`.

## Config rapida (config JSON / CLI)

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 8,
  "layersync_teacher_block": 16,
  "layersync_lambda": 0.2
}
```

## Ajustes

- `layersync_student_block` / `layersync_teacher_block`: indices amigaveis 1-based; tentamos `idx-1` primeiro e depois `idx`.
- `layersync_lambda`: escala a loss de cosseno; deve ser > 0 quando habilitado (padrao `0.2`).
- O teacher padrao vira o mesmo bloco do student quando omitido, tornando a loss auto-similaridade.
- VRAM: ativacoes de ambas as camadas sao mantidas ate a loss auxiliar rodar; desative LayerSync (ou CREPA) se precisar liberar memoria.
- Funciona bem com CREPA/TwinFlow; compartilham o mesmo buffer de hidden states.

<details>
<summary>Como funciona (pratico)</summary>

- Calcula similaridade de cosseno negativa entre tokens achatados do student e do teacher; peso mais alto puxa o student para os recursos do teacher.
- Tokens do teacher sao sempre detachados para evitar gradientes voltando.
- Suporta hidden states 3D `(B, S, D)` e 4D `(B, T, P, D)` para transformers de imagem e video.
- Mapeamento de opcoes upstream:
  - `--encoder-depth` -> `--layersync_student_block`
  - `--gt-encoder-depth` -> `--layersync_teacher_block`
  - `--reg-weight` -> `--layersync_lambda`
- Padroes: desligado por padrao; quando habilitado e nao definido, `layersync_lambda=0.2`.

</details>

<details>
<summary>Tecnico (internals do SimpleTuner)</summary>

- Implementacao: `simpletuner/helpers/training/layersync.py`; chamada por `ModelFoundation._apply_layersync_regularizer`.
- Captura de hidden state: acionada quando LayerSync ou CREPA solicita; transformers armazenam estados como `layer_{idx}` via `_store_hidden_state`.
- Resolucao de camada: tenta indices 1-based e depois 0-based; erro se as camadas solicitadas estiverem ausentes.
- Caminho de loss: normaliza tokens de student/teacher, calcula similaridade media de cosseno, registra `layersync_loss` e `layersync_similarity` e adiciona a loss escalada ao objetivo principal.
- Interacao: roda apos CREPA para que ambos reutilizem o mesmo buffer; limpa o buffer depois.

</details>

## Armadilhas comuns

- Student block ausente -> erro na inicializacao; defina `layersync_student_block` explicitamente.
- Peso <= 0 -> erro na inicializacao; mantenha o padrao `0.2` se tiver duvida.
- Pedir blocos mais profundos do que o modelo expõe -> erros “LayerSync could not find layer”; reduza os indices.
- Habilitar em modelos que nao expõem hidden states de transformer (Kolors, Lumina2, Stable Cascade C, Kandinsky5 Image, OmniGen) falhara; use apenas familias com transformer.
- Picos de VRAM: reduza indices dos blocos ou desative CREPA/LayerSync para liberar o buffer de hidden states.

Use LayerSync quando quiser um regularizador barato e embutido para guiar representacoes intermediarias sem adicionar teachers externos.
