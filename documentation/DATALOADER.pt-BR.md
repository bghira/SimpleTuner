# Arquivo de configuração do dataloader

Aqui está o exemplo mais básico de um arquivo de configuração do dataloader, como `multidatabackend.example.json`.

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": true,
    "crop_style": "center",
    "crop_aspect": "square",
    "resolution": 1024,
    "minimum_image_size": 768,
    "maximum_image_size": 2048,
    "minimum_aspect_ratio": 0.50,
    "maximum_aspect_ratio": 3.00,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "textfile",
    "cache_dir_vae": "/path/to/vaecache",
    "repeats": 0
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  }
]
```

## Opções de configuração

### `id`

- **Descrição:** Identificador único do dataset. Ele deve permanecer constante depois de definido, pois vincula o dataset às entradas de rastreamento de estado.

### `disabled`

- **Valores:** `true` | `false`
- **Descrição:** Quando definido como `true`, este dataset é totalmente ignorado durante o treinamento. Útil para excluir temporariamente um dataset sem remover sua configuração.
- **Nota:** Também aceita a grafia `disable`.

### `dataset_type`

- **Valores:** `image` | `video` | `audio` | `text_embeds` | `image_embeds` | `conditioning_image_embeds` | `conditioning`
- **Descrição:** Datasets `image`, `video` e `audio` contêm as amostras principais de treinamento. `text_embeds` contém as saídas do cache do text encoder, `image_embeds` contém os latentes do VAE (quando um modelo usa um), e `conditioning_image_embeds` armazenam embeddings de imagens de condicionamento em cache (como recursos de visão do CLIP). Quando um dataset é marcado como `conditioning`, é possível pareá-lo ao seu dataset `image` via [a opção conditioning_data](#conditioning_data)
- **Nota:** Datasets de text embed e image embed são definidos de forma diferente de datasets de imagem. Um dataset de text embed armazena APENAS os objetos de text embed. Um dataset de imagem armazena os dados de treinamento.
- **Nota:** Não combine imagens e vídeo em um **único** dataset. Separe-os.

### `default`

- **Aplica-se apenas a `dataset_type=text_embeds`**
- Se definido como `true`, este dataset de text embed será onde o SimpleTuner armazenará o cache de text embeds, por exemplo, para embeddings de prompts de validação. Como eles não se vinculam a dados de imagem, precisa haver um local específico para eles.

### `cache_dir`

- **Aplica-se apenas a `dataset_type=text_embeds` e `dataset_type=image_embeds`**
- **Descrição:** Especifica onde os arquivos de cache de embeds são armazenados para este dataset. Para `text_embeds`, é onde as saídas do text encoder são gravadas. Para `image_embeds`, é onde os latentes do VAE são armazenados.
- **Nota:** Diferente de `cache_dir_vae`, que é definido em datasets de imagem/vídeo primários para especificar onde o cache de VAE vai.

### `write_batch_size`

- **Aplica-se apenas a `dataset_type=text_embeds`**
- **Descrição:** Número de text embeds a serem gravados em uma única operação de batch. Valores maiores podem melhorar a vazão de escrita, mas usam mais memória.
- **Padrão:** Usa o argumento `--write_batch_size` do trainer (tipicamente 128).

### `text_embeds`

- **Aplica-se apenas a `dataset_type=image`**
- Se não definido, o dataset `text_embeds` padrão será usado. Se definido como o `id` existente de um dataset `text_embeds`, ele usará esse. Permite associar datasets específicos de text embeds a um dataset de imagens.

### `image_embeds`

- **Aplica-se apenas a `dataset_type=image`**
- Se não definido, as saídas do VAE serão armazenadas no backend de imagem. Caso contrário, você pode definir isso como o `id` de um dataset `image_embeds`, e as saídas do VAE serão armazenadas lá. Permite associar o dataset image_embed aos dados de imagem.

### `conditioning_image_embeds`

- **Aplica-se a `dataset_type=image` e `dataset_type=video`**
- Quando um modelo reporta `requires_conditioning_image_embeds`, defina isso para o `id` de um dataset `conditioning_image_embeds` para armazenar embeddings de imagem de condicionamento em cache (por exemplo, recursos de visão do CLIP para Wan 2.2 I2V). Se não definido, o SimpleTuner grava o cache em `cache/conditioning_image_embeds/<dataset_id>` por padrão, garantindo que não conflite com o cache do VAE.
- Modelos que precisam desses embeds devem expor um image encoder por meio do pipeline principal. Se o modelo não conseguir fornecer o encoder, o pré-processamento falhará cedo em vez de gerar arquivos vazios silenciosamente.

#### `cache_dir_conditioning_image_embeds`

- **Substituição opcional do destino do cache de conditioning image embeds.**
- Defina isso quando quiser fixar o cache em um local específico do sistema de arquivos ou tiver um backend remoto dedicado (`dataset_type=conditioning_image_embeds`). Quando omitido, o caminho de cache descrito acima é usado automaticamente.

#### `conditioning_image_embed_batch_size`

- **Substituição opcional do tamanho do batch usado ao gerar conditioning image embeds.**
- Usa por padrão o argumento do trainer `conditioning_image_embed_batch_size` ou o tamanho de batch do VAE quando não fornecido explicitamente.

### Configuração de dataset de áudio (`dataset_type=audio`)

Backends de áudio suportam um bloco `audio` dedicado para que metadados e cálculo de buckets levem em conta a duração. Exemplo:

```json
"audio": {
  "max_duration_seconds": 90,
  "channels": 2,
  "bucket_strategy": "duration",
  "duration_interval": 15,
  "truncation_mode": "beginning"
}
```

- **`bucket_strategy`** – atualmente `duration` é o padrão e trunca clipes em buckets espaçados de forma uniforme para que a amostragem por GPU respeite o cálculo de batch.
- **`duration_interval`** – arredondamento de bucket em segundos (padrão **3** quando não definido). Com `15`, um clipe de 77 s é colocado no bucket de 75 s. Isso impede que clipes longos únicos prejudiquem outros ranks e força truncamento no mesmo intervalo.
- **`max_duration_seconds`** – clipes mais longos que isso são ignorados durante a descoberta de metadados para que faixas excepcionalmente longas não consumam buckets inesperadamente.
- **`truncation_mode`** – determina qual parte do clipe é mantida quando ajustamos para o intervalo do bucket. Opções: `beginning`, `end` ou `random` (padrão: `beginning`).
- **`audio_only`** – modo de treinamento apenas áudio (LTX-2): treina apenas a geração de áudio sem arquivos de vídeo. Os latentes de vídeo são zerados automaticamente e a perda de vídeo é mascarada.
- **`target_resolution`** – resolução de vídeo alvo para o modo apenas áudio (usada para calcular dimensões de latentes).
- Configurações padrão de áudio (contagem de canais, diretório de cache) mapeiam diretamente para o backend de áudio em runtime criado por `simpletuner.helpers.data_backend.factory`. O padding é intencionalmente evitado — clipes são truncados em vez de estendidos para manter o comportamento consistente com treinadores de difusão como o ACE-Step.

### Captions de áudio (Hugging Face)
Para datasets de áudio do Hugging Face, você pode especificar quais colunas formam a caption (prompt) e qual coluna contém as letras:
```json
"config": {
    "audio_caption_fields": ["prompt", "tags"],
    "lyrics_column": "lyrics"
}
```
*   `audio_caption_fields`: Junta múltiplas colunas para formar o prompt de texto (usado pelo text encoder).
*   `lyrics_column`: Especifica a coluna das letras (usada pelo lyric encoder).

Durante a descoberta de metadados, o loader registra `sample_rate`, `num_samples`, `num_channels` e `duration_seconds` para cada arquivo. Relatórios de buckets no CLI agora falam em **amostras** em vez de **imagens**, e diagnósticos de dataset vazio listarão o `bucket_strategy`/`duration_interval` ativo (além de qualquer limite `max_duration_seconds`) para que você ajuste os intervalos sem mergulhar nos logs.

### `type`

- **Valores:** `aws` | `local` | `csv` | `huggingface`
- **Descrição:** Determina o backend de armazenamento (local, csv ou nuvem) usado para este dataset.

### `conditioning_type`

- **Valores:** `controlnet` | `mask` | `reference_strict` | `reference_loose`
- **Descrição:** Especifica o tipo de condicionamento para um dataset `conditioning`.
  - **controlnet**: Entradas de condicionamento do ControlNet para treinamento de sinal de controle.
  - **mask**: Máscaras binárias para treinamento de inpainting.
  - **reference_strict**: Imagens de referência com alinhamento estrito de pixels (para modelos de edição como Qwen Edit).
  - **reference_loose**: Imagens de referência com alinhamento frouxo.

### `source_dataset_id`

- **Aplica-se apenas a `dataset_type=conditioning`** com `conditioning_type` de `reference_strict`, `reference_loose` ou `mask`
- **Descrição:** Vincula um dataset de condicionamento ao dataset de imagem/vídeo de origem para alinhamento de pixels. Quando definido, o SimpleTuner duplica metadados do dataset de origem para garantir que as imagens de condicionamento se alinhem aos seus alvos.
- **Nota:** Obrigatório para modos de alinhamento estrito; opcional para alinhamento frouxo.

### `conditioning_data`

- **Valores:** valor `id` do dataset de condicionamento ou um array de valores `id`
- **Descrição:** Conforme descrito no [guia do ControlNet](CONTROLNET.md), um dataset `image` pode ser pareado ao seu ControlNet ou dados de máscara de imagem via esta opção.
- **Nota:** Se você tiver múltiplos datasets de condicionamento, pode especificá-los como um array de valores `id`. Ao treinar Flux Kontext, isso permite alternar aleatoriamente entre condições ou juntar entradas para treinar tarefas mais avançadas de composição multi-imagem.

### `instance_data_dir` / `aws_data_prefix`

- **Local:** Caminho para os dados no sistema de arquivos.
- **AWS:** Prefixo S3 para os dados no bucket.

### `caption_strategy`

- **textfile** exige que sua image.png esteja ao lado de uma image.txt que contém uma ou mais captions, separadas por novas linhas. Esses pares imagem+texto **devem estar no mesmo diretório**.
- **instanceprompt** exige que um valor `instance_prompt` também seja fornecido e usará **apenas** esse valor como a caption de todas as imagens do conjunto.
- **filename** usará uma versão convertida e limpa do nome do arquivo como caption, por exemplo, após trocar sublinhados por espaços.
- **parquet** extrairá captions da tabela parquet que contém o restante dos metadados da imagem. Use o campo `parquet` para configurar isso. Veja [Estratégia de captions Parquet](#estrategia-de-captions-parquet-e-datasets-json-lines).

Tanto `textfile` quanto `parquet` suportam multi-captions:
- textfiles são separadas por novas linhas. Cada nova linha será sua própria caption.
- tabelas parquet podem ter um tipo iterável no campo.

### `disable_multiline_split`

- Quando definido como `true`, impede que arquivos de texto de caption sejam divididos por novas linhas em múltiplas variantes de caption.
- Útil quando suas captions contêm quebras de linha intencionais que devem ser preservadas como uma única caption.
- Padrão: `false` (captions são divididas por novas linhas)

### `caption_shuffle`

Gera variantes embaralhadas determinísticas de captions baseadas em tags para aumento de dados. Isso ajuda o modelo a aprender que a ordem das tags não importa e reduz o overfitting em sequências específicas de tags.

**Configuração:**

```json
{
  "caption_shuffle": {
    "enable": true,
    "count": 3,
    "seed": 42,
    "split_on": "comma",
    "position_start": 1,
    "include_original": true
  }
}
```

**Parâmetros:**

- `enable` (bool): Se deve habilitar o embaralhamento de captions. Padrão: `false`
- `count` (int): Número de variantes embaralhadas a gerar por caption. Padrão: `1`
- `seed` (int): Seed para embaralhamento determinístico. Se não especificado, usa o valor global `--seed`.
- `split_on` (string): Delimitador para dividir captions em tags. Opções: `comma`, `space`, `period`. Padrão: `comma`
- `position_start` (int): Manter as primeiras N tags em sua posição original (útil para manter tags de assunto/estilo primeiro). Padrão: `0`
- `include_original` (bool): Se deve incluir a caption original não embaralhada junto com as variantes embaralhadas. Padrão: `true`

**Exemplo:**

Com `split_on: "comma"`, `position_start: 1`, `count: 2`:

- Original: `"dog, running, park, sunny day"`
- Resultado: `["dog, running, park, sunny day", "dog, park, sunny day, running", "dog, sunny day, running, park"]`

A primeira tag "dog" permanece fixa enquanto as tags restantes são embaralhadas.

**Notas:**

- O embaralhamento é aplicado durante o pré-cache de embeddings de texto, então todas as variantes são calculadas de uma vez.
- Durante o treinamento, uma variante é selecionada aleatoriamente por amostra.
- Se uma caption tiver menos tags que `position_start + 2`, o embaralhamento é pulado (nada significativo para embaralhar).
- Quando `include_original: false` mas o embaralhamento não é possível, a original é incluída mesmo assim com um aviso.

### `metadata_backend`

- **Valores:** `discovery` | `parquet` | `huggingface`
- **Descrição:** Controla como o SimpleTuner descobre dimensões de imagem e outros metadados durante a preparação do dataset.
  - **discovery** (padrão): Examina arquivos de imagem reais para ler dimensões. Funciona com qualquer backend de armazenamento, mas pode ser lento para datasets grandes.
  - **parquet**: Lê dimensões de `width_column` e `height_column` em um arquivo parquet/JSONL, evitando acessar os arquivos. Veja [Estratégia de captions Parquet](#estrategia-de-captions-parquet-e-datasets-json-lines).
  - **huggingface**: Usa metadados de datasets do Hugging Face. Veja [Suporte a Hugging Face Datasets](#hugging-face-datasets-support).
- **Nota:** Ao usar `parquet`, você também deve configurar o bloco `parquet` com `width_column` e `height_column`. Isso acelera drasticamente a inicialização para datasets grandes.

### `metadata_update_interval`

- **Valores:** Inteiro (segundos)
- **Descrição:** Com que frequência (em segundos) atualizar os metadados do dataset durante o treinamento. Útil para datasets que podem mudar durante uma execução longa.
- **Padrão:** Usa o argumento do trainer `--metadata_update_interval`.

### Opções de corte

- `crop`: Habilita ou desabilita o corte de imagem.
- `crop_style`: Seleciona o estilo de corte (`random`, `center`, `corner`, `face`).
- `crop_aspect`: Escolhe o aspecto do corte (`closest`, `random`, `square` ou `preserve`).
- `crop_aspect_buckets`: Quando `crop_aspect` é definido como `closest` ou `random`, um bucket desta lista será selecionado. Por padrão, todos os buckets estão disponíveis (permitindo upscaling ilimitado). Use `max_upscale_threshold` para limitar o upscaling se necessário.

### `resolution`

- **resolution_type=area:** O tamanho final da imagem é determinado pela contagem de megapixels - um valor de 1.05 aqui corresponde a buckets de aspecto em torno de 1024^2 (1024x1024) de área total de pixels, ~1_050_000 pixels.
- **resolution_type=pixel_area:** Como `area`, o tamanho final da imagem é pela área, mas medido em pixels em vez de megapixels. Um valor de 1024 aqui gerará buckets de aspecto em torno de 1024^2 (1024x1024) de área total de pixels, ~1_050_000 pixels.
- **resolution_type=pixel:** O tamanho final da imagem será determinado pela menor borda ter este valor.

> **NOTA**: Se as imagens são ampliadas, reduzidas ou cortadas, isso depende dos valores de `minimum_image_size`, `maximum_target_size`, `target_downsample_size`, `crop` e `crop_aspect`.

### `minimum_image_size`

- Quaisquer imagens cujo tamanho fique abaixo desse valor serão **excluídas** do treinamento.
- Quando `resolution` é medida em megapixels (`resolution_type=area`), isso deve estar em megapixels também (ex.: `1.05` megapixels para excluir imagens abaixo de 1024x1024 de **área**)
- Quando `resolution` é medida em pixels, você deve usar a mesma unidade aqui (ex.: `1024` para excluir imagens com **lado menor** abaixo de 1024px)
- **Recomendação**: Mantenha `minimum_image_size` igual a `resolution`, a menos que você queira arriscar treinar com imagens que foram ampliadas de forma ruim.

### `minimum_aspect_ratio`

- **Descrição:** A proporção mínima da imagem. Se a proporção da imagem for menor que este valor, ela será excluída do treinamento.
- **Nota**: Se o número de imagens qualificadas para exclusão for excessivo, isso pode desperdiçar tempo na inicialização, pois o trainer tentará escaneá-las e criar buckets se elas estiverem faltando nas listas de buckets.

> **Nota**: Depois que as listas de aspecto e metadados forem construídas para o seu dataset, usar `skip_file_discovery="vae aspect metadata"` impedirá que o trainer escaneie o dataset na inicialização, economizando muito tempo.

### `maximum_aspect_ratio`

- **Descrição:** A proporção máxima da imagem. Se a proporção da imagem for maior que este valor, ela será excluída do treinamento.
- **Nota**: Se o número de imagens qualificadas para exclusão for excessivo, isso pode desperdiçar tempo na inicialização, pois o trainer tentará escaneá-las e criar buckets se elas estiverem faltando nas listas de buckets.

> **Nota**: Depois que as listas de aspecto e metadados forem construídas para o seu dataset, usar `skip_file_discovery="vae aspect metadata"` impedirá que o trainer escaneie o dataset na inicialização, economizando muito tempo.

### `conditioning`

- **Valores:** Array de objetos de configuração de condicionamento
- **Descrição:** Gera automaticamente datasets de condicionamento a partir das suas imagens de origem. Cada tipo de condicionamento cria um dataset separado que pode ser usado para treinamento de ControlNet ou outras tarefas de condicionamento.
- **Nota:** Quando especificado, o SimpleTuner criará automaticamente datasets de condicionamento com IDs como `{source_id}_conditioning_{type}`

Cada objeto de condicionamento pode conter:
- `type`: O tipo de condicionamento a gerar (obrigatório)
- `params`: Parâmetros específicos do tipo (opcional)
- `captions`: Estratégia de caption para o dataset gerado (opcional)
  - Pode ser `false` (sem captions)
  - Uma string única (usada como instance prompt para todas as imagens)
  - Um array de strings (selecionado aleatoriamente para cada imagem)
  - Se omitido, as captions do dataset de origem são usadas

#### Tipos de condicionamento disponíveis

##### `superresolution`
Gera versões de baixa qualidade das imagens para treinamento de super-resolução:
```json
{
  "type": "superresolution",
  "blur_radius": 2.5,
  "blur_type": "gaussian",
  "add_noise": true,
  "noise_level": 0.03,
  "jpeg_quality": 85,
  "downscale_factor": 2
}
```

##### `jpeg_artifacts`
Cria artefatos de compressão JPEG para treinamento de remoção de artefatos:
```json
{
  "type": "jpeg_artifacts",
  "quality_mode": "range",
  "quality_range": [10, 30],
  "compression_rounds": 1,
  "enhance_blocks": false
}
```

##### `depth` / `depth_midas`
Gera mapas de profundidade usando modelos DPT:
```json
{
  "type": "depth_midas",
  "model_type": "DPT"
}
```
**Nota:** A geração de profundidade exige GPU e roda no processo principal, o que pode ser mais lento do que geradores baseados em CPU.

##### `random_masks` / `inpainting`
Cria máscaras aleatórias para treinamento de inpainting:
```json
{
  "type": "random_masks",
  "mask_types": ["rectangle", "circle", "brush", "irregular"],
  "min_coverage": 0.1,
  "max_coverage": 0.5,
  "output_mode": "mask"
}
```

##### `canny` / `edges`
Gera mapas de detecção de bordas Canny:
```json
{
  "type": "canny",
  "low_threshold": 100,
  "high_threshold": 200
}
```

Veja [o guia do ControlNet](CONTROLNET.md) para mais detalhes sobre como usar esses datasets de condicionamento.

#### Exemplos

##### Dataset de vídeo

Um dataset de vídeo deve ser uma pasta de arquivos de vídeo (ex.: mp4) e os métodos usuais de armazenar captions.

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

- Na subseção `video`, temos as seguintes chaves que podemos definir:
  - `num_frames` (opcional, int) é quantos frames de dados vamos treinar.
    - A 25 fps, 125 frames são 5 segundos de vídeo, saída padrão. Este deve ser seu alvo.
  - `min_frames` (opcional, int) determina o comprimento mínimo de um vídeo que será considerado para treinamento.
    - Isso deve ser pelo menos igual a `num_frames`. Não definir garante que ele será igual.
  - `max_frames` (opcional, int) determina o comprimento máximo de um vídeo que será considerado para treinamento.
  - `is_i2v` (opcional, bool) determina se o treinamento i2v será feito em um dataset.
    - Isso é definido como True por padrão para LTX. Você pode desativar, porém.
  - `bucket_strategy` (opcional, string) determina como os vídeos são agrupados em buckets:
    - `aspect_ratio` (padrão): Agrupa apenas pela proporção espacial (ex.: `1.78`, `0.75`). Mesmo comportamento que datasets de imagem.
    - `resolution_frames`: Agrupa por resolução e contagem de frames no formato `WxH@F` (ex.: `1920x1080@125`). Útil para treinar em datasets com resoluções e durações variadas.
  - `frame_interval` (opcional, int) ao usar `bucket_strategy: "resolution_frames"`, a contagem de frames é arredondada para baixo para o múltiplo mais próximo desse valor. Defina isso como o fator de contagem de frames exigido pelo seu modelo (alguns modelos exigem que `num_frames - 1` seja divisível por um certo valor).

**Ajuste Automático de Contagem de Frames:** SimpleTuner ajusta automaticamente as contagens de frames de vídeo para atender aos requisitos específicos do modelo. Por exemplo, LTX-2 requer contagens de frames que satisfaçam `frames % 8 == 1` (ex.: 49, 57, 65, 73, 81, etc.). Se seus vídeos tiverem contagens de frames diferentes (ex.: 119 frames), eles serão automaticamente cortados para a contagem de frames válida mais próxima (ex.: 113 frames). Vídeos que ficam mais curtos que `min_frames` após o ajuste são ignorados com uma mensagem de aviso. Este ajuste automático previne erros de treinamento e não requer nenhuma configuração da sua parte.

**Nota:** Ao usar `bucket_strategy: "resolution_frames"` com `num_frames` definido, você terá um único bucket de frames e vídeos menores que `num_frames` serão descartados. Remova `num_frames` se você quiser múltiplos buckets de frames com menos descartes.

Exemplo usando bucketing `resolution_frames` para datasets de vídeo com resolução mista:

```json
{
  "id": "mixed-resolution-videos",
  "type": "local",
  "dataset_type": "video",
  "resolution": 720,
  "resolution_type": "pixel_area",
  "instance_data_dir": "datasets/videos",
  "video": {
      "bucket_strategy": "resolution_frames",
      "frame_interval": 25,
      "min_frames": 25,
      "max_frames": 250
  }
}
```

Essa configuração criará buckets como `1280x720@100`, `1920x1080@125`, `640x480@75`, etc. Os vídeos são agrupados pela resolução de treinamento e contagem de frames (arredondada para múltiplos de 25 frames).


##### Configuração
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel"
```
##### Resultado
- Qualquer imagem com lado menor inferior a **1024px** será completamente excluída do treinamento.
- Imagens como `768x1024` ou `1280x768` seriam excluídas, mas `1760x1024` e `1024x1024` não.
- Nenhuma imagem será ampliada, porque `minimum_image_size` é igual a `resolution`

##### Configuração
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel_area" # diferente da configuração acima, que é 'pixel'
```
##### Resultado
- A área total da imagem (largura * altura) sendo menor que a área mínima (1024 * 1024) resultará na exclusão do treinamento.
- Imagens como `1280x960` **não** seriam excluídas porque `(1280 * 960)` é maior que `(1024 * 1024)`
- Nenhuma imagem será ampliada, porque `minimum_image_size` é igual a `resolution`

##### Configuração
```json
    "minimum_image_size": 0, # ou completamente não definido, ausente na configuração
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": false
```

##### Resultado
- As imagens serão redimensionadas para que o lado menor seja 1024px mantendo a proporção
- Nenhuma imagem será excluída por tamanho
- Imagens pequenas serão ampliadas usando métodos ingênuos de `PIL.resize` que não ficam bons
  - É recomendado evitar upscaling, a menos que seja feito manualmente com um upscaler de sua escolha antes de iniciar o treinamento

### `maximum_image_size` e `target_downsample_size`

As imagens não são redimensionadas antes do corte **a menos que** `maximum_image_size` e `target_downsample_size` estejam ambos definidos. Em outras palavras, uma imagem `4096x4096` será cortada diretamente para um alvo `1024x1024`, o que pode ser indesejável.

- `maximum_image_size` especifica o limite em que o redimensionamento começará. Ele fará downsample das imagens antes do corte se elas forem maiores que isso.
- `target_downsample_size` especifica o quão grande a imagem ficará após o reamostramento e antes do corte.

#### Exemplos

##### Configuração
```json
    "resolution_type": "pixel_area",
    "resolution": 1024,
    "maximum_image_size": 1536,
    "target_downsample_size": 1280,
    "crop": true,
    "crop_aspect": "square"
```

##### Resultado
- Qualquer imagem com área de pixels maior que `(1536 * 1536)` será redimensionada para que sua área de pixels fique aproximadamente `(1280 * 1280)` mantendo a proporção original
- O tamanho final da imagem será um corte aleatório para uma área de pixels `(1024 * 1024)`
- Útil para treinar em datasets, por exemplo, de 20 megapixels que precisam ser redimensionados consideravelmente antes do corte para evitar grande perda de contexto na imagem (como cortar uma foto de uma pessoa para apenas um azulejo de parede ou uma seção borrada do fundo)

### `max_upscale_threshold`

Por padrão, o SimpleTuner fará upscaling de imagens pequenas para atender à resolução alvo, o que pode resultar em degradação de qualidade. A opção `max_upscale_threshold` permite limitar esse comportamento de upscaling.

- **Padrão:** `null` (permite upscaling ilimitado)
- **Quando definido:** Filtra buckets de aspecto que exigiriam upscaling além do limite especificado
- **Faixa de valor:** Entre 0 e 1 (ex.: `0.2` = permitir até 20% de upscaling)
- **Aplica-se a:** Seleção de bucket de aspecto quando `crop_aspect` é definido como `closest` ou `random`

#### Exemplos

##### Configuração
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": null
```

##### Resultado
- Todos os buckets de aspecto estão disponíveis para seleção
- Uma imagem 256x256 pode ser ampliada para 1024x1024 (escala 4x)
- Pode resultar em degradação de qualidade para imagens muito pequenas

##### Configuração
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": 0.2
```

##### Resultado
- Apenas buckets de aspecto que exigem ≤20% de upscaling estão disponíveis
- Uma imagem 256x256 tentando escalar para 1024x1024 (4x = 300% de upscaling) não teria buckets disponíveis
- Uma imagem 850x850 pode usar todos os buckets, já que 1024/850 ≈ 1.2 (20% de upscaling)
- Ajuda a manter a qualidade do treinamento ao excluir imagens muito ampliadas

---

### `prepend_instance_prompt`

- Quando habilitado, todas as captions incluirão o valor de `instance_prompt` no início.

### `only_instance_prompt`

- Além de `prepend_instance_prompt`, substitui todas as captions do dataset por uma única frase ou palavra de ativação.

### `repeats`

- Especifica quantas vezes todas as amostras do dataset são vistas durante uma época. Útil para dar mais impacto a datasets menores ou maximizar o uso de objetos de cache do VAE.
- Se você tem um dataset com 1000 imagens vs um com 100 imagens, você provavelmente vai querer dar ao dataset menor um `repeats` de `9` **ou maior** para levá-lo a 1000 imagens totais amostradas.

> ℹ️ Este valor se comporta de maneira diferente da mesma opção nos scripts do Kohya, onde um valor de 1 significa sem repetições. **No SimpleTuner, um valor de 0 significa sem repetições**. Subtraia um do valor do seu config do Kohya para obter o equivalente no SimpleTuner; portanto um valor **9** resulta do cálculo `(dataset_length + repeats * dataset_length)` .

#### Treinamento multi-GPU e dimensionamento de dataset

Ao treinar com múltiplas GPUs, seu dataset deve ser grande o suficiente para acomodar o **tamanho efetivo do batch**, calculado como:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

Por exemplo, com 4 GPUs, `train_batch_size=4` e `gradient_accumulation_steps=1`, você precisa de pelo menos **16 amostras** (após aplicar repeats) em cada bucket de aspecto.

**Importante:** O SimpleTuner lançará um erro se sua configuração de dataset produzir zero batches utilizáveis. A mensagem de erro mostrará:
- Valores atuais de configuração (batch size, contagem de GPUs, repeats)
- Quais buckets de aspecto têm amostras insuficientes
- Repeats mínimos exatos necessários para cada bucket
- Soluções sugeridas

##### Oversubscription automática de dataset

Para ajustar automaticamente `repeats` quando seu dataset é menor que o tamanho efetivo do batch, use a flag `--allow_dataset_oversubscription` (documentada em [OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription)).

Quando habilitado, o SimpleTuner irá:
- Calcular o mínimo de repeats necessário para o treinamento
- Aumentar automaticamente `repeats` para atender ao requisito
- Registrar um aviso mostrando o ajuste
- **Respeitar valores de repeats definidos manualmente** - se você configurar explicitamente `repeats` no seu config de dataset, o ajuste automático será ignorado

Isso é particularmente útil quando:
- Treinar datasets pequenos (< 100 imagens)
- Usar muitos GPUs com datasets pequenos
- Experimentar diferentes tamanhos de batch sem reconfigurar datasets

**Exemplo de cenário:**
- Dataset: 25 imagens
- Configuração: 8 GPUs, `train_batch_size=4`, `gradient_accumulation_steps=1`
- Tamanho efetivo do batch: 32 amostras necessárias
- Sem oversubscription: Erro lançado
- Com `--allow_dataset_oversubscription`: Repeats ajustado automaticamente para 1 (25 × 2 = 50 amostras)

### `max_num_samples`

- **Descrição:** Limita o dataset a um número máximo de amostras. Quando definido, um subconjunto aleatório determinístico do tamanho especificado é selecionado do dataset completo.
- **Caso de uso:** Útil para grandes datasets de regularização onde você deseja usar apenas uma parte dos dados para evitar sobrecarregar conjuntos de treinamento menores.
- **Seleção determinística:** A seleção aleatória usa o `id` do dataset como semente, garantindo que o mesmo subconjunto seja selecionado entre sessões de treinamento para reprodutibilidade.
- **Padrão:** `null` (sem limite, todas as amostras são usadas)

#### Exemplo
```json
{
  "id": "regularization-data",
  "max_num_samples": 1000,
  ...
}
```

Isso selecionará deterministicamente 1000 amostras do dataset, com a mesma seleção usada toda vez que o treinamento for executado.

### `start_epoch` / `start_step`

- Agenda quando um dataset começa a amostrar.
- `start_epoch` (padrão: `1`) condiciona pelo número de épocas; `start_step` (padrão: `0`) condiciona pelo step do otimizador (após gradient accumulation). Ambas as condições devem ser satisfeitas antes de amostrar.
- Pelo menos um dataset deve ter `start_epoch<=1` **e** `start_step<=1`; caso contrário, o treinamento dará erro porque não há dados disponíveis na inicialização.
- Datasets que nunca atendem sua condição de início (por exemplo, `start_epoch` além de `--num_train_epochs`) serão ignorados e anotados no model card.
- Estimativas de steps na barra de progresso são aproximadas quando datasets programados ativam no meio da execução, porque o tamanho da época pode aumentar quando novos dados entram.

### `end_epoch` / `end_step`

- Agenda quando um dataset **para** de amostrar (complementando `start_epoch`/`start_step`).
- `end_epoch` (padrão: `null` = sem limite) para de amostrar após esta época; `end_step` (padrão: `null` = sem limite) para de amostrar após este step do otimizador.
- Qualquer uma das condições que terminar irá parar o dataset; funcionam independentemente.
- Útil para workflows de **aprendizado curricular** onde você deseja:
  - Treinar com dados de baixa resolução primeiro, depois mudar para dados de maior resolução.
  - Eliminar gradualmente dados de regularização após certo ponto.
  - Criar treinamento multi-estágio em um único arquivo de configuração.

**Exemplo: Aprendizado Curricular**
```json
[
  {
    "id": "lowres-512",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/512",
    "end_step": 300
  },
  {
    "id": "highres-1024",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/1024",
    "start_step": 300
  }
]
```

Neste exemplo, o dataset de 512px é usado para steps 1-300, então o dataset de 1024px assume a partir do step 300 em diante.

### `is_regularisation_data`

- Também pode ser escrito como `is_regularization_data`
- Habilita treinamento parent-teacher para adapters LyCORIS para que o alvo de predição prefira o resultado do modelo base para um dado dataset.
  - LoRA padrão não é suportado no momento.

### `delete_unwanted_images`

- **Valores:** `true` | `false`
- **Descrição:** Quando habilitado, imagens que falham nos filtros de tamanho ou proporção (ex.: abaixo de `minimum_image_size` ou fora de `minimum_aspect_ratio`/`maximum_aspect_ratio`) são permanentemente deletadas do diretório do dataset.
- **Aviso:** Isso é destrutivo e não pode ser desfeito. Use com cuidado.
- **Padrão:** Usa o argumento do trainer `--delete_unwanted_images` (padrão: `false`).

### `delete_problematic_images`

- **Valores:** `true` | `false`
- **Descrição:** Quando habilitado, imagens que falham durante a codificação VAE (arquivos corrompidos, formatos não suportados, etc.) são permanentemente deletadas do diretório do dataset.
- **Aviso:** Isso é destrutivo e não pode ser desfeito. Use com cuidado.
- **Padrão:** Usa o argumento do trainer `--delete_problematic_images` (padrão: `false`).

### Visualizando Estatísticas de Filtragem

Quando o SimpleTuner processa seu dataset, ele rastreia quantos arquivos foram filtrados e por quê. Essas estatísticas são armazenadas no arquivo de cache do dataset (`aspect_ratio_bucket_indices_*.json`) e podem ser visualizadas na WebUI.

**Estatísticas rastreadas:**
- **total_processed**: Número de arquivos processados
- **too_small**: Arquivos filtrados por estarem abaixo de `minimum_image_size`
- **too_long**: Arquivos filtrados por excederem limites de duração (áudio/vídeo)
- **metadata_missing**: Arquivos ignorados por falta de metadados
- **not_found**: Arquivos que não puderam ser localizados
- **already_exists**: Arquivos já no cache (não reprocessados)
- **other**: Arquivos filtrados por outros motivos

**Visualizando na WebUI:**

Ao navegar pelos datasets no navegador de arquivos da WebUI, selecionar um diretório com um dataset existente exibirá estatísticas de filtragem, se disponíveis. Isso ajuda a diagnosticar por que seu dataset pode ter menos amostras utilizáveis do que o esperado.

**Solucionando problemas de arquivos filtrados:**

Se muitos arquivos estão sendo filtrados como `too_small`:
1. Verifique sua configuração de `minimum_image_size` — deve corresponder a `resolution` e `resolution_type`
2. Para `resolution_type=pixel`, `minimum_image_size` é o comprimento mínimo da borda mais curta
3. Para `resolution_type=area` ou `pixel_area`, `minimum_image_size` é a área total mínima

Veja a seção [Solução de Problemas](#solucionando-problemas-de-datasets-filtrados) abaixo para mais detalhes.

### `slider_strength`

- **Valores:** Qualquer valor float (positivo, negativo ou zero)
- **Descrição:** Marca um dataset para treinamento de slider LoRA, que aprende "opostos" contrastivos para criar adapters de conceitos controláveis.
  - **Valores positivos** (ex.: `0.5`): "Mais do conceito" — olhos mais brilhantes, sorriso mais forte, etc.
  - **Valores negativos** (ex.: `-0.5`): "Menos do conceito" — olhos mais opacos, expressão neutra, etc.
  - **Zero ou omitido**: Exemplos neutros que não empurram o conceito em nenhuma direção.
- **Nota:** Quando datasets têm valores de `slider_strength`, o SimpleTuner alterna batches em um ciclo fixo: positivo → negativo → neutro. Dentro de cada grupo, as probabilidades padrão do backend ainda se aplicam.
- **Veja também:** [SLIDER_LORA.md](SLIDER_LORA.md) para um guia completo sobre configuração de treinamento de slider LoRA.

### `vae_cache_clear_each_epoch`

- Quando habilitado, todos os objetos de cache do VAE são deletados do sistema de arquivos no final de cada ciclo de repeats do dataset. Isso pode ser intensivo em recursos para datasets grandes, mas combinado com `crop_style=random` e/ou `crop_aspect=random` você vai querer isso habilitado para garantir que uma faixa completa de cortes seja amostrada de cada imagem.
- Na verdade, esta opção é **habilitada por padrão** ao usar bucketing ou cortes aleatórios.

### `vae_cache_disable`

- **Valores:** `true` | `false`
- **Descrição:** Quando habilitado (via argumento de linha de comando `--vae_cache_disable`), esta opção habilita implicitamente o cache VAE sob demanda, mas desativa a gravação dos embeddings gerados em disco. Isso é útil para datasets grandes em que espaço em disco é uma preocupação ou a escrita é impraticável.
- **Nota:** Este é um argumento de nível do trainer, não uma opção por dataset, mas afeta como o dataloader interage com o cache do VAE.

### `skip_file_discovery`

- Você provavelmente não quer definir isso - ele é útil apenas para datasets muito grandes.
- Este parâmetro aceita uma lista separada por vírgulas ou espaços de valores, por exemplo `vae metadata aspect text` para pular a descoberta de arquivos em uma ou mais etapas da configuração do loader.
- Isso é equivalente à opção de linha de comando `--skip_file_discovery`
- Isso é útil se você tiver datasets que não precisam ser escaneados pelo trainer a cada inicialização, por exemplo, seus latentes/embeds já estão totalmente em cache. Isso permite uma inicialização e retomada de treinamento mais rápidas.

### `preserve_data_backend_cache`

- Você provavelmente não quer definir isso - ele é útil apenas para datasets AWS muito grandes.
- Assim como `skip_file_discovery`, esta opção pode ser definida para evitar escaneamentos de sistema de arquivos desnecessários, longos e caros na inicialização.
- Ela recebe um valor booleano, e se definida como `true`, o arquivo de cache da lista do sistema de arquivos gerado não será removido no lançamento.
- Isso é útil para sistemas de armazenamento muito grandes e lentos, como S3 ou discos rígidos locais SMR com tempos de resposta extremamente lentos.
- Além disso, no S3, a listagem do backend pode acumular custos e deve ser evitada.

> ⚠️ **Infelizmente, isso não pode ser definido se os dados estiverem sendo alterados ativamente.** O trainer não verá novos dados adicionados ao pool; será necessário fazer outra varredura completa.

### `hash_filenames`

- Os nomes de arquivos das entradas do cache do VAE são sempre hash. Isso não é configurável pelo usuário e garante que datasets com nomes de arquivo muito longos possam ser usados facilmente sem problemas de comprimento de caminho. Qualquer configuração `hash_filenames` no seu arquivo será ignorada.

## Filtragem de captions

### `caption_filter_list`

- **Apenas para datasets de text embeds.** Pode ser uma lista JSON, um caminho para um arquivo txt ou um caminho para um documento JSON. Strings de filtro podem ser termos simples para remover de todas as captions, ou podem ser expressões regulares. Além disso, entradas no estilo sed `s/search/replace/` podem ser usadas para _substituir_ strings na caption em vez de simplesmente removê-las.

#### Exemplo de lista de filtros

Um exemplo completo pode ser encontrado [aqui](/config/caption_filter_list.txt.example). Ele contém strings repetitivas e negativas comuns que seriam retornadas por BLIP (variações comuns), LLaVA e CogVLM.

Este é um exemplo reduzido, que será explicado abaixo:

```
arafed
this .* has a
^this is the beginning of the string
s/this/will be found and replaced/
```

Em ordem, as linhas se comportam da seguinte forma:

- `arafed ` (com um espaço ao final) será removido de qualquer caption em que for encontrado. Incluir um espaço ao final faz a caption ficar mais bonita, já que não sobrará espaço duplo. Isso não é necessário, mas fica bonito.
- `this .* has a` é uma expressão regular que removerá qualquer coisa que contenha "this ... has a", incluindo qualquer texto entre essas duas strings; `.*` é uma expressão regular que significa "tudo o que encontramos" até encontrar a string "has a", quando para de casar.
- `^this is the beginning of the string` removerá a frase "this is the beginning of the string" de qualquer caption, mas apenas quando ela aparecer no início da caption.
- `s/this/will be found and replaced/` fará com que a primeira ocorrência do termo "this" em qualquer caption seja substituída por "will be found and replaced".

> ❗Use [regex 101](https://regex101.com) para ajuda ao depurar e testar expressões regulares.

# Técnicas avançadas

## Exemplo avançado de configuração

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": false,
    "crop_style": "random|center|corner|face",
    "crop_aspect": "square|preserve|closest|random",
    "crop_aspect_buckets": [0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    "resolution": 1.0,
    "resolution_type": "area|pixel",
    "minimum_image_size": 1.0,
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "filename|instanceprompt|parquet|textfile",
    "disable_multiline_split": false,
    "cache_dir_vae": "/path/to/vaecache",
    "vae_cache_clear_each_epoch": true,
    "probability": 1.0,
    "repeats": 0,
    "start_epoch": 1,
    "start_step": 0,
    "text_embeds": "alt-embed-cache",
    "image_embeds": "vae-embeds-example",
    "conditioning_image_embeds": "conditioning-embeds-example"
  },
  {
    "id": "another-special-name-for-another-backend",
    "type": "aws",
    "aws_bucket_name": "something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir_vae": "s3prefix/for/vaecache",
    "vae_cache_clear_each_epoch": true,
    "repeats": 0
  },
  {
      "id": "vae-embeds-example",
      "type": "local",
      "dataset_type": "image_embeds",
      "disabled": false,
  },
  {
      "id": "conditioning-embeds-example",
      "type": "local",
      "dataset_type": "conditioning_image_embeds",
      "disabled": false
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  },
  {
    "id": "alt-embed-cache",
    "dataset_type": "text_embeds",
    "default": false,
    "type": "local",
    "cache_dir": "/path/to/textembed_cache"
  }
]
```

## Treinar diretamente a partir de uma lista de URLs em CSV

**Nota: Seu CSV deve conter as captions para suas imagens.**

> ⚠️ Este é um recurso avançado **e** experimental, e você pode encontrar problemas. Se isso acontecer, abra uma [issue](https://github.com/bghira/simpletuner/issues)!

Em vez de baixar seus dados manualmente a partir de uma lista de URLs, talvez você queira conectá-los diretamente ao trainer.

**Nota:** É sempre melhor baixar os dados de imagem manualmente. Outra estratégia para economizar espaço em disco local pode ser tentar [usar armazenamento em nuvem com caches locais de encoder](#local-cache-with-cloud-dataset).

### Vantagens

- Não é necessário baixar os dados diretamente
- Pode usar o toolkit de captions do SimpleTuner para legendar diretamente a lista de URLs
- Economiza espaço em disco, já que apenas image embeds (se aplicável) e text embeds são armazenados

### Desvantagens

- Requer uma varredura de buckets de aspecto custosa e potencialmente lenta, onde cada imagem é baixada e seus metadados coletados
- As imagens baixadas são armazenadas em cache no disco, o que pode crescer muito. Esta é uma área de melhoria, pois o gerenciamento de cache nesta versão é bem básico, apenas grava e nunca deleta
- Se seu dataset tiver muitas URLs inválidas, isso pode desperdiçar tempo ao retomar pois, atualmente, amostras ruins **nunca** são removidas da lista de URLs
  - **Sugestão:** Execute uma tarefa de validação de URLs antes e remova amostras ruins.

### Configuração

Chaves obrigatórias:

- `type: "csv"`
- `csv_caption_column`
- `csv_cache_dir`
- `caption_strategy: "csv"`

```json
[
    {
        "id": "csvtest",
        "type": "csv",
        "csv_caption_column": "caption",
        "csv_file": "/Volumes/ml/dataset/test_list.csv",
        "csv_cache_dir": "/Volumes/ml/cache/csv/test",
        "cache_dir_vae": "/Volumes/ml/cache/vae/sdxl",
        "caption_strategy": "csv",
        "image_embeds": "image-embeds",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel",
        "minimum_image_size": 0,
        "disabled": false,
        "skip_file_discovery": "",
        "preserve_data_backend_cache": false
    },
    {
      "id": "image-embeds",
      "type": "local"
    },
    {
        "id": "text-embeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/Volumes/ml/cache/text/sdxl",
        "disabled": false,
        "preserve_data_backend_cache": false,
        "skip_file_discovery": "",
        "write_batch_size": 128
    }
]
```

## Estratégia de captions Parquet e datasets JSON Lines

> ⚠️ Este é um recurso avançado e não será necessário para a maioria dos usuários.

Ao treinar um modelo com um dataset muito grande, com centenas de milhares ou milhões de imagens, é mais rápido armazenar seus metadados dentro de um banco parquet em vez de arquivos txt - especialmente quando seus dados de treinamento são armazenados em um bucket S3.

Usar a estratégia de captions parquet permite nomear todos os seus arquivos pelo valor `id` e alterar a coluna de caption via um valor de configuração em vez de atualizar muitos arquivos de texto ou ter que renomear os arquivos para atualizar suas captions.

Aqui está um exemplo de configuração de dataloader que faz uso das captions e dos dados do dataset [photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket):

```json
{
  "id": "photo-concept-bucket",
  "type": "local",
  "instance_data_dir": "/models/training/datasets/photo-concept-bucket-downloads",
  "caption_strategy": "parquet",
  "metadata_backend": "parquet",
  "parquet": {
    "path": "photo-concept-bucket.parquet",
    "filename_column": "id",
    "caption_column": "cogvlm_caption",
    "fallback_caption_column": "tags",
    "width_column": "width",
    "height_column": "height",
    "identifier_includes_extension": false
  },
  "resolution": 1.0,
  "minimum_image_size": 0.75,
  "maximum_image_size": 2.0,
  "target_downsample_size": 1.5,
  "prepend_instance_prompt": false,
  "instance_prompt": null,
  "only_instance_prompt": false,
  "disable": false,
  "cache_dir_vae": "/models/training/vae_cache/photo-concept-bucket",
  "probability": 1.0,
  "skip_file_discovery": "",
  "preserve_data_backend_cache": false,
  "vae_cache_clear_each_epoch": true,
  "repeats": 1,
  "crop": true,
  "crop_aspect": "closest",
  "crop_style": "random",
  "crop_aspect_buckets": [1.0, 0.75, 1.23],
  "resolution_type": "area"
}
```

Nesta configuração:

- `caption_strategy` é definido como `parquet`.
- `metadata_backend` é definido como `parquet`.
- Uma nova seção, `parquet`, deve ser definida:
  - `path` é o caminho para o arquivo parquet ou JSONL.
  - `filename_column` é o nome da coluna na tabela que contém os nomes dos arquivos. Neste caso, usamos a coluna numérica `id` (recomendado).
  - `caption_column` é o nome da coluna na tabela que contém as captions. Neste caso, usamos a coluna `cogvlm_caption`. Para datasets LAION, isso seria o campo TEXT.
  - `width_column` e `height_column` podem ser colunas contendo strings, int, ou até um tipo de dados Series de uma única entrada, medindo as dimensões reais da imagem. Isso melhora notavelmente o tempo de preparação do dataset, pois não precisamos acessar as imagens reais para descobrir essa informação.
  - `fallback_caption_column` é um nome opcional de uma coluna na tabela que contém captions de fallback. Elas são usadas se o campo de caption principal estiver vazio. Neste caso, usamos a coluna `tags`.
  - `identifier_includes_extension` deve ser definido como `true` quando sua coluna de nome de arquivo contém a extensão da imagem. Caso contrário, a extensão será assumida como `.png`. É recomendado incluir extensões de arquivo na coluna de nome de arquivo da tabela.

> ⚠️ O suporte a parquet é limitado à leitura de captions. Você deve preencher separadamente uma fonte de dados com suas amostras de imagem usando "{id}.png" como nome do arquivo. Veja scripts no diretório [scripts/toolkit/datasets](scripts/toolkit/datasets) para ideias.

Como em outras configurações de dataloader:

- `prepend_instance_prompt` e `instance_prompt` se comportam normalmente.
- Atualizar a caption de uma amostra entre execuções de treinamento vai armazenar o novo embed em cache, mas não remover o antigo (órfão).
- Quando uma imagem não existe em um dataset, seu nome de arquivo será usado como sua caption e um erro será emitido.

## Cache local com dataset na nuvem

Para maximizar o uso de armazenamento local NVMe caro, talvez você queira armazenar apenas os arquivos de imagem (png, jpg) em um bucket S3 e usar o armazenamento local para fazer cache dos mapas de recursos extraídos do(s) text encoder(s) e do VAE (se aplicável).

Neste exemplo de configuração:

- Dados de imagem são armazenados em um bucket compatível com S3
- Dados do VAE são armazenados em /local/path/to/cache/vae
- Text embeds são armazenados em /local/path/to/cache/textencoder

> ⚠️ Lembre-se de configurar as outras opções do dataset, como `resolution` e `crop`

```json
[
    {
        "id": "data",
        "type": "aws",
        "aws_bucket_name": "text-vae-embeds",
        "aws_endpoint_url": "https://storage.provider.example",
        "aws_access_key_id": "exampleAccessKey",
        "aws_secret_access_key": "exampleSecretKey",
        "aws_region_name": null,
        "cache_dir_vae": "/local/path/to/cache/vae/",
        "caption_strategy": "parquet",
        "metadata_backend": "parquet",
        "parquet": {
            "path": "train.parquet",
            "caption_column": "caption",
            "filename_column": "filename",
            "width_column": "width",
            "height_column": "height",
            "identifier_includes_extension": true
        },
        "preserve_data_backend_cache": false,
        "image_embeds": "vae-embed-storage"
    },
    {
        "id": "vae-embed-storage",
        "type": "local",
        "dataset_type": "image_embeds"
    },
    {
        "id": "text-embed-storage",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/local/path/to/cache/textencoder/",
        "write_batch_size": 128
    }
]
```

**Nota:** O dataset `image_embeds` não tem opções para definir caminhos de dados. Eles são configurados via `cache_dir_vae` no backend de imagem.

### Suporte a Hugging Face Datasets

O SimpleTuner agora suporta carregar datasets diretamente do Hugging Face Hub sem baixar o dataset inteiro localmente. Esse recurso experimental é ideal para:

- Datasets em grande escala hospedados no Hugging Face
- Datasets com metadados e avaliações de qualidade embutidos
- Experimentação rápida sem requisitos de armazenamento local

Para documentação completa sobre este recurso, consulte [este documento](HUGGINGFACE_DATASETS.md).

Para um exemplo básico de como usar um dataset do Hugging Face, defina `"type": "huggingface"` na sua configuração de dataloader:

```json
{
  "id": "my-hf-dataset",
  "type": "huggingface",
  "dataset_name": "username/dataset-name",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "caption",
  "image_column": "image"
}
```

## Mapeamento personalizado de proporção para resolução

Quando o SimpleTuner inicia pela primeira vez, ele gera listas de mapeamento de aspecto específicas de resolução que vinculam um valor decimal de proporção a seu tamanho de pixel alvo.

É possível criar um mapeamento personalizado que force o trainer a se ajustar à sua resolução alvo escolhida em vez de seus próprios cálculos. Essa funcionalidade é fornecida por sua conta e risco, pois pode causar grande dano se configurada incorretamente.

Para criar o mapeamento personalizado:

- Crie um arquivo que siga o exemplo (abaixo)
- Nomeie o arquivo usando o formato `aspect_ratio_map-{resolution}.json`
  - Para um valor de configuração `resolution=1.0` / `resolution_type=area`, o nome do arquivo de mapeamento será `aspect_resolution_map-1.0.json`
- Coloque esse arquivo no local especificado como `--output_dir`
  - Este é o mesmo local onde seus checkpoints e imagens de validação serão encontrados.
- Nenhum flag ou opção adicional de configuração é necessário. Ele será descoberto e usado automaticamente, desde que o nome e o local estejam corretos.

### Exemplo de configuração de mapeamento

Este é um exemplo de mapeamento de proporção gerado pelo SimpleTuner. Você não precisa configurá-lo manualmente, pois o trainer criará um automaticamente. No entanto, para controle total sobre as resoluções resultantes, esses mapeamentos são fornecidos como ponto de partida para modificação.

- O dataset tinha mais de 1 milhão de imagens
- O `resolution` do dataloader foi definido como `1.0`
- O `resolution_type` do dataloader foi definido como `area`

Esta é a configuração mais comum e a lista de buckets de aspecto treináveis para um modelo de 1 megapixel.

```json
{
    "0.07": [320, 4544],    "0.38": [640, 1664],    "0.88": [960, 1088],    "1.92": [1472, 768],    "3.11": [1792, 576],    "5.71": [2560, 448],
    "0.08": [320, 3968],    "0.4": [640, 1600],     "0.89": [1024, 1152],   "2.09": [1472, 704],    "3.22": [1856, 576],    "6.83": [2624, 384],
    "0.1": [320, 3328],     "0.41": [704, 1728],    "0.94": [1024, 1088],   "2.18": [1536, 704],    "3.33": [1920, 576],    "7.0": [2688, 384],
    "0.11": [384, 3520],    "0.42": [704, 1664],    "1.06": [1088, 1024],   "2.27": [1600, 704],    "3.44": [1984, 576],    "8.0": [3072, 384],
    "0.12": [384, 3200],    "0.44": [704, 1600],    "1.12": [1152, 1024],   "2.5": [1600, 640],     "3.88": [1984, 512],
    "0.14": [384, 2688],    "0.46": [704, 1536],    "1.13": [1088, 960],    "2.6": [1664, 640],     "4.0": [2048, 512],
    "0.15": [448, 3008],    "0.48": [704, 1472],    "1.2": [1152, 960],     "2.7": [1728, 640],     "4.12": [2112, 512],
    "0.16": [448, 2816],    "0.5": [768, 1536],     "1.36": [1216, 896],    "2.8": [1792, 640],     "4.25": [2176, 512],
    "0.19": [448, 2304],    "0.52": [768, 1472],    "1.46": [1216, 832],    "3.11": [1792, 576],    "4.38": [2240, 512],
    "0.24": [512, 2112],    "0.55": [768, 1408],    "1.54": [1280, 832],    "3.22": [1856, 576],    "5.0": [2240, 448],
    "0.26": [512, 1984],    "0.59": [832, 1408],    "1.83": [1408, 768],    "3.33": [1920, 576],    "5.14": [2304, 448],
    "0.29": [576, 1984],    "0.62": [832, 1344],    "1.92": [1472, 768],    "3.44": [1984, 576],    "5.71": [2560, 448],
    "0.31": [576, 1856],    "0.65": [832, 1280],    "2.09": [1472, 704],    "3.88": [1984, 512],    "6.83": [2624, 384],
    "0.34": [640, 1856],    "0.68": [832, 1216],    "2.18": [1536, 704],    "4.0": [2048, 512],     "7.0": [2688, 384],
    "0.38": [640, 1664],    "0.74": [896, 1216],    "2.27": [1600, 704],    "4.12": [2112, 512],    "8.0": [3072, 384],
    "0.4": [640, 1600],     "0.83": [960, 1152],    "2.5": [1600, 640],     "4.25": [2176, 512],
    "0.41": [704, 1728],    "0.88": [960, 1088],    "2.6": [1664, 640],     "4.38": [2240, 512],
    "0.42": [704, 1664],    "0.89": [1024, 1152],   "2.7": [1728, 640],     "5.0": [2240, 448],
    "0.44": [704, 1600],    "0.94": [1024, 1088],   "2.8": [1792, 640],     "5.14": [2304, 448]
}
```

Para modelos Stable Diffusion 1.5 / 2.0-base (512px), o mapeamento a seguir funciona:

```json
{
    "1.3": [832, 640], "1.0": [768, 768], "2.0": [1024, 512],
    "0.64": [576, 896], "0.77": [640, 832], "0.79": [704, 896],
    "0.53": [576, 1088], "1.18": [832, 704], "0.85": [704, 832],
    "0.56": [576, 1024], "0.92": [704, 768], "1.78": [1024, 576],
    "1.56": [896, 576], "0.67": [640, 960], "1.67": [960, 576],
    "0.5": [512, 1024], "1.09": [768, 704], "1.08": [832, 768],
    "0.44": [512, 1152], "0.71": [640, 896], "1.4": [896, 640],
    "0.39": [448, 1152], "2.25": [1152, 512], "2.57": [1152, 448],
    "0.4": [512, 1280], "3.5": [1344, 384], "2.12": [1088, 512],
    "0.3": [448, 1472], "2.71": [1216, 448], "8.25": [2112, 256],
    "0.29": [384, 1344], "2.86": [1280, 448], "6.2": [1984, 320],
    "0.6": [576, 960]
}
```
