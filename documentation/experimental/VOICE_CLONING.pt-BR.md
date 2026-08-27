# Transforms de Clonagem de Voz

Transforms de clonagem de voz sao uma funcionalidade experimental planejada para datasets de audio. Eles expandem o conjunto de treino transferindo uma identidade vocal para musicas, stems ou performances adicionais antes do treino principal do modelo.

O objetivo nao e transformar o SimpleTuner em uma estacao separada de voice conversion. O objetivo e reduzir o entrelacamento no dataset de audio. Se uma cantora ou cantor aparece apenas em um estilo muito estreito, uma LoRA pode aprender "essa voz dentro desse arranjo", em vez da identidade vocal em si. Um split expandido com voice cloning pode colocar a mesma identidade vocal em arranjos, captions, letras e estruturas musicais mais variados.

Esta funcionalidade e apenas para datasets de audio.

!!! warning "Consentimento e direitos"
    Use este workflow somente com vozes e gravacoes que voce tem permissao para usar. Identidade vocal e dado biometrico e criativo sensivel. O transform pode criar audio derivado que soa como uma pessoa real, entao permissao, licenciamento e divulgacao importam.

## ELI5

Imagine que voce tem seis gravacoes de uma pessoa cantando, mas todas sao da mesma banda e do mesmo genero. Se voce treina apenas nessas musicas, o modelo pode aprender que a voz, a guitarra, a bateria, o tempo e a estrutura da musica sao uma coisa so.

Transforms de clonagem de voz tentam separar essas ideias:

1. Aprendem um pequeno modelo de conversao de voz a partir dos exemplos da voz alvo.
2. Pegam um conjunto mais amplo de musicas ou vocal stems.
3. Substituem o timbre vocal de origem pelo timbre alvo.
4. Mantem as novas captions e letras alinhadas ao audio gerado.
5. Adicionam o audio gerado como outro split normal de treino.

Assim o modelo principal ve a voz alvo em mais contextos, em vez de memorizar apenas o dataset estreito original.

## Para Que Serve

Use quando:

- voce tem gravacoes autorizadas da voz alvo
- a identidade esta muito ligada a um unico genero, banda, estilo de producao ou estrutura musical
- trigger words funcionam apenas no dominio original
- duas ou mais vozes no mesmo dataset viram uma voz media
- voce quer LoRAs separadas para identidades vocais separadas
- voce quer que o SimpleTuner prepare o split expandido dentro do mesmo setup de treino

Evite quando:

- voce ja tem um dataset grande, variado e limpo da mesma voz
- o audio fonte de expansao tem baixa qualidade ou nao bate com as captions
- voce precisa publicar resultados e nao tem direitos claros
- o modelo generativo base nao aprende a identidade alvo nem com exemplos diretos limpos

## Como Entra No Treino

Clonagem de voz e um transform de preparacao de dados, nao um dataset de conditioning.

`conditioning_data` e para entradas auxiliares pareadas que ficam anexadas a uma amostra primaria durante o treino, como imagens de referencia ou mapas de conditioning gerados.

Clonagem de voz deve ficar em uma lista `data_transforms` no nivel do dataset. O transform materializa novos arquivos de audio, captions e letras opcionais, e registra o resultado como outro dataset primario `audio`. Depois disso, o dataloader normal o ve como qualquer outro split de treino.

Formato de pseudo config:

```text
audio dataset:
    id: target-voice
    dataset_type: audio
    data_transforms:
        - task: identity_transfer
          source: expansion-audio-backend
          target: generated-audio-backend
          method: rvc
          audio_mode: separate_convert_remix
```

Comportamento de startup em pseudocodigo:

```text
for each audio dataset:
    for each data transform:
        if task is identity_transfer:
            prepare or reuse the target voice-conversion model
            prepare or reuse generated audio
            append generated audio as a normal train split

continue with normal metadata discovery, bucketing, caching, and training
```

## Transferencia de Identidade Estilo RVC

A primeira implementacao e conversao de voz estilo RVC, usando features HuBERT, pitch RMVPE, generator NSF/VITS, multi-period discriminator, losses mel/adversarial e indice de retrieval opcional.

Neste contexto, o "modelo RVC" e especifico da voz. Ele e treinado a partir do dataset de identidade alvo. O indice de recuperacao tambem e especifico da voz e e construido com features da mesma voz alvo. Componentes pre-treinados amplos, como features de conteudo, extracao de pitch ou modelos de separacao, sao infraestrutura reutilizavel; o modelo de conversao e o indice sao artifacts especificos da cantora, cantor ou locutor.

O SimpleTuner deve conseguir:

1. Reutilizar um modelo de voice conversion e indice fornecidos.
2. Treinar o modelo de voice conversion se nenhum modelo for fornecido.
3. Construir o indice de recuperacao a partir dos dados da voz alvo.
4. Cachear modelo, indice e audio gerado no diretorio de saida do treino.
5. Reutilizar artifacts em cache no startup quando os dados e settings nao mudaram.
6. Opcionalmente reutilizar ou publicar o modelo de voice conversion por um repositorio de modelo no Hub.

## Comportamento Padrao

Os defaults sao conservadores. Neste workflow, o backend de audio e a musica de expansao que sera convertida, `model.identity_data_dir` e o dataset da voz alvo, e `target.instance_data_dir` e apenas o caminho do split gerado.

| Setting | Default | Por que |
| --- | --- | --- |
| `task` | `identity_transfer` | Identifica explicitamente o transform. |
| `method` | `rvc` | Primeiro backend de transferencia vocal suportado. |
| `train_if_missing` | `true` | O SimpleTuner deve bootstrapar o modelo vocal a partir do dataset alvo. |
| `force_retrain` | `false` | Reutiliza um modelo em cache valido quando possivel. |
| `build_index` | `true` | Retrieval costuma melhorar estabilidade de identidade e reduzir vazamento. |
| `identity_data_dir` | obrigatorio no treino sob demanda | Aponta para exemplos vocais limpos da voz que sera transferida para as musicas de expansao. |
| `identity_audio_mode` | `separate` | Executa Demucs nos clips de identidade antes do treino. Use `vocal_only` se o dataset de identidade ja contem vocal stems. |
| `identity_stem_debug_dir` | nao definido | Diretorio opcional para salvar previews `vocals.wav` e `no_vocals.wav` da identidade. Use para confirmar que o RVC esta treinando com vocais isolados, nao com vazamento de instrumentos. |
| `asset_hub_model_id` | `lj1995/VoiceConversionWebUI` | Default RVC asset repository for HuBERT, RMVPE, and v2 48k pretrained generator/discriminator checkpoints. |
| `model_name` | transform or Hub repo name | Human-readable name saved into the RVC artifact so downloaded caches are identifiable outside their folder name. |
| `sample_rate` | `48000` | Current implementation targets RVC v2 48k assets. Other rates need matching pretrained assets and configs. |
| `training_steps` | `1000` | Runs RVC generator/discriminator fine-tuning during startup. Increase for larger or more varied identity datasets. |
| `batch_size` | `4` | RVC training batch size before distributed sharding. Lower it for memory pressure. |
| `learning_rate` | `1e-4` | Standard RVC AdamW default. |
| `hub_model_id` | nao definido | Nenhum cache remoto de modelo vocal e usado sem opt-in do usuario. |
| `reuse_from_hub` | `true` quando `hub_model_id` esta definido | Verifica o Hub antes de gastar tempo treinando um modelo sob demanda. |
| `push_to_hub` | `false` | Upload de modelo vocal deve ser explicito porque o artifact representa uma identidade vocal. |
| `public` | `false` | Hub uploads are private by default. Set this to `true` only when the voice artifact can be published publicly. |
| `audio_mode` | `separate_convert_remix` para musicas completas, `vocal_only` para vocal stems | Mix completo precisa de separacao; stems nao. |
| `separation_method` | `demucs` quando separacao e necessaria | Demucs e o stem separator default esperado. |
| `timbre_strength` | `1.0` | Controls how strongly the synthesized target voice replaces the source vocal. Lower values blend source and converted vocals. |
| `retrieval_strength` | `0.75` | Blends nearest target-voice content frames from the retrieval index into the generator input. |
| tipo do split gerado | dataset primario `audio` | Dados gerados treinam como audio normal, nao conditioning. |
| local de cache | dentro de `output_dir` | Mantem artifacts ligados ao treino e reutilizaveis no restart. |
| captions | copia captions da fonte salvo configuracao diferente | O novo split deve preservar letras e contexto de arranjo. |

Se um modelo de voice conversion existente for fornecido, o SimpleTuner deve usa-lo e so treinar um novo quando isso for solicitado ou quando artifacts necessarios estiverem ausentes.

## Cache no Hub

Um modelo de voice conversion pode ser caro o suficiente para que treino sob demanda repetido vire uma armadilha. O transform deve entao suportar um cache opcional no Hub para o modelo vocal e o indice de retrieval.

A ordem segura de busca e:

```text
if local voice-conversion cache matches:
    reuse local model and index
else if hub_model_id is configured and reuse_from_hub is enabled:
    check the Hub repository
    download only if it has a SimpleTuner voice-transform manifest
    reuse only if the manifest matches this transform
else if train_if_missing is enabled:
    train the voice-conversion model
    build the retrieval index
    cache locally
    push to hub only when push_to_hub is true
else:
    stop and ask for a model path or a reusable cache
```

O repositorio no Hub deve usar um layout especifico do SimpleTuner, nao apenas arquivos soltos:

```text
config.json
voice_transform/
    manifest.json
    model.safetensors
    features.safetensors
    index.index
```

O manifest e o contrato. Ele deve registrar fingerprint do dataset de identidade alvo, settings de treino RVC, settings do indice, sample rate esperado, versoes das ferramentas e versao do formato voice-transform do SimpleTuner. O SimpleTuner nao deve reutilizar um artifact do Hub sem esse manifest ou com manifest que nao corresponde ao transform atual. Isso evita aplicar silenciosamente o modelo vocal errado a um novo dataset.

Publicacao deve ser opt-in. Uma pseudo config razoavel:

```text
identity_transfer:
    method: rvc
    model:
        train_if_missing: true
        model_name: Target voice RVC
        hub_model_id: org/target-voice-rvc
        reuse_from_hub: true
        push_to_hub: true
        public: false
```

Para identidades privadas, mantenha o repositorio do Hub privado salvo permissao explicita para publicar o modelo vocal. Audio gerado e artifacts de modelo podem ter direitos diferentes, entao trate seus settings de upload separadamente.

## Configuracao no WebUI

O treino do modelo RVC deve ser configuravel pelo WebUI, nao apenas por JSON bruto do dataloader.

O formato esperado no WebUI e um editor de transforms no dataset de audio:

```text
Audio dataset
    Data transforms
        Add transform: Identity transfer
            Method: RVC
            Audio mode: vocal_only / separate_convert_remix / full_mix_convert
            Train RVC model if missing: on
            Force retrain: off
            Build retrieval index: on
            Hub model id: optional
            Reuse from Hub: on when Hub model id is set
            Push RVC model to Hub: off by default
            Hub repo privacy: private by default
            Caption rules: copy, append, remove
```

O WebUI deve deixar os dois setups comuns bem claros:

- **Ja tem vocal stems:** escolha `vocal_only`, deixe Demucs desativado e escreva vocal stems gerados.
- **Tem musicas completas:** escolha `separate_convert_remix`, use separacao com Demucs, converta somente o vocal stem e faça remix com os stems instrumentais originais.

A interface deve mostrar que o audio gerado vira outro split primario de treino de audio. Ela nao deve apresentar identity transfer como `conditioning_data`, porque isso sugeriria comportamento de conditioning pareado durante o treino.

## Comportamento Distribuido no Startup

Quando o SimpleTuner inicia com varios ranks data-parallel, o startup de voice cloning deve usar as GPUs disponiveis em vez de deixar rank 0 fazer todo o trabalho.

Existem duas fases distribuidas separadas:

1. **Treino do modelo RVC:** se `train_if_missing=true`, nao existe cache local correspondente e nao existe artifact correspondente no Hub, o loop de treino RVC deve rodar com DDP quando `world_size > 1`. Cada rank deve receber batches diferentes da voz alvo pelo padrao normal de distributed sampler.
2. **Preparacao do audio gerado:** entradas fonte de expansao devem ser divididas por rank, de forma parecida com TextEmbedCache e VAECache. Cada rank separa, converte e escreve apenas seu shard; depois todos os ranks sincronizam antes da metadata discovery continuar.

Pseudocodigo:

```text
if world_size > 1:
    if RVC model must be trained:
        train RVC with DDP across all ranks
        save final model and index once

    split expansion inputs by global rank
    each rank generates its own audio shard
    barrier
    rank 0 writes or verifies the combined manifest
    barrier
else:
    train and generate serially
```

Apenas um processo deve publicar o modelo vocal final no Hub. O mesmo vale para updates finais do manifest. Outputs gerados por rank podem ser escritos independentemente, desde que os nomes sejam deterministicos e nao sobrepostos.

Isso evita desperdiçar tempo de GPU em sistemas multi-GPU e mantem o startup alinhado ao modelo existente de preparacao de cache do SimpleTuner.

## Logs do Treino RVC

O treino RVC no startup ainda nao deve criar runs do TensorBoard ou WandB. Esses loggers sao configurados para o job principal de treino do SimpleTuner, e reutiliza-los para um job aninhado de voice conversion exigiria nomes de run, paths, regras de resume e politicas de artifact extras.

O estagio RVC ainda pode reportar stats uteis pelo logger nativo de treino do SimpleTuner:

```text
output_dir/
    logs/
        rvc/
            training_stats.jsonl
            summary.json
```

Stats locais uteis incluem loss de treino RVC, pitch loss se ativado, reconstruction ou discriminator loss quando aplicavel, samples processados, tempo decorrido, DDP world size, motivo de cache hit ou miss, e se o modelo final veio de cache local, cache do Hub ou treino sob demanda.

Esses stats sao apenas locais, a menos que uma implementacao futura adicione explicitamente integracao com logger externo para RVC transforms.

## Escolhendo `audio_mode`

### `vocal_only`

Use quando seu dataset de expansao ja esta pre-processado em vocal stems limpos.

```text
source vocal stem -> RVC conversion -> generated vocal stem
```

Gotchas:

- Nao rode Demucs novamente em stems limpos sem motivo.
- Captions devem descrever vocais e letras, nao um arranjo de banda completo, a menos que voce va remixar depois.
- Se o modelo principal espera musicas completas, dados gerados vocal-only podem ensinar uma distribuicao diferente.

### `separate_convert_remix`

Use quando seu dataset de expansao contem musicas completas mixadas.

```text
source full song
    -> Demucs separates vocals and instrumental stems
    -> RVC converts the vocal stem
    -> converted vocal is remixed with the original instrumental stems
    -> generated full song is added to training
```

Este e o modo preferido para expansao de musicas completas, porque evita converter bateria, baixo, guitarras, sala e artifacts de masterizacao como se fossem parte da voz.

Gotchas:

- Separacao de stems pode deixar bleed, artifacts ou problemas de fase.
- Se o vocal stem for fraco, reverberante ou enterrado, a voz convertida pode ficar instavel.
- Loudness do remix importa. Um split gerado sempre mais alto ou mais baixo pode enviesar o treino.
- Captions devem descrever o resultado remixado final, nao apenas a musica fonte.

### `full_mix_convert`

Use somente para testes rapidos.

```text
source full song -> RVC conversion over the whole mix -> generated full song
```

E rapido, mas geralmente tem menor qualidade. Pode arrastar instrumentos pelo conversor de voz e ensinar artifacts indesejados para a LoRA final.

## Captions e Letras

O split gerado deve ter captions que batem com o audio gerado.

Um bom default:

```text
copy source caption
remove source-vocal identity words when configured
append target-vocal identity or style words when configured
copy lyrics sidecar when lyrics still match
```

Para letras, copiar costuma estar correto quando a performance fonte e a convertida usam as mesmas palavras. Nao esta correto quando o transform muda a musica, edita secoes, remove vocais ou usa uma fonte sem letra.

Para captions, copiar cegamente pode estar errado. Se a caption fonte diz "female pop vocal" e a saida convertida tem timbre masculino de rock, a caption precisa mudar. O transform deve suportar regras simples de append/remove, e reescrita avancada pode vir depois.

## Cache e Reuso

O transform deve escrever dois tipos de cache:

```text
voice-conversion cache:
    model checkpoint
    retrieval index
    manifest

generated audio cache:
    generated audio files
    captions
    lyrics, when available
    manifest
```

O manifest deve registrar fingerprint do dataset de identidade, settings do transform, fingerprint dos dados fonte de expansao e versoes das ferramentas. Se esses valores baterem, o startup pode reutilizar artifacts existentes. Se mudarem, o SimpleTuner deve regenerar o estagio afetado.

## Conselhos Praticos

Para a voz alvo em `model.identity_data_dir`, a duracao importa menos do que cobertura vocal limpa.

- **Teste rapido:** 30-60 segundos de audio vocal limpo podem provar que o pipeline roda, mas a voz convertida normalmente ficara rudimentar.
- **Inicio utilizavel:** 5-10 minutos de voz isolada limpa e um primeiro alvo razoavel para um dataset de voz pessoal.
- **Identidade cantada:** 10-30 minutos e melhor quando voce precisa de faixa de pitch, vogais, dinamica, articulacao e fraseado expressivo.

Use muitos clips curtos em vez de um unico arquivo longo. Clips de 5-20 segundos sao mais faceis de revisar, separar e reutilizar. O trainer RVC atual reamostra o audio de identidade para 48 kHz e trunca cada arquivo de identidade para `max_seconds_per_file`, que por default e `180`. Se um usuario fornece um arquivo de 30 minutos, por default apenas os primeiros tres minutos sao usados. Dividir o dataset evita descartar cobertura vocal util por acidente.

O projeto standalone [`huggingface-hub-rvc`](https://github.com/SimpleTuner-io/huggingface-hub-rvc) pode treinar, salvar, carregar e publicar o artifact RVC sem executar um job completo do SimpleTuner. Dentro do SimpleTuner, `scripts/run_rvc_model.py` oferece uma entrada direta para experimentar com a parte de treinamento e conversao RVC do pipeline. Use quando quiser ajustar o dataset de identidade, modo Demucs, retrieval strength, transfer strength ou reutilizacao de artifacts do Hub antes de gastar tempo no treino LoRA principal.

- Mantenha uma voz alvo por LoRA quando controle de identidade importa.
- Prefira exemplos vocais limpos e secos para treinar o modelo de voice conversion.
- Evite duetos, a menos que o objetivo seja aprender o blend do dueto.
- Use musicas de expansao com variedade de tempo, tom, genero, dinamica e fraseado.
- Varie captions para que tokens de identidade nao fiquem grudados em um unico arranjo.
- Ouça amostras geradas antes de treinos longos.

## Falhas Comuns

| Sintoma | Causa provavel |
| --- | --- |
| A LoRA so funciona em um genero | A identidade vocal ainda esta entrelacada com captions de arranjo ou dados fonte. |
| O split gerado soa oco ou com fase estranha | Artifacts de separacao/remix em processamento de musica completa. |
| Instrumentos parecem convertidos como voz | `full_mix_convert` foi usado quando separacao era necessaria. |
| O modelo vocal parece aprender instrumentos | A separacao de identidade gerou vocal stems com vazamento demais de acompanhamento. Defina `model.identity_stem_debug_dir`, inspecione os stems salvos, ou preprocesse stems vocais mais limpos e use `identity_audio_mode=vocal_only`. |
| Identidade vocal fraca | O modelo precisa de dados alvo mais limpos, mais dados ou indice melhor. |
| Captions nao controlam a voz | Captions ainda mencionam a voz fonte ou omitem a identidade alvo. |
| O modelo principal aprende artifacts | Audio gerado tem baixa qualidade ou peso demais no mix de treino. |

## Relacao Com Dados de Regularizacao

Dados gerados por identity transfer nao sao dados de regularizacao por default.

Dados de regularizacao normalmente ensinam a LoRA a preservar o comportamento do modelo base. Dados de identity transfer ensinam a voz alvo em mais contextos. Regularizacao demais com poucos dados diretos de identidade pode enfraquecer os tokens de identidade. Dados gerados demais podem ensinar artifacts de conversao.

Trate como controles separados:

- dataset alvo direto: sinal de identidade mais forte
- dataset gerado por identity transfer: maior cobertura de contexto e estilo
- dataset de regularizacao: preservacao do modelo base

## Status

Esta pagina descreve um workflow experimental `data_transforms`. A implementacao atual treina ou reutiliza um artifact RVC v2 F0 do SimpleTuner, extrai features HuBERT e pitch RMVPE dos clips de identidade, ajusta o gerador/discriminador RVC pretreinado, gera o split expandido, cacheia os resultados e continua para o treino normal sem exigir uma segunda etapa manual de preprocessing.
