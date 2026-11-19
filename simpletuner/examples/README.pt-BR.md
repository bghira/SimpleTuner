# Exemplos de treinamento

Os exemplos seguem `model_family/model_flavour/vram/recipe`. Escolha a família, a variante e uma faixa de memória como `24g`, `48g` ou `80g`. A faixa de VRAM `default` não define uma capacidade específica; a variante `default` usa o padrão configurado pelo modelo. Uma faixa como `8x80g` exige oito GPUs com 80 GB cada.

Pastas nomeadas separam receitas LoRA, LoKr, ControlNet, AnyFlow e variantes de condicionamento. Planos de dados, bibliotecas de prompts e scripts de validação compartilhados ficam em `_shared/`.

## Listar e executar exemplos

Após instalar o SimpleTuner, liste as receitas e informe o caminho relativo completo em `example=`:

```bash
simpletuner examples list
simpletuner train example=flux3_image/klein/24g/peft-lora
```

## Copiar e personalizar

O comando mantém as pastas do catálogo no destino e copia os arquivos JSON referenciados para a pasta da receita. Edite a configuração e o plano de dados antes de treinar nesse ambiente:

```bash
simpletuner examples copy flux3_image/klein/24g/peft-lora --dest config
# config/flux3_image/klein/24g/peft-lora/config.json
cd config/flux3_image/klein/24g/peft-lora
simpletuner train
```

No criador de ambientes e no assistente de configuração da WebUI, selecione primeiro a família. O seletor mostra somente receitas dessa família. Alterar a família limpa a seleção anterior.

Estas receitas são pontos de partida. Consulte o guia do modelo e ajuste o lote, a resolução e a duração ao seu conjunto de dados e hardware.

## Treinamento progressivo com Parquet

Copie a receita do PixArt Sigma e ajuste o bucket de captura, o manifesto e os diretórios de cache do dataloader. Passe as credenciais pelas variáveis de ambiente da AWS e configure `AWS_ENDPOINT_URL` para um endpoint compatível com S3. Para R2, defina `aws_region_name` como `auto` nas entradas do Webshart.

```bash
simpletuner examples copy pixart/600M-512/default/progressive-webshart-full --dest config
```

O exemplo ajusta o modelo 600M-512 em etapas sem recorte de 128, 256 e 512 pixels. O texto é codificado no lote atual e a amostragem segue os shards. Para embaralhar globalmente, use `webshart.sampling_policy: global`. Defina os limites das etapas antes de iniciar e mantenha os mesmos limites ao retomar de um checkpoint.
