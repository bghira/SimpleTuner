# Tutorial da WebUI do SimpleTuner

## Introducao

Este tutorial vai ajudar voce a começar com a interface web do SimpleTuner.

## Instalando requisitos

Para sistemas Ubuntu, comece instalando os pacotes necessarios:

```bash
apt -y install python3.13-venv python3.13-dev
apt -y install libopenmpi-dev openmpi-bin cuda-toolkit-12-8 libaio-dev # if you're using DeepSpeed
apt -y install ffmpeg # if training video models
```

## Criando um diretorio de workspace

Um workspace contem suas configuracoes, modelos de output, imagens de validacao e potencialmente seus datasets.

No Vast ou provedores similares, voce pode usar o diretorio `/workspace/simpletuner`:

```bash
mkdir -p /workspace/simpletuner
export SIMPLETUNER_WORKSPACE=/workspace/simpletuner
cd $SIMPLETUNER_WORKSPACE
```

Se voce quiser criar no seu home:
```bash
mkdir ~/simpletuner-workspace
export SIMPLETUNER_WORKSPACE=~/simpletuner-workspace
cd $SIMPLETUNER_WORKSPACE
```

## Instalando o SimpleTuner no workspace

Crie um virtualenv para instalar dependencias:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
```

### Dependencias especificas CUDA

Usuarios NVIDIA devem usar o extra CUDA para instalar dependencias corretas:

```bash
pip install -e 'simpletuner[cuda]'
# CUDA 13 / Blackwell users (NVIDIA B-series GPUs):
# pip install -e 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
# or, if you've cloned via git:
# pip install -e '.[cuda]'
```

Ha outros extras para usuarios apple e rocm, veja as [instrucoes de instalacao](../INSTALL.md).

## Iniciando o servidor

Para iniciar o servidor com SSL na porta 8080:

```bash
# for DeepSpeed, we'll need CUDA_HOME pointing to the correct location
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

simpletuner server --ssl --port 8080
```

Agora, visite https://localhost:8080 no seu navegador.

Voce pode precisar encaminhar a porta via SSH, por exemplo:

```bash
ssh -L 8080:localhost:8080 user@remote-server
```

> **Dica:** Se voce tiver um ambiente de configuracao existente (ex.: de uso anterior do CLI), pode iniciar o servidor com `--env` para iniciar o treinamento automaticamente quando o servidor ficar pronto:
>
> ```bash
> simpletuner server --ssl --port 8080 --env my-training-config
> ```
>
> Isso e equivalente a iniciar o servidor e clicar manualmente em "Start Training" na WebUI, mas permite inicializacao nao assistida.

## Setup inicial: criando uma conta admin

Na primeira inicializacao, o SimpleTuner exige que voce crie uma conta administradora. Ao visitar a WebUI pela primeira vez, voce vera uma tela de setup solicitando a criacao do primeiro usuario admin.

Informe seu email, username e uma senha segura. Essa conta tera privilegios administrativos completos.

### Gerenciando usuarios

Apos o setup, voce pode gerenciar usuarios na pagina **Manage Users** (acessivel pelo sidebar quando logado como admin):

- **Users tab**: Criar, editar e deletar contas. Atribuir niveis de permissao (viewer, researcher, lead, admin).
- **Levels tab**: Definir niveis de permissao customizados com controle fino.
- **Auth Providers tab**: Configurar autenticacao externa (OIDC, LDAP) para SSO.
- **Registration tab**: Controlar se novos usuarios podem se registrar (desabilitado por padrao).

### API keys para automacao

Usuarios podem gerar API keys para acesso via scripts no perfil ou painel admin. API keys usam o prefixo `st_` e podem ser usadas com o header `X-API-Key`:

```bash
curl -s http://localhost:8080/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

> **Nota:** Para deploys privados/internos, mantenha o registro publico desabilitado e crie contas manualmente pelo painel admin.

## Usando a WebUI

### Passos de onboarding

Assim que a pagina carregar, voce vera perguntas de onboarding para configurar seu ambiente.

#### Diretorio de configuracao

O valor especial de configuracao `configs_dir` aponta para uma pasta que contem todas as configuracoes do SimpleTuner, recomendadas para serem organizadas em subdiretorios - **a Web UI faz isso por voce**:

```
configs/
├── an-environment-named-something
│   ├── config.json
│   ├── lycoris_config.json
│   └── multidatabackend-DataBackend-Name.json
```

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/4be22081-f13d-4aed-a87c-2313ddefc8a4" />

##### Migrando do uso via linha de comando

Se voce ja usava o SimpleTuner sem WebUI, pode apontar para sua pasta config/ existente e todos os ambientes serao auto-descobertos.

Para novos usuarios, o local padrao das configs e datasets sera `~/.simpletuner/` e e recomendado mover seus datasets para algum lugar com mais espaco:

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/c5b3ab53-654e-4a9b-8e2d-7951f11619ef" />


#### Selecao e configuracao (multi-)GPU

Depois de configurar caminhos padrao, voce chegara a uma etapa onde multi-GPU pode ser configurado (na imagem, em um sistema NVIDIA)

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/61d5a7bc-0a02-4a0a-8df0-207cce4b7bc1" />

Se voce tem varias GPUs e quer usar apenas a segunda, e aqui que voce faz isso.

> **Nota para usuarios multi-GPU:** Ao treinar com multiplas GPUs, os requisitos de tamanho de dataset aumentam proporcionalmente. O batch size efetivo e calculado como `train_batch_size × num_gpus × gradient_accumulation_steps`. Se o seu dataset for menor que esse valor, voce precisara aumentar `repeats` no config do dataset ou habilitar `--allow_dataset_oversubscription` nas configuracoes avancadas. Veja a [secao de batch size](#multi-gpu-batch-size-considerations) abaixo para mais detalhes.

#### Criando seu primeiro ambiente de treinamento

Se nao houver configuracoes pre-existentes no seu `configs_dir`, voce sera solicitado a criar **seu primeiro ambiente de treinamento**:

<img width="500" height="640" alt="image" src="https://github.com/user-attachments/assets/2110287a-16fd-4f87-893b-86d2a555a10f" />

Use **Bootstrap From Example** para selecionar um exemplo de config para iniciar, ou apenas informe um nome descritivo e crie um ambiente aleatorio se preferir usar um wizard.

### Alternando entre ambientes de treinamento

Se voce tinha ambientes de configuracao pre-existentes, eles aparecerão nesse menu suspenso.

Caso contrario, a opcao que voce criou durante o onboarding ja estara selecionada e ativa.

<img width="448" height="225" alt="image" src="https://github.com/user-attachments/assets/66fef6a9-2040-47fd-b22d-918470677992" />

Use **Manage Configs** para ir para a aba `Environment`, onde uma lista dos seus ambientes, dataloaders e outras configuracoes podem ser encontradas.

### Wizard de configuracao

Trabalhei bastante para fornecer um wizard de setup abrangente que ajuda a configurar alguns dos ajustes mais importantes em um bootstrap objetivo para comecar.

<img width="470" height="358" alt="image" src="https://github.com/user-attachments/assets/e4bf1a4e-716c-4101-b753-e9e24bb42d8a" />

No menu de navegacao superior esquerdo, o botao Wizard abre um dialogo de selecao:

<img width="448" height="440" alt="image" src="https://github.com/user-attachments/assets/68324fa8-3ca9-45b1-b947-1e7738fd1d8c" />

E entao todas as variantes de modelo embutidas sao oferecidas. Cada variante pre-habilita configuracoes necessarias como Attention Masking ou limites de token estendidos.

#### Opcoes de modelo LoRA

Se voce quiser treinar uma LoRA, podera definir opcoes de quantizacao do modelo aqui.

Em geral, a menos que voce esteja treinando um modelo do tipo Stable Diffusion, int8-quanto e recomendado pois nao prejudica a qualidade e permite batch sizes maiores.

Alguns modelos pequenos como Cosmos2, Sana e PixArt realmente nao gostam de quantizacao.

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/c2e721f2-b4da-4cd0-84fd-7ac81993e87c" />

#### Treinamento full-rank

Treinamento full-rank e desencorajado, pois geralmente leva muito mais tempo e custa mais recursos do que uma LoRA/LyCORIS, para o mesmo dataset.

No entanto, se voce quiser treinar um checkpoint completo, pode configurar estagios DeepSpeed ZeRO aqui, o que sera necessario para modelos maiores como Auraflow, Flux e maiores.

O FSDP2 e suportado, mas nao configuravel neste wizard. Apenas deixe o DeepSpeed desabilitado e configure o FSDP2 manualmente depois se desejar usa-lo

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/88438f1c-b0a2-4249-afd0-7878aa1abada" />

#### Quanto tempo voce quer treinar?

Voce precisara decidir se quer medir tempo de treino em epocas ou steps. No fim e praticamente equivalente, embora algumas pessoas prefiram um ou outro.

<img width="508" height="475" alt="image" src="https://github.com/user-attachments/assets/dcb54279-0ce7-4c66-a9ab-4dd26f87278c" />

#### Compartilhando seu modelo via Hugging Face Hub

Opcionalmente, voce pode publicar seus checkpoints finais *e* intermediarios no [Hugging Face Hub](https://hf.co), mas sera necessario ter uma conta — voce pode fazer login no hub via wizard ou na aba Publishing. De qualquer forma, voce sempre pode mudar de ideia e habilitar ou desabilitar.

Se voce optar por publicar seu modelo, lembre-se de selecionar `Private repo` se nao quiser que o modelo fique acessivel ao publico.

<img width="508" height="370" alt="image" src="https://github.com/user-attachments/assets/8d2d282b-e66f-48a8-a40e-4e4ecc2d280b" />

#### Frequencia de checkpoints

Durante o treinamento, seu modelo sera salvo periodicamente em disco. Manter mais checkpoints exige mais espaco em disco.

Checkpoints permitem retomar o treinamento depois sem repetir todos os passos. Manter alguns checkpoints permite testar varias versoes do seu modelo e ficar com a que funciona melhor para voce.

E recomendado manter um checkpoint a cada 10%, embora isso dependa da quantidade de dados usada no treinamento. Com um dataset pequeno, convem salvar checkpoints com frequencia para garantir que voce nao esteja overfitting.

Datasets extremamente grandes se beneficiam de intervalos de checkpoint maiores para evitar perder tempo gravando-os em disco.

<img width="508" height="485" alt="image" src="https://github.com/user-attachments/assets/c7b1cd0b-a1b9-47ec-87f9-1ecac2e0841a" />

#### Validacoes do modelo

Se voce quiser que o trainer gere imagens periodicamente, voce pode configurar um unico prompt de validacao neste ponto do wizard. Uma biblioteca de prompts multiplos pode ser configurada na aba `Validations & Output` apos o wizard.

Quer terceirizar a validacao para seu proprio script ou servico? Mude o **Validation Method** para `external-script` na aba de validacao apos o wizard e forneca `--validation_external_script`. Voce pode passar contexto de treinamento para o script com placeholders como `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}` e quaisquer valores `validation_*` (ex.: `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`). Habilite `--validation_external_background` para fire-and-forget sem bloquear o treinamento.

Precisa de um hook no momento em que um checkpoint vai para o disco? Use `--post_checkpoint_script` para rodar um script logo apos cada salvamento (antes de iniciar uploads). Ele aceita os mesmos placeholders, com `{remote_checkpoint_path}` vazio.

Se voce quiser manter os provedores de publicacao embutidos do SimpleTuner (ou uploads para o Hugging Face Hub) mas ainda disparar sua automacao com a URL remota, use `--post_upload_script`. Ele roda uma vez por upload com placeholders `{remote_checkpoint_path}`, `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}`. O SimpleTuner nao captura a saida do script — emita quaisquer atualizacoes de tracker diretamente do seu script.

Exemplo de hook:

```bash
--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
```

Onde `notify.sh` envia a URL para sua API de tracker. Sinta-se livre para adaptar para Slack, dashboards customizados ou qualquer outra integracao.

Exemplo funcional: `simpletuner/examples/external-validation/replicate_post_upload.py` demonstra usar `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}` e `{huggingface_path}` para disparar uma inferencia Replicate apos uploads.

Outro exemplo: `simpletuner/examples/external-validation/wavespeed_post_upload.py` chama a API WaveSpeed e faz polling do resultado, usando os mesmos placeholders.

Exemplo focado em Flux: `simpletuner/examples/external-validation/fal_post_upload.py` chama o endpoint fal.ai Flux LoRA; requer `FAL_KEY` e so roda quando `model_family` inclui `flux`.

Exemplo de GPU local: `simpletuner/examples/external-validation/use_second_gpu.py` roda inferencia Flux LoRA em outra GPU (padrao `cuda:1`) e pode ser usado mesmo quando nao ha uploads.

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/e699ba57-526b-4f60-9e8c-0ba410761c9f" />

#### Logging de estatisticas de treinamento

O SimpleTuner suporta varias APIs de destino caso voce queira enviar suas estatisticas de treinamento para uma delas.

Nota: Nenhum dos seus dados pessoais, logs de treinamento, captions ou dados e **jamais** enviado aos desenvolvedores do projeto SimpleTuner. O controle dos seus dados esta em **suas** maos.

<img width="508" height="438" alt="image" src="https://github.com/user-attachments/assets/0f8d15c5-456f-4637-af7e-c2f5f31cb968" />

#### Configuracao de dataset

Neste ponto, voce pode decidir manter algum dataset existente ou criar uma nova configuracao (deixando os outros intactos) por meio do Dataset Creation Wizard, que aparecera ao clicar.

<img width="508" height="290" alt="image" src="https://github.com/user-attachments/assets/b5a7f883-e180-4662-b84c-fff609c6b1df" />

##### Dataset Wizard

Se voce optou por criar um novo dataset, vera o wizard a seguir, que guia a adicao de um dataset local ou em nuvem.

<img width="508" height="332" alt="image" src="https://github.com/user-attachments/assets/c523930b-563e-4b5d-b104-8e7ce4658b2c" />

<img width="508" height="508" alt="image" src="https://github.com/user-attachments/assets/c263f58e-fd85-437e-811a-967b94e309fd" />

Para um dataset local, voce podera usar o botao **Browse directories** para acessar o modal do navegador de datasets.

<img width="396" height="576" alt="image" src="https://github.com/user-attachments/assets/14c51685-3559-4d16-be59-ed4b0959ca32" />

Se voce apontou corretamente o diretorio de datasets durante o onboarding, voce vera seus dados aqui.

Clique no diretorio que deseja adicionar e depois **Select Directory**.

<img width="454" height="356" alt="image" src="https://github.com/user-attachments/assets/1d482655-158a-4e3f-93b7-ef158396813c" />

Depois disso, voce sera guiado por configuracoes de resolucao e recorte.

Ao chegar na etapa de configurar captions, **considere cuidadosamente** qual opcao e correta.

Se voce quer usar apenas uma palavra de gatilho, essa sera a opcao **Instance Prompt**.

<img width="1146" height="896" alt="image" src="https://github.com/user-attachments/assets/6252bf9a-5e68-41c6-8a95-906993f2f546" />

##### Opcional: enviar dataset pelo navegador

Se suas imagens e captions ainda nao estao na maquina, o wizard de dataset agora inclui um botao **Upload** ao lado de **Browse directories**. Voce pode:

- Criar uma nova subpasta sob seu diretorio de datasets configurado e enviar arquivos individuais ou um ZIP (imagens + metadados .txt/.jsonl/.csv sao aceitos).
- Deixar o SimpleTuner extrair o ZIP nessa pasta (dimensionado para backends locais; arquivos muito grandes sao rejeitados).
- Selecionar a pasta recem enviada no navegador e continuar o wizard sem sair da UI.

#### Learning rate, batch size e otimizador

Depois que voce conclui o wizard de dataset (ou se escolheu manter os datasets existentes), voce tera presets para otimizador/learning rate e batch size.

Esses sao pontos de partida que ajudam iniciantes a fazer escolhas um pouco melhores para seus primeiros treinos — para usuarios experientes, use **Manual configuration** para controle completo.

**NOTA**: Se voce planeja usar DeepSpeed depois, a escolha de otimizador nao importa muito aqui.

##### Consideracoes de batch size multi-GPU

Ao treinar com multiplas GPUs, saiba que seu dataset deve acomodar o **batch size efetivo**:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

Se o seu dataset for menor do que esse valor, o SimpleTuner retornara um erro com orientacao especifica. Voce pode:
- Reduzir o batch size
- Aumentar `repeats` na configuracao do dataset
- Habilitar **Allow Dataset Oversubscription** nas configuracoes avancadas para ajustar repeats automaticamente

Veja [DATALOADER.md](../DATALOADER.md#multi-gpu-training-and-dataset-sizing) para mais detalhes sobre dimensionamento de dataset.

<img width="1118" height="1015" alt="image" src="https://github.com/user-attachments/assets/25d5650d-e77b-42fe-b749-06c0ec92b1e2" />

#### Presets de otimizacao de memoria

Para facilitar o setup em hardware de consumidor, cada modelo tem presets customizados que permitem selecionar economia de memoria leve, balanceada ou agressiva.

Na secao **Memory Optimisation** da aba **Training**, voce encontra o botao **Load Presets**:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/804e84f6-7eb8-493e-95d2-a89d930bafa5" />

Que abre esta interface:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/775aaee5-c3c0-4659-bbea-ebb39e3eb098" />


#### Revisar e salvar

Se estiver satisfeito com os valores selecionados, clique em **Finish** no wizard.

Voce vera seu novo ambiente selecionado e pronto para treinamento!

Na maioria dos casos, essas configuracoes serao tudo o que voce precisa. Talvez voce queira adicionar datasets extras ou ajustar outros parametros.

<img width="1096" height="1403" alt="image" src="https://github.com/user-attachments/assets/29fd0bb3-aab2-4455-9612-583ed949ce64" />

Na pagina **Environment**, voce vera o job configurado e botoes para baixar ou duplicar a configuracao, se desejar usa-la como template.

<img width="1881" height="874" alt="image" src="https://github.com/user-attachments/assets/33c0cafa-3fd8-40ee-b6fa-3704b6e698da" />

**NOTA**: O ambiente **Default** e especial e nao e recomendado como ambiente de treinamento geral; suas configuracoes podem ser mescladas automaticamente a qualquer ambiente que habilite a opcao **Use environment defaults**:

<img width="1521" height="991" alt="image" src="https://github.com/user-attachments/assets/9d18b0c1-608e-4ab2-be14-65b98907ec69" />
