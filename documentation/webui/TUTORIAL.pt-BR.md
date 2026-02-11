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

<img width="788" height="465" alt="image" src="https://github.com/user-attachments/assets/656aa287-3b59-476d-ac45-6ede325fe858" />

##### Migrando do uso via linha de comando

Se voce ja usava o SimpleTuner sem WebUI, pode apontar para sua pasta config/ existente e todos os ambientes serao auto-descobertos.

Para novos usuarios, o local padrao das configs e datasets sera `~/.simpletuner/` e e recomendado mover seus datasets para algum lugar com mais espaco:

<img width="775" height="454" alt="image" src="https://github.com/user-attachments/assets/39238810-da26-4bde-8fc9-1002251f778a" />


#### Selecao e configuracao (multi-)GPU

Depois de configurar caminhos padrao, voce chegara a uma etapa onde multi-GPU pode ser configurado (na imagem, em um Macbook)

<img width="755" height="646" alt="image" src="https://github.com/user-attachments/assets/de43a09d-06a7-45c0-8111-7a0b014499c8" />

Se voce tem varias GPUs e quer usar apenas a segunda, e aqui que voce faz isso.

> **Nota para usuarios multi-GPU:** Ao treinar com multiplas GPUs, os requisitos de tamanho de dataset aumentam proporcionalmente. O batch size efetivo e calculado como `train_batch_size × num_gpus × gradient_accumulation_steps`. Se o seu dataset for menor que esse valor, voce precisara aumentar `repeats` no config do dataset ou habilitar `--allow_dataset_oversubscription` nas configuracoes avancadas. Veja a [secao de batch size](#multi-gpu-batch-size-considerations) abaixo para mais detalhes.

#### Criando seu primeiro ambiente de treinamento

Se nao houver configuracoes pre-existentes no seu `configs_dir`, voce sera solicitado a criar **seu primeiro ambiente de treinamento**:

<img width="750" height="1381" alt="image" src="https://github.com/user-attachments/assets/4a3ee88f-c70f-416c-ae5d-6593deb9ca35" />

Use **Bootstrap From Example** para selecionar um exemplo de config para iniciar, ou apenas informe um nome descritivo e crie um ambiente aleatorio se preferir usar um wizard.

### Alternando entre ambientes de treinamento

Se voce tinha ambientes de configuracao pre-existentes, eles aparecerão nesse menu suspenso.

Caso contrario, a opcao que voce criou durante o onboarding ja estara selecionada e ativa.

<img width="965" height="449" alt="image" src="https://github.com/user-attachments/assets/d8c73cef-ecbb-4229-ad54-9ccd55f8175a" />

Use **Manage Configs** para ir para a aba `Environment`, onde uma lista dos seus ambientes, dataloaders e outras configuracoes podem ser encontradas.

### Wizard de configuracao

Trabalhei bastante para fornecer um wizard de setup abrangente que ajuda a configurar alguns dos ajustes mais importantes em um bootstrap objetivo para comecar.

<img width="394" height="286" alt="image" src="https://github.com/user-attachments/assets/21e99854-1d75-4ba9-8be6-15e715d77f4e" />

No menu de navegacao superior esquerdo, o botao Wizard abre um dialogo de selecao:

<img width="1186" height="1756" alt="image" src="https://github.com/user-attachments/assets/f6d4ac57-e3f6-4060-a4d3-b7f0829d7350" />

E entao todas as variantes de modelo embutidas sao oferecidas. Cada variante pre-habilita configuracoes necessarias como Attention Masking ou limites de token estendidos.

#### Opcoes de modelo LoRA

Se voce quiser treinar uma LoRA, podera definir opcoes de quantizacao do modelo aqui.

Em geral, a menos que voce esteja treinando um modelo do tipo Stable Diffusion, int8-quanto e recomendado pois nao prejudica a qualidade e permite batch sizes maiores.

Alguns modelos pequenos como Cosmos2, Sana e PixArt realmente nao gostam de quantizacao.

<img width="1106" height="1464" alt="image" src="https://github.com/user-attachments/assets/0284d987-6060-4692-934a-0905ef2d5ca1" />

#### Treinamento full-rank

Treinamento full-rank e desencorajado, pois geralmente leva muito mais tempo e custa mais recursos do que uma LoRA/LyCORIS, para o mesmo dataset.

No entanto, se voce quiser treinar um checkpoint completo, pode configurar estagios DeepSpeed ZeRO aqui, o que sera necessario para modelos maiores como Auraflow, Flux e maiores.

O FSDP2 e suportado, mas nao configuravel neste wizard. Apenas deixe o DeepSpeed desabilitado e configure o FSDP2 manualmente depois se desejar usa-lo

<img width="1097" height="1278" alt="image" src="https://github.com/user-attachments/assets/60475318-facd-4da1-a2a1-67cecff18e04" />

#### Quanto tempo voce quer treinar?

Voce precisara decidir se quer medir tempo de treino em epocas ou steps. No fim e praticamente equivalente, embora algumas pessoas prefiram um ou outro.

<img width="1136" height="1091" alt="image" src="https://github.com/user-attachments/assets/9146cdcd-f277-45e5-92cb-f74f23039d51" />

#### Compartilhando seu modelo via Hugging Face Hub

Opcionalmente, voce pode publicar seus checkpoints finais *e* intermediarios no [Hugging Face Hub](https://hf.co), mas sera necessario ter uma conta — voce pode fazer login no hub via wizard ou na aba Publishing. De qualquer forma, voce sempre pode mudar de ideia e habilitar ou desabilitar.

Se voce optar por publicar seu modelo, lembre-se de selecionar `Private repo` se nao quiser que o modelo fique acessivel ao publico.

<img width="1090" height="859" alt="image" src="https://github.com/user-attachments/assets/d1f86b6b-b0d5-4caa-b3ff-6bd106928094" />

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

<img width="1101" height="1357" alt="image" src="https://github.com/user-attachments/assets/97bdd3f1-b54c-4087-b4d5-05da8b271751" />

#### Logging de estatisticas de treinamento

O SimpleTuner suporta varias APIs de destino caso voce queira enviar suas estatisticas de treinamento para uma delas.

Nota: Nenhum dos seus dados pessoais, logs de treinamento, captions ou dados e **jamais** enviado aos desenvolvedores do projeto SimpleTuner. O controle dos seus dados esta em **suas** maos.

<img width="1099" height="1067" alt="image" src="https://github.com/user-attachments/assets/c9be9a20-12ad-402a-9605-66ba5771e630" />

#### Configuracao de dataset

Neste ponto, voce pode decidir manter algum dataset existente ou criar uma nova configuracao (deixando os outros intactos) por meio do Dataset Creation Wizard, que aparecera ao clicar.

<img width="1103" height="877" alt="image" src="https://github.com/user-attachments/assets/3d3cc391-52ed-422e-a4a1-676ca342df10" />

##### Dataset Wizard

Se voce optou por criar um novo dataset, vera o wizard a seguir, que guia a adicao de um dataset local ou em nuvem.

<img width="1110" height="857" alt="image" src="https://github.com/user-attachments/assets/3719e0f5-774e-461d-be02-902e08a679f6" />

<img width="1082" height="1255" alt="image" src="https://github.com/user-attachments/assets/ac38a3de-364a-447f-a734-cab2bdd5338d" />

Para um dataset local, voce podera usar o botao **Browse directories** para acessar o modal do navegador de datasets.

<img width="1201" height="1160" alt="image" src="https://github.com/user-attachments/assets/66a333d0-30fa-45d1-a5b2-1e859d789677" />

Se voce apontou corretamente o diretorio de datasets durante o onboarding, voce vera seus dados aqui.

Clique no diretorio que deseja adicionar e depois **Select Directory**.

<img width="907" height="709" alt="image" src="https://github.com/user-attachments/assets/1d482655-158a-4e3f-93b7-ef158396813c" />

Depois disso, voce sera guiado por configuracoes de resolucao e recorte.

**NOTA**: O SimpleTuner nao faz *upscale* de imagens, entao garanta que elas sejam pelo menos do tamanho da resolucao configurada.

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
