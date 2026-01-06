# Detalhes de implementacao da WebUI do SimpleTuner

## Visao de design

A API do SimpleTuner foi construida com flexibilidade em mente.

No modo `trainer`, uma unica porta e aberta para integracao do FastAPI em outros servicos.
Com o modo `unified`, uma porta adicional e aberta para a WebUI receber eventos de callback de um processo remoto `trainer`.

## Framework web

A WebUI e construida e servida usando FastAPI;

- Alpine.js e usado para componentes reativos
- HTMX e usado para carregamento dinamico de conteudo e interatividade
- FastAPI com Starlette e SSE-Starlette sao usados para servir uma API centrada em dados com server-sent events (SSE) para atualizacoes em tempo real
- Jinja2 e usado para templating HTML

Alpine foi escolhido por sua simplicidade e facilidade de integracao - ele tira o NodeJS da stack, tornando mais facil fazer deploy e manter.

HTMX tem sintaxe simples e leve que combina bem com Alpine. Ele oferece capacidades extensas para carregamento dinamico, tratamento de formularios e interatividade sem precisar de um framework frontend completo.

Escolhi Starlette e SSE-Starlette porque queria manter a duplicacao de codigo no minimo; muito refactoring foi necessario para sair de um espalhamento ad-hoc de codigo procedural para uma abordagem mais declarativa.

### Fluxo de dados

Historicamente o app FastAPI servia como um “service worker” dentro de clusters de jobs: o trainer iniciava, expunha uma superficie de callback limitada, e orquestradores remotos faziam streaming de updates de status via HTTP. A WebUI reutiliza esse mesmo barramento de callbacks. No modo unified rodamos tanto o trainer quanto a interface no mesmo processo, enquanto deploys apenas de trainer ainda podem enviar eventos para `/callbacks` e deixar uma instancia separada da WebUI consumi-los via SSE. Nenhuma fila ou broker novo necessario — aproveitamos a infraestrutura que ja acompanha os deploys headless.

## Arquitetura de backend

A UI do trainer roda em cima do SDK central que agora expõe servicos bem definidos em vez de helpers procedurais soltos. O FastAPI ainda termina cada requisicao, mas a maioria das rotas e uma delegacao fina para objetos de servico. Isso deixa a camada HTTP simples e maximiza a reutilizacao para o CLI, wizard de configuracao e APIs futuras.

### Handlers de rota

`simpletuner/simpletuner_sdk/server/routes/web.py` conecta a superficie `/web/trainer`. Ha apenas dois endpoints interessantes:

- `trainer_page` – renderiza o chrome externo (navegacao, seletor de config, lista de abas). Pede metadados ao `TabService` e envia tudo para o template `trainer_htmx.html`.
- `render_tab` – um destino HTMX generico. Cada botao de aba chama esse endpoint com o nome da aba; a rota resolve o layout correspondente via `TabService.render_tab` e retorna o trecho HTML.

O restante do roteador HTTP fica em `simpletuner/simpletuner_sdk/server/routes/` e segue o mesmo padrao: a logica de negocio vive em um modulo de servico, a rota extrai parametros, chama o servico e transforma o resultado em JSON ou HTML.

### TabService

`TabService` e o orquestrador central do formulario de treinamento. Ele define:

- metadados estaticos para cada aba (titulo, icone, template, hook de contexto opcional)
- `render_tab()` que
  1. pega a configuracao da aba (template, descricao)
  2. pede ao `FieldService` o bundle de campos pertencentes a aba/secao
  3. injeta contexto especifico da aba (lista de datasets, inventario de GPU, estado de onboarding)
  4. retorna um render Jinja de `form_tab.html`, `datasets_tab.html`, etc.

Ao empurrar essa logica para uma classe, podemos reutilizar a mesma renderizacao para HTMX, o wizard CLI e testes. Nada no template acessa estado global — tudo e fornecido explicitamente via contexto.

### FieldService e FieldRegistry

`FieldService` converte entradas do registry em dicionarios prontos para o template. Responsabilidades:

- filtrar campos por contexto de plataforma/modelo (ex.: ocultar controles CUDA-only em maquinas MPS)
- avaliar regras de dependencia (`FieldDependency`) para a UI desabilitar ou ocultar controles (por exemplo, extras do Dynamo permanecem cinza ate um backend ser selecionado)
- enriquecer campos com hints, escolhas dinamicas, formatacao de display e classes de coluna

Ele delega o catalogo bruto de campos ao `FieldRegistry`, uma lista declarativa em `simpletuner/simpletuner_sdk/server/services/field_registry`. Cada `ConfigField` descreve nomes de flags CLI, regras de validacao, ordenacao de importancia, metadados de dependencia e copy padrao da UI. Esse arranjo permite que outras camadas (parser CLI, API, gerador de docs) compartilhem a mesma fonte de verdade enquanto apresentam no seu proprio formato.

### Persistencia de estado e onboarding

A WebUI armazena preferencias leves via `WebUIStateStore`. Ela le defaults de `$SIMPLETUNER_WEB_UI_CONFIG` (ou um caminho XDG) e expoe:

- tema, raiz de datasets, defaults de output dir
- estado do checklist de onboarding por feature
- overrides de Accelerate em cache (apenas chaves whitelist, como `--num_processes`, `--dynamo_backend`)

Esses valores sao injetados na pagina durante o render inicial de `/web/trainer` para que os stores Alpine possam iniciar sem round-trips extras.

### Interacao HTMX + Alpine

Cada painel de configuracao e apenas um bloco de HTML com `x-data` para comportamento Alpine. Botoes de aba acionam GETs HTMX em `/web/trainer/tabs/{tab}`; o servidor responde com o formulario renderizado e o Alpine mantem o estado do componente. Um helper pequeno (`trainer-form.js`) reaplica alteracoes salvas para que o usuario nao perca edicoes em andamento ao trocar de abas.

Atualizacoes do servidor (status de treinamento, telemetria de GPU) fluem pelos endpoints SSE (`sse_manager.js`) e alimentam stores Alpine que dirigem toasts, barras de progresso e banners do sistema.

### Guia rapido de layout de arquivos

- `templates/` – templates Jinja; `partials/form_field.html` renderiza controles individuais. `partials/form_field_htmx.html` e a variante amigavel ao HTMX usada quando um wizard precisa de two-way binding.
- `static/js/modules/` – scripts de componentes Alpine (formulario do trainer, inventario de hardware, navegador de datasets).
- `static/js/services/` – helpers compartilhados (avaliacao de dependencia, SSE manager, event bus).
- `simpletuner/simpletuner_sdk/server/services/` – camada de servicos backend (fields, tabs, configs, datasets, maintenance, events).

Juntos, isso mantem a WebUI stateless do lado do servidor, com partes stateful (dados de formulario, toasts) vivendo no navegador. O backend se limita a transformacoes de dados puras, o que facilita testes e evita problemas de threading quando o trainer e o servidor web rodam no mesmo processo.
