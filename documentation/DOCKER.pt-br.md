# Docker para SimpleTuner

Esta configuração Docker fornece um ambiente abrangente para executar o SimpleTuner em várias plataformas, incluindo Runpod, Vast.ai e outros hosts compatíveis com Docker. Ela é otimizada para facilidade de uso e robustez, integrando ferramentas e bibliotecas essenciais para projetos de machine learning.

## Recursos do contêiner

- **Imagem base com CUDA habilitado**: Construída a partir de `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` para suportar aplicações aceleradas por GPU.
- **Ferramentas de desenvolvimento**: Inclui Git, SSH e vários utilitários como `tmux`, `vim`, `htop`.
- **Python e bibliotecas**: Vem com Python 3.10 e SimpleTuner pré-instalado via pip.
- **Integração com Hugging Face e WandB**: Pré-configurada para integração fluida com Hugging Face Hub e WandB, facilitando compartilhamento de modelos e rastreamento de experimentos.

## Primeiros passos

### Suporte a Windows via WSL (Experimental)

O guia a seguir foi testado em uma distribuição WSL2 com Dockerengine instalado.


### 1. Construindo o contêiner

Clone o repositório e navegue até o diretório que contém o Dockerfile. Construa a imagem Docker usando:

```bash
docker build -t simpletuner .
```

### 2. Executando o contêiner

Para executar o contêiner com suporte a GPU, execute:

```bash
docker run --gpus all -it -p 22:22 simpletuner
```

Este comando configura o contêiner com acesso à GPU e mapeia a porta SSH para conectividade externa.

### 3. Variáveis de ambiente

Para facilitar a integração com ferramentas externas, o contêiner suporta variáveis de ambiente para os tokens do Hugging Face e WandB. Passe-as em tempo de execução da seguinte forma:

```bash
docker run --gpus all -e HF_TOKEN='your_token' -e WANDB_API_KEY='your_token' -it -p 22:22 simpletuner
```

### 4. Volumes de dados

Para armazenamento persistente e compartilhamento de dados entre o host e o contêiner, monte um volume de dados:

```bash
docker run --gpus all -v /path/on/host:/workspace -it -p 22:22 simpletuner
```

### 5. Acesso SSH

O SSH no contêiner é configurado por padrão. Certifique-se de fornecer sua chave pública SSH através da variável de ambiente apropriada (`SSH_PUBLIC_KEY` para Vast.ai ou `PUBLIC_KEY` para Runpod).

### 6. Usando SimpleTuner

SimpleTuner já vem pré-instalado e pronto para uso. Você pode executar comandos de treinamento diretamente:

```bash
simpletuner configure
simpletuner train
```

Para configuração e setup, consulte a [documentação de instalação](INSTALL.md) e os [guias de início rápido](QUICKSTART.md).

## Configuração adicional

### Scripts e configurações personalizadas

Se você quiser adicionar scripts de inicialização personalizados ou modificar configurações, estenda o script de entrada (`docker-start.sh`) para atender às suas necessidades específicas.

Se alguma capacidade não puder ser obtida por meio desta configuração, por favor abra uma nova issue.

### Docker Compose

Para usuários que preferem `docker-compose.yaml`, este template é fornecido para você estender e personalizar conforme suas necessidades.

Depois que a stack for implantada, você pode se conectar ao contêiner e começar a operar nele conforme mencionado nos passos acima.

```bash
docker compose up -d

docker exec -it simpletuner /bin/bash
```

```docker-compose.yaml
services:
  simpletuner:
    container_name: simpletuner
    build:
      context: [Path to the repository]/SimpleTuner
      dockerfile: Dockerfile
    ports:
      - "[port to connect to the container]:22"
    volumes:
      - "[path to your datasets]:/datasets"
      - "[path to your configs]:/workspace/config"
    environment:
      HF_TOKEN: [your hugging face token]
      WANDB_API_KEY: [your wanddb token]
    command: ["tail", "-f", "/dev/null"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

> ⚠️ Tenha cuidado com o manuseio de seus tokens do WandB e Hugging Face! É recomendável não commitá-los nem mesmo em um repositório privado para garantir que não sejam vazados. Para casos de uso em produção, recomenda-se armazenamento de gerenciamento de chaves, mas isso está fora do escopo deste guia.
---

## Solução de problemas

### Incompatibilidade de versão do CUDA

**Sintoma**: A aplicação falha ao utilizar a GPU, ou aparecem erros relacionados a bibliotecas CUDA ao tentar executar tarefas aceleradas por GPU.

**Causa**: Este problema pode ocorrer se a versão do CUDA instalada dentro do contêiner Docker não corresponder à versão do driver CUDA disponível na máquina host.

**Solução**:
1. **Verifique a versão do driver CUDA no host**: Determine a versão do driver CUDA instalada na máquina host executando:
   ```bash
   nvidia-smi
   ```
   Este comando exibirá a versão do CUDA no canto superior direito da saída.

2. **Combine a versão do CUDA no contêiner**: Garanta que a versão do toolkit CUDA na sua imagem Docker seja compatível com o driver CUDA do host. A NVIDIA geralmente permite compatibilidade futura, mas verifique a matriz de compatibilidade específica no site da NVIDIA.

3. **Recrie a imagem**: Se necessário, modifique a imagem base no Dockerfile para corresponder ao driver CUDA do host. Por exemplo, se seu host usa CUDA 11.2 e o contêiner está configurado para CUDA 11.8, pode ser necessário mudar para uma imagem base apropriada:
   ```Dockerfile
   FROM nvidia/cuda:11.2.0-runtime-ubuntu22.04
   ```
   Depois de modificar o Dockerfile, reconstrua a imagem Docker.

### Problemas de conexão SSH

**Sintoma**: Não é possível conectar ao contêiner via SSH.

**Causa**: Configuração incorreta das chaves SSH ou o serviço SSH não iniciando corretamente.

**Solução**:
1. **Verifique a configuração do SSH**: Certifique-se de que a chave pública SSH foi adicionada corretamente em `~/.ssh/authorized_keys` no contêiner. Além disso, verifique se o serviço SSH está em execução entrando no contêiner e executando:
   ```bash
   service ssh status
   ```
2. **Portas expostas**: Confirme que a porta SSH (22) está devidamente exposta e mapeada ao iniciar o contêiner, conforme mostrado nas instruções de execução:
   ```bash
   docker run --gpus all -it -p 22:22 simpletuner
   ```

### Conselhos gerais

- **Logs e saída**: Revise os logs e a saída do contêiner em busca de mensagens de erro ou avisos que possam fornecer mais contexto sobre o problema.
- **Documentação e fóruns**: Consulte a documentação do Docker e do NVIDIA CUDA para mais dicas de solução de problemas. Fóruns da comunidade e issue trackers relacionados ao software ou às dependências específicas também podem ser recursos valiosos.
