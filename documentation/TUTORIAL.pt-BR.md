# Tutorial

Este tutorial foi consolidado no [Guia de Instalação](INSTALL.md).

## Início Rápido

1. **Instale o SimpleTuner**: `pip install 'simpletuner[cuda]'` (veja o README para outras plataformas)
   - CUDA 13 / Blackwell users (NVIDIA B-series GPUs): `pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130`
2. **Configure**: `simpletuner configure` (configuração interativa)
3. **Treine**: `simpletuner train`

## Guias detalhados

- **[Guia de Instalação](INSTALL.md)** - Configuração completa, incluindo preparação dos dados de treinamento
- **[Guias de Início Rápido](QUICKSTART.md)** - Guias de treinamento específicos por modelo
- **[Requisitos de Hardware](https://github.com/bghira/SimpleTuner#hardware-requirements)** - VRAM e requisitos do sistema

Para mais informações, veja:

- **[Guia de Instalação](INSTALL.md)** - Configuração completa, incluindo preparação dos dados de treinamento
- **[Referência de Opções](OPTIONS.md)** - Lista completa de parâmetros
- **[Configuração do Dataloader](DATALOADER.md)** - Configuração de datasets
