# Verificações de Classificador NSFW

O SimpleTuner inclui verificações opcionais de classificador que podem rejeitar samples durante o pré-processamento do cache VAE. Este recurso é uma ferramenta local de filtragem. Ele não é aconselhamento jurídico, sistema de compliance, nem garantia de que um dataset é legal ou aceitável para um uso específico.

## Sua responsabilidade

Você é responsável por decidir se seu dataset, execução de treinamento, saída do modelo e planos de publicação ou distribuição cumprem as regras aplicáveis a você.

Essas regras podem incluir requisitos locais, regionais, nacionais e específicos de plataformas. Elas podem depender de consentimento, idade, direitos de imagem, privacidade, direito de publicidade, regras de obscenidade, política institucional ou de trabalho, e se o resultado retrata ou impersona uma pessoa real. Leis também mudam com o tempo e variam por jurisdição.

O SimpleTuner não decide isso por você. Ele não vai avisar que sua política está incompleta, verificar se seus thresholds correspondem à lei, ou confirmar que uma saída do modelo é segura para publicar. Se você tiver dúvida, obtenha aconselhamento jurídico qualificado para sua jurisdição e caso de uso.

## Privacidade

As verificações NSFW rodam localmente na máquina que executa o SimpleTuner.

- Samples do dataset não são enviados por este recurso para uma API de moderação de terceiros.
- Resultados do classificador não são encaminhados a terceiros.
- A opção de telemetria de treinamento `--report_to` não recebe resultados do classificador NSFW.
- Relatórios são armazenados localmente na instância, no diretório de cache VAE, como `nsfw_classifier_report_rank*.json`.

O comportamento com rede a esperar é o carregamento normal de modelo pelo Hugging Face se os pesos do classificador ainda não estiverem no cache local de modelos. Depois que o modelo está disponível localmente, a classificação em si roda na instância.

## Comportamento opt-in

O recurso vem desativado por padrão. Habilite com:

```bash
--enable_nsfw_check=true
```

As verificações só se aplicam a samples sem cache que o cache VAE está prestes a processar. Caches VAE existentes são confiados, e `skip_file_discovery=vae` ignora o enforcement porque o SimpleTuner assume que você já preparou o cache segundo sua própria política.

Datasets de avaliação não são verificados.

## Classificadores suportados

O SimpleTuner suporta modelos padrão de classificação de imagem do Hugging Face Transformers por meio de `AutoImageProcessor` e `AutoModelForImageClassification`.

Os modelos padrão são:

```text
Falconsai/nsfw_image_detection:threshold=0.5,AdamCodd/vit-base-nsfw-detector:threshold=0.5
```

Você pode fornecer sua própria lista CSV:

```bash
--nsfw_check_models="org/model-a:threshold=0.5,org/model-b:threshold=0.7"
```

O SimpleTuner não habilita `trust_remote_code` para esses classificadores e não adiciona `timm` como dependência para este recurso. Modelos que exigem código customizado ou backends que não sejam Transformers não são suportados por este scanner.

## Uso não NSFW

Apesar dos nomes das opções, este mecanismo não se limita a filtragem de conteúdo sexual. Ele pode ser usado para outras verificações binárias ou por score de labels se o classificador emitir labels e scores reconhecíveis que mapeiam claramente para os hints unsafe/safe esperados pelo SimpleTuner.

Exemplos podem incluir rejeitar samples com uma categoria visual proibida, conteúdo sensível para marca, ou outra política local de dataset. Você ainda é responsável por validar que labels, thresholds e configurações de votação do classificador correspondem à sua política.

## Contexto legal

Conteúdo sexual adulto não é automaticamente ilegal em todos os lugares, e treinamento de modelos NSFW não é automaticamente proibido pelo SimpleTuner. Isso não significa que um dataset, saída ou deploy específico seja legal.

Áreas de alto risco incluem:

- Conteúdo envolvendo menores ou pessoas que aparentam ser menores. Nos Estados Unidos, o FBI Internet Crime Complaint Center afirma que material de abuso sexual infantil criado por IA generativa e ferramentas similares é ilegal.
- Imagens íntimas sem consentimento, exploração sexual, assédio, chantagem ou distribuição sem permissão.
- Saídas que impersonam, recriam ou retratam de forma enganosa uma pessoa real, especialmente para fins sexuais, fraudulentos ou prejudiciais à reputação. A FTC destacou riscos de impersonação por IA e fraude com deepfakes.
- Regras de transparência e divulgação de deepfake. Por exemplo, o Artigo 50 do EU AI Act inclui obrigações de transparência para certos conteúdos de imagem, áudio ou vídeo gerados ou manipulados por IA que constituam deepfakes.
- Regras contratuais ou de plataforma, incluindo licenças de datasets, políticas de provedores de hospedagem, regras de trabalho, regras de processadores de pagamento e termos de distribuição de modelos.

Trate o classificador como um controle no seu próprio processo de revisão, não como o processo de revisão inteiro.

## Opções relacionadas

- `--enable_nsfw_check`
- `--nsfw_check_models`
- `--nsfw_check_min_votes`
- `--nsfw_check_backend_types`
- `--nsfw_check_sample_types`
- `--delete_nsfw_images`
- `--nsfw_check_video_frame_count`
- `--nsfw_check_video_frame_selection`
- `--nsfw_check_video_min_flagged_frames`

Veja [DATALOADER.pt-BR.md#nsfw-classifier-checks-during-vae-caching](DATALOADER.pt-BR.md#nsfw-classifier-checks-during-vae-caching) para detalhes da integração com o cache VAE.

## Referências

- [FBI IC3: Child Sexual Abuse Material Created by Generative AI and Similar Online Tools is Illegal](https://www.ic3.gov/PSA/2024/PSA240329)
- [FTC: Proposed protections to combat AI impersonation of individuals](https://www.ftc.gov/news-events/news/press-releases/2024/02/ftc-proposes-new-protections-combat-ai-impersonation-individuals)
- [EU AI Act Article 50: transparency obligations](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50)
