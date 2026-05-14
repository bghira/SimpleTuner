# NSFW Classifier Checks

SimpleTuner includes optional classifier checks that can reject samples during VAE cache preprocessing. This feature is a local filtering tool. It is not legal advice, a compliance system, or a guarantee that a dataset is lawful or acceptable for a particular use.

## Your Responsibility

You are responsible for deciding whether your dataset, training run, model output, and publication or distribution plans comply with the rules that apply to you.

Those rules can include local, regional, national, and platform-specific requirements. They may depend on consent, age, likeness rights, privacy, publicity rights, obscenity rules, employment or institutional policy, and whether the result depicts or impersonates a real person. Laws also change over time and differ by jurisdiction.

SimpleTuner will not decide this for you. It will not warn you that your policy is incomplete, check whether your thresholds match your law, or confirm that a model output is safe to publish. If you are unsure, get qualified legal advice for your jurisdiction and use case.

## Privacy

NSFW classifier checks run locally on the machine running SimpleTuner.

- Dataset samples are not sent to a third-party moderation API by this feature.
- Classifier results are not forwarded to third parties.
- The `--report_to` training telemetry option does not receive NSFW classifier results.
- Reports are stored locally on the instance in the VAE cache directory as `nsfw_classifier_report_rank*.json`.

The one network-facing behavior to expect is normal Hugging Face model loading if the classifier weights are not already present in your local model cache. After the model is available locally, classification itself runs on-instance.

## Opt-In Behavior

The feature is disabled by default. Enable it with:

```bash
--enable_nsfw_check=true
```

Checks only apply to uncached samples that VAE caching is about to process. Existing VAE caches are trusted, and `skip_file_discovery=vae` bypasses enforcement because SimpleTuner assumes you have already prepared the cache under your own policy.

Evaluation datasets are not scanned.

## Supported Classifiers

SimpleTuner supports standard Hugging Face Transformers image-classification models through `AutoImageProcessor` and `AutoModelForImageClassification`.

The default models are:

```text
Falconsai/nsfw_image_detection:threshold=0.5,AdamCodd/vit-base-nsfw-detector:threshold=0.5
```

You can provide your own CSV list:

```bash
--nsfw_check_models="org/model-a:threshold=0.5,org/model-b:threshold=0.7"
```

SimpleTuner does not enable `trust_remote_code` for these classifiers and does not add `timm` as a dependency for this feature. Models that require custom code or non-Transformers backends are not supported by this scanner.

## Non-NSFW Use

Despite the option names, this mechanism is not limited to sexual-content filtering. It can be used for other binary or label-score checks if the classifier emits recognizable labels and scores that map cleanly to the unsafe/safe label hints SimpleTuner expects.

Examples might include rejecting samples with a prohibited visual category, brand-sensitive content, or other locally defined dataset policy. You are still responsible for validating that the classifier labels, thresholds, and vote settings match your policy.

## Legal Context

Adult sexual content is not automatically illegal everywhere, and NSFW model training is not automatically disallowed by SimpleTuner. That does not mean a given dataset, output, or deployment is lawful.

High-risk areas include:

- Content involving minors or apparent minors. In the United States, the FBI Internet Crime Complaint Center states that child sexual abuse material created by generative AI and similar tools is illegal.
- Non-consensual intimate imagery, sexual exploitation, harassment, blackmail, or distribution without permission.
- Outputs that impersonate, recreate, or misleadingly depict a real person, especially for sexual, fraudulent, or reputationally harmful purposes. The FTC has highlighted AI-enabled impersonation and deepfake fraud risks.
- Deepfake disclosure and transparency rules. For example, EU AI Act Article 50 includes transparency obligations for certain AI-generated or manipulated image, audio, or video content constituting deepfakes.
- Contractual or platform rules, including dataset licenses, hosting provider policies, workplace rules, payment processor rules, and model distribution terms.

Treat the classifier as one control in your own review process, not as the review process itself.

## Related Options

- `--enable_nsfw_check`
- `--nsfw_check_models`
- `--nsfw_check_min_votes`
- `--nsfw_check_backend_types`
- `--nsfw_check_sample_types`
- `--delete_nsfw_images`
- `--nsfw_check_video_frame_count`
- `--nsfw_check_video_frame_selection`
- `--nsfw_check_video_min_flagged_frames`

See [DATALOADER.md#nsfw-classifier-checks-during-vae-caching](DATALOADER.md#nsfw-classifier-checks-during-vae-caching) for VAE cache integration details.

## References

- [FBI IC3: Child Sexual Abuse Material Created by Generative AI and Similar Online Tools is Illegal](https://www.ic3.gov/PSA/2024/PSA240329)
- [FTC: Proposed protections to combat AI impersonation of individuals](https://www.ftc.gov/news-events/news/press-releases/2024/02/ftc-proposes-new-protections-combat-ai-impersonation-individuals)
- [EU AI Act Article 50: transparency obligations](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50)
