"""Validate dataset relationships before constructing providers."""

from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type


class DatasetRelationshipError(ValueError):
    def __init__(self, dataset_id, field, message):
        super().__init__(f"(id={dataset_id}) {message}")
        self.dataset_id = dataset_id
        self.field = field


def resolve_dataset_relationships(configs, *, authored=False):
    """Return conditioning and audio links between enabled datasets."""
    enabled = [
        config
        for config in configs
        if isinstance(config.get("id"), str) and config["id"] and not (config.get("disabled") or config.get("disable"))
    ]
    roles = {}
    for config in enabled:
        if config.get("conditioning_type") is not None:
            roles[config["id"]] = DatasetType.CONDITIONING
        else:
            try:
                roles[config["id"]] = ensure_dataset_type(config.get("dataset_type"), default=DatasetType.IMAGE)
            except ValueError:
                if not authored:
                    raise

    conditioning, audio = {}, {}
    source_roles = {DatasetType.IMAGE, DatasetType.VIDEO, DatasetType.AUDIO, DatasetType.EVAL, DatasetType.CONDITIONING}
    for config in enabled:
        dataset_id = config["id"]
        role = roles.get(dataset_id)
        for field, target_role, eligible_source_roles, links in (
            (
                "conditioning_data",
                DatasetType.CONDITIONING,
                (DatasetType.IMAGE, DatasetType.VIDEO, DatasetType.AUDIO, DatasetType.EVAL),
                conditioning,
            ),
            ("s2v_datasets", DatasetType.AUDIO, (DatasetType.VIDEO, DatasetType.EVAL), audio),
        ):
            targets = config.get(field)
            if role not in eligible_source_roles or not targets:
                continue
            if authored and field == "conditioning_data" and config.get("conditioning"):
                continue
            if isinstance(targets, str):
                targets = [targets]
            if not isinstance(targets, (list, tuple)):
                raise DatasetRelationshipError(dataset_id, field, f"{field} must be a dataset ID or a list of IDs.")
            for target in targets:
                if not isinstance(target, str) or roles.get(target) is not target_role:
                    raise DatasetRelationshipError(
                        dataset_id,
                        field,
                        f"{field} references {target!r}, which is not an enabled {target_role.value} dataset.",
                    )
            links[dataset_id] = list(targets)

        audio_config = config.get("audio")
        if role is DatasetType.AUDIO and isinstance(audio_config, dict) and audio_config.get("source_from_video"):
            source = config.get("source_dataset_id") or audio_config.get("source_dataset_id")
            if source and (not isinstance(source, str) or roles.get(source) not in {DatasetType.VIDEO, DatasetType.EVAL}):
                field = "source_dataset_id" if config.get("source_dataset_id") else "audio.source_dataset_id"
                raise DatasetRelationshipError(dataset_id, field, f"Video source {source!r} is unavailable or disabled.")

        for field in ("source_dataset_id", "metadata_clone_source_id"):
            source = config.get(field)
            if source is None:
                continue
            if not isinstance(source, str) or source == dataset_id or roles.get(source) not in source_roles:
                raise DatasetRelationshipError(
                    dataset_id, field, f"{field} references {source!r}, which is not a separate enabled source dataset."
                )

    return conditioning, audio
