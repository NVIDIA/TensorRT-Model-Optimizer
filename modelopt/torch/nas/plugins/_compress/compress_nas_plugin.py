from torch import nn

from modelopt.torch.nas.conversion import NASModeRegistry
from modelopt.torch.opt.config import ModeloptBaseConfig, ModeloptField
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    MetadataDict,
    ModeDescriptor,
    RestoreEntrypoint,
)
from modelopt.torch.opt.searcher import BaseSearcher


class CompressModel(nn.Module):
    pass


class CompressConfig(ModeloptBaseConfig):
    """Configuration for Compress NAS algorithm."""

    hydra_config_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    puzzle_dir: str = ModeloptField(
        default="",
        title="",
        description="",
    )

    dataset_path: str = ModeloptField(
        default="",
        title="",
        description="",
    )


# TOD: Why is it called SuperNetMLP?
class SuperNetMLP(CompressModel):
    """Marker subclass indicating converted/search-space state for CompressConfig.
    TODO: Provide better description
    """

    hydra_config_dir: str
    puzzle_dir: str
    dataset_path: str


def convert_compress_model(model: nn.Module, config: CompressConfig) -> ConvertReturnType:
    """Convert the model to a search space model."""
    print("=" * 80)
    print(f"[convert] before convert:\n{model}")
    model.__class__ = SuperNetMLP
    model.hydra_config_dir = config.hydra_config_dir
    model.puzzle_dir = config.puzzle_dir
    model.dataset_path = config.dataset_path
    print(f"[convert] after convert:\n{model}")
    return model, {}


def restore_compress_model(
    model: nn.Module, config: CompressConfig, metadata: MetadataDict
) -> nn.Module:
    """Reuse convert to produce the same behavior on restore."""
    return convert_compress_model(model, config)[0]


@NASModeRegistry.register_mode
class CompressDescriptor(ModeDescriptor):
    """Descriptor for the Compress mode."""

    @property
    def name(self) -> str:
        """String identifier for this mode."""
        return "compress"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Configuration class for this mode."""
        return CompressConfig

    @property
    def search_algorithm(self) -> type[BaseSearcher]:
        """Return the associated searcher implementation."""
        raise NotImplementedError("Compress mode does not have a search algorithm.")

    @property
    def convert(self) -> ConvertEntrypoint:
        """Entrypoint to convert a model."""
        return convert_compress_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """Entrypoint to restore a model."""
        return restore_compress_model

    @property
    def export_mode(self) -> str | None:
        """The mode that corresponds to the export mode of this mode."""
        return "export"
