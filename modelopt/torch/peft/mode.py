from modelopt.torch.opt.config import ModeloptBaseConfig
from modelopt.torch.opt.mode import (
    ConvertEntrypoint,
    ConvertReturnType,
    ModeConfigList,
    ModeDescriptor,
    RestoreEntrypoint,
    UpdateEntrypoint,
    _ModeRegistryCls,
)
from .config import PEFTConfig, ExportPEFTConfig
from .conversion import convert_to_peft_model, restore_peft_model, update_peft_metadata, export_peft_model, restore_export_peft_model

PEFTModeRegistry = _ModeRegistryCls("PEFT")

@PEFTModeRegistry.register_mode
class PEFTModeDescriptor(ModeDescriptor):
    @property
    def name(self) -> str:
        return "peft"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        return PEFTConfig

    @property
    def export_mode(self) -> str | None:
        return "export_peft"

    @property
    def convert(self) -> ConvertEntrypoint:
        return convert_to_peft_model

    @property
    def restore(self) -> RestoreEntrypoint:
        return restore_peft_model

    @property
    def update_for_save(self) -> UpdateEntrypoint:
        return update_peft_metadata

    @property
    def update_for_new_mode(self) -> UpdateEntrypoint:
        """The mode's entrypoint for updating the models state before new mode."""
        return update_peft_metadata

@PEFTModeRegistry.register_mode
class ExportPEFTModeDescriptor(ModeDescriptor):

    @property
    def name(self) -> str:
        """Returns the value (str representation) of the mode."""
        return "export_peft"

    @property
    def config_class(self) -> type[ModeloptBaseConfig]:
        """Specifies the config class for the mode."""
        return ExportPEFTConfig

    @property
    def is_export_mode(self) -> bool:
        """Specifies whether the mode is an export mode."""
        return True

    @property
    def convert(self) -> ConvertEntrypoint:
        """The mode's entrypoint for converting a model."""
        return export_peft_model

    @property
    def restore(self) -> RestoreEntrypoint:
        """The mode's entrypoint for restoring a model."""
        return restore_export_peft_model