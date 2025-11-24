import os
import torch
import torch.utils.data
    
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from nemo_automodel.recipes.base_recipe import _find_latest_checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils import is_quantized, set_quantizer_state_dict
from modelopt.torch.opt.conversion import restore_from_modelopt_state
from modelopt.torch.utils.logging import print_rank_0


class CustomFinetuneRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction):
    def setup(self):
        super().setup()
        self._quantize_model()

    def _get_quantize_forward_loop(self, data_loader):
        def calibrate_loop(_model):
            device = _model.device
            for batch in data_loader:
                _model(**{"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)})
        return calibrate_loop

    def save_checkpoint(self, epoch: int, step: int):
        """
        Save the current training state as a checkpoint.

        As long as the object has a 'load_state_dict' and 'state_dict' function, it will be saved.

        Args:
            epoch (int): The current epoch.
            step (int): The current step.
        """
        super().save_checkpoint(epoch, step)
        ckpt_dir = _find_latest_checkpoint(self.checkpointer.config.checkpoint_dir)
        model_dir = os.path.join(ckpt_dir, "model")
        modelopt_state = mto.modelopt_state(self.model_parts[0])
        modelopt_path = os.path.join(model_dir, "modelopt_state.pth")
        torch.save(modelopt_state, modelopt_path)
        print_rank_0(f"Modelopt state saved to {modelopt_path}")
        if self.checkpointer.config.save_consolidated:
            consolidated_path = os.path.join(model_dir, "consolidated")
            if os.path.exists(consolidated_path):
                torch.save(modelopt_state, os.path.join(consolidated_path, "modelopt_state_consolidated.pth"))

    def load_checkpoint(self, restore_from=None):

        restore_from = self.cfg.get("checkpoint.restore_from", None)
        if restore_from is None:
            restore_from = _find_latest_checkpoint(self.checkpointer.config.checkpoint_dir)
        if restore_from is not None:
            model_dir = os.path.join(restore_from, "model")
            if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "modelopt_state.pth")):
                modelopt_state = torch.load(os.path.join(model_dir, "modelopt_state.pth"), weights_only=False)
                restore_from_modelopt_state(self.model_parts[0], modelopt_state)
                print_rank_0(f"Restored modelopt state from {model_dir}/modelopt_state.pth")
        super().load_checkpoint()

    def _quantize_model(self, use_eval_loop=True):
        """Quantize the model. Restore the quantization state if it exists."""
        model = self.model_parts[0]

        model.eval()
        if not is_quantized(model):
            self.quant_cfg = getattr(mtq, self.cfg.modelopt.quant_cfg)
            val_ds_name = list(self.val_dataloaders.keys())[0]
            
            forward_loop = self._get_quantize_forward_loop(self.val_dataloaders[val_ds_name])
            print_rank_0("Quantizing the model")
            mtq.quantize(model, self.quant_cfg, forward_loop)
            print_rank_0("Quantization done!")
        else:
            print_rank_0("Model is already quantized.")

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(model)
        
        model.train()


def main(default_config_path="examples/llm/llama_3_2_1b_hellaswag.yaml"):
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the recipe, and initiates the training loop.
    """
    mto.enable_huggingface_checkpointing()
    cfg = parse_args_and_load_config(default_config_path)
    recipe = CustomFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()