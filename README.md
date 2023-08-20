# BioMed-AdapterGPT
THe AI generative model BioMedical based on LLM Quantization 

### Intro
The main idea behind this Project is Implement LoRa Method to reduce the complexity of the BioMedGPT which relate to topic of

 **Quantization**
The main Method used Please Follwo this [QBioMed](src/QBiomed/LoRa/Quantized.py)


**Note** The Repo still under Progress 


### Applying LoRa 

```python 

"""
    Apply LoRa to reduce the complexity of the model 
    Params:
    lora_r = 4
    USE_LORA = True
    """
    lora_r = 4
    USE_LORA = True
    if USE_LORA:
        make_lora_replace(model, verbose=True)
        if cfg.model.bitfit:
            for name, param in model.named_parameters():
                if ("layer_norm" in name and "bias" in name) or ("fc" in name and "bias" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
```

### Update the Saving Wieght Function to support LoRa :


**utils_Checkpoint.py** and **Trainer.py**


```python

def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        logger.info(f"Saving checkpoint to {filename}")
        # call state_dict on all ranks in case it needs internal communication
        state_dict = utils.move_to_cpu(self.state_dict())
        state_dict["extra_state"].update(extra_state)
        if self.should_save_checkpoint_on_current_rank:

            """
            save the model Low-Rank LoRa
            
            """
            checkpoint_utils._torch_persistent_save_LoRa(
                state_dict,
                filename,
                async_write=self.cfg.checkpoint.write_checkpoints_asynchronously,
            )
        logger.info(f"Finished saving checkpoint to {filename}")
            
```

```python

def _torch_persistent_save_LoRa(obj, f):
    if isinstance(f, str):
        with PathManager.open(f, "wb") as h:
            torch_persistent_save(obj, h)
        return
    for i in range(3):
        try:     
            """
            Saving the Wieghts LoRa 
            
            """       
            return torch.save(lora.lora_state_dict(obj), f)

            #return torch.save(obj, f)

        except Exception:
            if i == 2:
                logger.error(traceback.format_exc())
                raise
```