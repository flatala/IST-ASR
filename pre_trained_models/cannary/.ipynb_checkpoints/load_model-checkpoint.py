from nemo.collections.asr.models import EncDecMultiTaskModel

# 1. Load the pretrained model
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-flash')



# 3. Save the model to your desired location
save_path = '/path/to/your/location/canary_1b_flash.nemo'
canary_model.save_to(save_path)

# Later, you can load the saved model with:
# restored_model = EncDecMultiTaskModel.restore_from(save_path)


# 2. Update the decoding configuration (set beam size to 1 for greedy decoding)
# decode_cfg = restored_model.cfg.decoding
# decode_cfg.beam.beam_size = 1
# restored_model.change_decoding_strategy(decode_cfg)