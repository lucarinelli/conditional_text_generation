from pprint import pprint

class ExperimentParameters:
    def __init__(self, 
    run_name="unnamed_run", use_control_codes=True, 
    control_codes_type="special_token", use_supercategories=True, 
    use_categories=False, control_codes_powerset=True, 
    max_control_codes_per_caption=3, captions_per_image_id=5,
    limited_run=True, 
    max_train_set_len=5000, max_val_set_len=1000, model="gpt2", 
    chunk_size_json_mp=500, force_dataset_update=False, 
    random_seed=42, training_args=None, metrics_for_all_epochs=False,
    freezed_layers = 0):
        self.run_name=run_name
        self.use_control_codes=use_control_codes
        self.control_codes_type=control_codes_type
        self.use_supercategories=use_supercategories
        self.use_categories=use_categories
        self.control_codes_powerset=control_codes_powerset
        self.max_control_codes_per_caption=max_control_codes_per_caption
        self.limited_run=limited_run
        self.max_train_set_len=max_train_set_len
        self.max_val_set_len=max_val_set_len
        self.model=model
        self.chunk_size_json_mp=chunk_size_json_mp
        self.force_dataset_update=force_dataset_update
        self.random_seed=random_seed
        self.training_args = training_args
        self.captions_per_image_id= captions_per_image_id
        self.metrics_for_all_epochs = metrics_for_all_epochs
        self.freezed_layers = freezed_layers

        print("Experiment parameters are:")
        pprint(vars(self))
        print("End of experiment parameters")