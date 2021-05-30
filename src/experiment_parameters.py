from pprint import pprint

class ExperimentParameters:
    def __init__(self, run_name="unnamed_run", use_control_codes=True, control_codes_type="special_token", use_supercategories=True, use_categories=False, control_codes_powerset=True, max_control_codes_per_caption=3, limited_run=True, max_train_set_len=5000, max_val_set_len=1000, model="gpt2", chunk_size_json_mp=500, force_dataset_update=False, training_args=None):
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
        self.training_args = training_args

        print("Experiment parameters are:")
        pprint(vars(self))
        print("End of experiment parameters")