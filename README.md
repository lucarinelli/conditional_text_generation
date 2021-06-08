# Conditional text generation

## How to use

### Install the necessary packages
Before being able to use our models and run the generation script, install the necessary dependencies by running the following command:<br></br>
`pip install torch transformers wandb tokenizers`<br></br>
If some of these packages have already been installed, they can be skipped.

### Run the the generation script
To execute the generation script, run the following command:

`python ./generation_script.py --model MODEL --input INPUT --control_codes CONTROL_CODES --max_len MAX_LEN --temperature TEMPERATURE --top_k TOP_K --repetition_penalty REPETITION_PENALTY --top_p TOP_P --num_returned_sequences NUM_RETURNED_SEQUENCES`
Where:
- **MODEL** is one of [ST,SEP,ST-10F,SEP-10F,D-ST,T-ST,ST-0] where:
  -  **ST** is GPT-2 using control codes (_12 layers_);
  -  **SEP** is GPT-2 using serparators (_12 layers_);
  -  **ST-10** is GPT-2 using control codes with 10 layers freezed (_12 layers_);
  -  **SEP-10** is GPT-2 using separators with the first 10 layers freezed (_12 layers_);
  -  **D-T** is Distil-GPT-2 using control codes (_6 layers_);
  -  **T-ST** is Tiny-GPT-2 using control codes (_2 layers_);
  -  **ST-0** is GPT-2 using control codes and trained from scratch (_12 layers_);
- **INPUT** is the sequence the model will start generating from. Note that empty inputs are only allowed on _SEP_ and _SP-0_, for further details, please refer to the _Tokenization_ section of the paper.
- **CONTROL CODES** is a list (possibly empty) of control codes to influence the text genration. For better results it is suggested to use one (or more) of the following: <br></br>
 ['kitchen', 'food', 'animal', 'furniture', 'indoor', 'accessory', 'person', 'vehicle', 'outdoor', 'sports', 'appliance', 'electronic']
- **MAX_LEN**  (_int_, optional, defaults to _16_) is the maximum number of tokens (words) the model will generate, including the input text.
- **TEMPERATURE** (_float_, optional, defaults to _0.9_) – The value used to module the next token probabilities.
- **TOP_K** (_int_, optional, defaults to _30_) – The number of highest probability vocabulary tokens to keep for top-k-filtering.
- **REPETITION PENALTY** (_float_, optional, defaults to _2.0_) – The parameter for repetition penalty. 1.0 means no penalty.
- **TOP_P** (_float_, optional, defaults to _0.7_) – If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
- **NUM_RETURNED_SEQUENCES**  (_int_, optional, defaults to _3_) – The number of independently computed returned sequences for each element in the batch.

