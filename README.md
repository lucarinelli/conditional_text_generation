# Conditional text generation

## How to use

### Install the necessary packages
Before being able to use our models and run the generation script, install the necessary dependencies by running the following command:
`pip install -r requirements.txt`

### Run the the generation script
To execute the generation script, tun the following command:

`python ./generation_script.py --model MODEL --input INPUT --control_codes CONTROL_CODES --max_len MAX_LEN --temperature TEMPERATURE --top_k TOP_K --repetition_penalty REPETITION_PENALTY --top_p TOP_P --num_returned_sequences NUM_RETURNED_SEQUENCES`
Where:
- **MODEL** is one of the following: [ST,SEP,ST_10F,SEP_10F,D_ST];
- **INPUT** is the sequence the model will start generating from.
- **CONTROL CODES** is a list (possibly empty) of control codes to influence the text genration. For better results it is suggested to use one (or more) of the following: <br></br>
 ['kitchen', 'food', 'animal', 'furniture', 'indoor', 'accessory', 'person', 'vehicle', 'outdoor', 'sports', 'appliance', 'electronic']
- **MAX_LEN**  (_int_, optional, defaults to _16_) is the maximum number of tokens (words) the model will generate, including the input text.
- **TEMPERATURE** (_float_, optional, defaults to _0.9_) – The value used to module the next token probabilities.
- **TOP_K** (_int_, optional, defaults to _30_) – The number of highest probability vocabulary tokens to keep for top-k-filtering.
- **REPETITION PENALTY** (_float_, optional, defaults to _2.0_) – The parameter for repetition penalty. 1.0 means no penalty.
- **TOP_P** (_float_, optional, defaults to _0.7_) – If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
- **NUM_RETURNED_SEQUENCES**  (_int_, optional, defaults to _3_) – The number of independently computed returned sequences for each element in the batch.

