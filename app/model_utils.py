# Imports
import torch
from tokenizers import Tokenizer
from pathlib import Path
from model import  build_transformer_model

# Special tokens
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ['<UNK>', '<PAD>', '<BOS>', '<EOS>']

#================================================================================
# Public functions to run model
#================================================================================

# Global objects to save model
loaded_model = None
loaded_model_config = None
loaded_model_tokenizers = None

# Function to get model
def get_model():

    # Global variables
    global loaded_model
    global loaded_model_config
    global loaded_model_tokenizers

    # If model is not loaded, load it
    if(loaded_model is None):
        print('Model not loaded, will load from disk.')
        loaded_model, loaded_model_config, loaded_model_tokenizers = _load_saved_model()
    
    # Return model
    return loaded_model, loaded_model_config, loaded_model_tokenizers

# Function to run inference
def inference(input_text):

    # Get model
    model, config, (tokenizer_src, tokenizer_tgt) = get_model()

    # Special tokens
    bos_token = torch.LongTensor([tokenizer_src.token_to_id(SPECIAL_TOKENS[BOS_IDX])])
    eos_token = torch.LongTensor([tokenizer_src.token_to_id(SPECIAL_TOKENS[EOS_IDX])])
    pad_token = torch.LongTensor([tokenizer_src.token_to_id(SPECIAL_TOKENS[PAD_IDX])])

    # Generate encoder inputs
    input_tokens = tokenizer_src.encode(input_text).ids
    padding_tokens_count = config['max_len'] - len(input_tokens) - 2
    encoder_input = torch.cat([
        bos_token, 
        torch.tensor(input_tokens, dtype=torch.int64), 
        eos_token, 
        torch.full((padding_tokens_count,), pad_token.item(), dtype=torch.int64)
    ]).to(config['device'])
    encoder_mask = (encoder_input != pad_token.item()).unsqueeze(0).unsqueeze(0).int().to(config['device'])

    # Generate output
    model_output = _greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, config['max_len'], config['device'])

    # Decode output and return
    pred_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
    return pred_text


#================================================================================
# Utility functions
#================================================================================

# Load saved model
def _load_saved_model():
    
    # Get saved model files and config
    print('Loading config...')
    files = _get_saved_model_files()
    config = _get_saved_model_config()

    # Figure out torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    # Load saved weights and tokenizers
    print('Loading artifacts...')
    model_state = torch.load(files['weights'])
    tokenizer_src = Tokenizer.from_file(str(Path(files['en_tokenizer'])))
    tokenizer_tgt = Tokenizer.from_file(str(Path(files['de_tokenizer'])))

    # Create model object
    print('Building model...')
    model = build_transformer_model(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config['seq_len'],
        config['seq_len'],
        config['d_model'],
        N=config['N'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)

    # Load weights into model
    print('Loading weights into model...')
    model.load_state_dict(model_state['model_state_dict'])

    # Set eval mode
    model.eval()

    # Return model
    print('Done.')
    return model, config, (tokenizer_src,tokenizer_tgt)

# Get saved model files
def _get_saved_model_files():
    return {
        'weights': './artifacts/tf_model_15.pt',
        'en_tokenizer': './artifacts/tokenizer_en.json',
        'de_tokenizer': './artifacts/tokenizer_de.json'
    }

# Get saved model config
def _get_saved_model_config():
    return {
        'seq_len': 50,
        'max_len': 50,
        'd_model': 512,
        'd_ff': 512,
        'N': 3,
        'dropout': 0.1,
    }

def _greedy_decode(model, src, src_mask, tokenizer_tgt, max_len, device):

    # Special tokens for tgt sequence
    bos_idx = tokenizer_tgt.token_to_id(SPECIAL_TOKENS[BOS_IDX])
    eos_idx = tokenizer_tgt.token_to_id(SPECIAL_TOKENS[EOS_IDX])

    # Compute encoder outputs
    encoder_output = model.encode(src, src_mask)

    # Compute decoder input and output
    decoder_input = torch.empty(1,1).fill_(bos_idx).type_as(src).to(device)
    while True:

        # Break if we are at max length
        if(decoder_input.size(1) == max_len):
            break

        # Target mask
        decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type_as(src).to(device)

        # Model output
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

        # Get next token and add to output
        probs = model.project(out[:,-1])
        _, next_token = torch.max(probs, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(src).fill_(next_token.item()).to(device)], dim=1)

        # Break if we generated EOS
        if next_token == eos_idx:
            break

    # Return output
    return decoder_input.squeeze(0)

#================================================================================
# Testing code
#================================================================================

# Code to test model
if __name__ == '__main__':

    # Sample texts to test
    input_texts = [
        'Three little boys are walking down the road.',
        'A man in an orange hat starring at something.',
        'A group of people standing in front of an igloo.',
        'Three people sit in a cave.',
        'People standing outside of a building.',
        'Group of Asian boys wait for meat to cook over barbecue.',
        'A man in uniform and a man in a blue shirt are standing in front of a truck.'
    ]

    # Test model loading and inference
    print('\nRunning tests...\n')
    output_texts = []
    for text in input_texts:
        output_texts.append(inference(text))

    # Print results
    print('\nResults:\n')
    for i in range(len(input_texts)):
        print(f'EN: {input_texts[i]}')
        print(f'DE: {output_texts[i]}')
        if(i != len(input_texts)-1):
            print('-'*100)

    