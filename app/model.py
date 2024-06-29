# Imports
import math
import torch
import torch.nn as nn

# Input embedder class
class InputEmbedder(nn.Module):

    # Constructor
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    # Forward function 
    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_model) # The original paper multiples embeddings by sqrt of d_model
        return out

# Positional encoder class
class PositionalEncoder(nn.Module):

    # Constructor
    def __init__(self, d_model:int, seq_length:int, dropout:float):

        # Initialize properties
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # PE matrix - (seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)
        
        # Positions vector for numerator of PE matrix formula - (seq_length, 1)
        num_term = torch.arange(0, seq_length, dtype=torch.float32).unsqueeze(1)
        
        # Denominator of sin/cos arg - Value of 10000^(2i/d_model) for i ranging from 0 to d_model/2 - (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
        
        # Apply sin/cos functions
        trig_arg = num_term * div_term
        pe[:,0::2] = torch.sin(trig_arg)
        pe[:,1::2] = torch.cos(trig_arg)

        # Add batch dimension to make it (1,seq_len,d_model)
        pe = pe.unsqueeze(0)

        # Register buffer
        self.register_buffer('pe', pe)

    # Forward function
    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    
# Layer normalization module
class LayerNormalization(nn.Module):

    # Constructor
    def __init__(self, eps: float = 1e-5):

        # Init
        super().__init__()
        self.eps = eps # This is required for numerical stability and avoid div by zero
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    # Forward method
    def forward(self, x):

        # Calculate mean and std within each sample
        mean = x.mean(dim=-1, keepdim=True) # x is of shape (batch, seq_length, d_model), mean is (batch, seq_length, 1)
        std = x.std(dim=-1, keepdim=True) # std is (batch, seq_length, 1)

        # Normalize and return
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias # Broadcasting will make the return value of shape (batch, seq_length, d_model)
    

# Feed forward layer module
class FeedForwardBlock(nn.Module):

    # Constructor
    def __init__(self, d_model: int, d_ff: int, dropout: float):

        # Init
        super().__init__()

        # Layers
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    # Forward method
    def forward(self, x):
        
        # Pass x (batch, seq_length, d_model) through the network
        out = torch.relu(self.linear_1(x))
        out = self.dropout(out)
        out = self.linear_2(out)

        # Return output
        return out
    

# Multi head attention block
class MultiHeadAttentionBlock(nn.Module):

    # Constructor
    def __init__(self, d_model: int, h: int, dropout: float):

        # Init
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model should be divisible by h"
        self.d_k = d_model // h

        # Setup weight matrices using linear layer
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    # Split heads
    def split_heads(self, x):
        # (batch, seq_len, d_model) => (batch, seq_len, h, d_k) => (batch, h, seq_len, d_k)
        # Each head will now get (seq_len, d_k), i.e. full sequence but partial embeddings
        batch_size, seq_length, d_model = x.shape
        return x.view(batch_size, seq_length, self.h, self.d_k).transpose(1,2)

    # Combine heads
    def combine_heads(self, x):
        batch_size, h, seq_length, d_k = x.shape
        # Revert transpose: (batch, h, seq_len, d_k) => (batch, seq_len, h, d_k)
        x = x.transpose(1, 2)
        # Combine h and d_k
        return x.contiguous().view(batch_size, -1, self.d_model)

    # Attention function
    def calculate_attention(self, query, key, value, mask = None):

        # Calculate scores: (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(self.d_k) # Transpose the (seq_len, dk) part for keys to get (seq_len, seq_len) scores - embeddings (in each head) of each word dotted with every other word  

        # Apply mask
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)

        # Softmax - dims stay (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1) # Last dim (-1) is the weight of each V for the Q in second last dim (-2)

        # Dropout
        attention_scores = self.dropout(attention_scores)

        # Generate output and return
        output = attention_scores @ value # (batch, h, seq_len, d_k) - seq_len, d_k is formed by take weighted sum of value according to attn scores
        return output, attention_scores
    
    # Forward method
    def forward(self, q, k, v, mask):

        # Generate Q', K' and V'. Dims for all of them remain same (batch, seq_len, d_model) => (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split heads. New dim: (batch, h, seq_len, d_k)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Calculate attention scores
        output, attention_scores = self.calculate_attention(query, key, value, mask)

        # Combine heads. New dim: (batch, seq_len, d_model)
        output = self.combine_heads(output)

        # Multiply by output weight matrix and return
        return self.w_o(output) # (batch, seq_len, d_model)
    
# Residual connection block
class ResidualConnectionBlock(nn.Module):

    # Constructor
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    # Forward method
    def forward(self, x, sublayer):

        # Apply norm, dropout, add and return
        return x + self.dropout(sublayer(self.norm(x))) # Applying norm first is different from paper. It improves convergence and removes need of warmig up
    
# Encoder block
class EncoderBlock(nn.Module):

    # Constructor
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(2)])

    # Forward method
    def forward(self, x, src_mask):

        # Apply multi head attention
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x, x, x, src_mask))

        # Apply feed forward network
        x = self.residual_connections[1](x, self.feed_forward_block)

        # Return output
        return x
    
# Encoder layer of the transformer model
class Encoder(nn.Module):

    # Constructor
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    # Forward method
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
# Decoder block
class DecoderBlock(nn.Module):

    # Constructor
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(3)])

    # Forward method
    def forward(self, x, encoder_output, src_mask, tgt_mask):

        # Apply self attention
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x, x, x, tgt_mask))

        # Apply cross attention
        x = self.residual_connections[1](x, lambda x:self.self_attention_block(x, encoder_output, encoder_output, src_mask))

        # Apply feed forward network
        x = self.residual_connections[2](x, self.feed_forward_block)

        # Return output
        return x
    
# Decoder layer of the transformer model
class Decoder(nn.Module):

    # Constructor
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    # Forward method
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

# Linear layer to get output of transformer model
class OutputLinearLayer(nn.Module):

    # Constructor
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    # Forward method
    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)
    
# Transformer model
class TransformerModel(nn.Module):

    # Constructor
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbedder, tgt_embedding: InputEmbedder, src_pos: PositionalEncoder, tgt_pos: PositionalEncoder, output_layer: OutputLinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.output_layer = output_layer

    # Encode method
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src

    # Decode method
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt

    # Output projection method
    def project(self, x):
        return self.output_layer(x)
    
# Function to build transformer model
def build_transformer_model(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> TransformerModel:

    # Create embedder layers
    src_embedding = InputEmbedder(d_model, src_vocab_size)
    tgt_embedding = InputEmbedder(d_model, tgt_vocab_size)

    # Positional encoder layers
    src_pos = PositionalEncoder(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoder(d_model, tgt_seq_len, dropout)

    # Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_ff = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attn, encoder_ff, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attn = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_ff = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attn, decoder_cross_attn, decoder_ff, dropout)
        decoder_blocks.append(decoder_block)

    # Encoder and decoder layers
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Output layer
    output_layer = OutputLinearLayer(d_model, tgt_vocab_size)

    # Transformer model
    transformer = TransformerModel(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, output_layer)

    # Initialize model params - make training faster with xavier
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Return model
    return transformer