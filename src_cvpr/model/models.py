import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm


class Encoder(Module):
    r"""Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    # def forward(self, src):
    #     r"""Pass the input through the endocder layers in turn.

    #     """
    #     output = src

    #     for i in range(self.num_layers):
    #         output = self.layers[i](output)

    #     if self.norm:
    #         output = self.norm(output)

    #     return output

    def forward(self, src, src_key_padding_mask=None):
        """Pass input and mask through encoder layers."""
        output = src
        
        for i in range(self.num_layers):
            output = self.layers[i](output, src_key_padding_mask)
        
        if self.norm:
            output = self.norm(output)
        
        return output


class Decoder(Module):
    r"""Decoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the DecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory)

        if self.norm:
            output = self.norm(output)

        return output


class EncoderLayer(Module):
    r"""EncoderLayer is mainly made up of self-attention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    # def forward(self, src):
    #     r"""Pass the input through the endocder layer.
    #     """
    #     src2 = self.self_attn(src, src, src)[0]
    #     src = src + self.dropout1(src2)
    #     src = self.norm1(src)
    #     if hasattr(self, "activation"):
    #         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    #     else:  # for backward compatibility
    #         src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
    #     src = src + self.dropout2(src2)
    #     src = self.norm2(src)
    #     return src

    def forward(self, src, src_key_padding_mask=None):
        """
        Forward pass with attention masking.
        src_key_padding_mask: (batch_size, seq_len) with True for positions to mask
        """
        src2 = self.self_attn(
            src, src, src,
            key_padding_mask=src_key_padding_mask
        )[0]
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if src_key_padding_mask is not None:
            padding_mask_expanded = (~src_key_padding_mask).transpose(0, 1).unsqueeze(-1)
            src = src * padding_mask_expanded  
        return src


class DecoderLayer(Module):
    r"""DecoderLayer is mainly made up of the proposed cross-modal relation attention (CMRA).

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer.
        """
        memory = torch.cat([memory, tgt], dim=0)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

