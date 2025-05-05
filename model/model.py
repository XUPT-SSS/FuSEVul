import torch
import torch.nn as nn
from selfattention import SelfAttention


class Code_Note(nn.Module):
    def __init__(self, code_encoder, text_encoder, input_size, hidden_size, output_size):
        super(Code_Note, self).__init__()
        self.code_encoder = code_encoder
        self.text_encoder = text_encoder
        self.attention = SelfAttention(768, 512)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(output_size, 2)

    def forward(self, inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask):
        code_output = self.code_encoder.encoder(inputs_code_id, attention_mask=inputs_code_mask).last_hidden_state
        text_output = self.text_encoder(inputs_text_id, attention_mask=inputs_text_mask).last_hidden_state
        # concat
        #output = torch.cat((code_output, text_output), dim=-1)
        # add
        #output = code_output + text_output
        # max pooling
        #output = torch.max(code_output, text_output)
        # self and cross attention
        output = self.attention(code_output, text_output)[:, 0, :]
        output = self.fc1(code_output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        return output
