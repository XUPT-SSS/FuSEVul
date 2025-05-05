import torch
import torch.nn as nn

class SelfAttention(torch.nn.Module):
    def __init__(self, embed_size, dimen_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.dimen_size = dimen_size

        # 初始化权重矩阵
        self.values = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.keys = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.queries = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.assist = torch.nn.Linear(embed_size, dimen_size, bias=False)

    def forward(self, code_output, text_output):
        outs = []
        for i in range(code_output.shape[0]):
            # 提取每个代码的特征向量
            code = code_output[i].unsqueeze(0)
            text = text_output[i].unsqueeze(0)

            values = self.values(code)
            keys = self.keys(code)
            queries = self.queries(code)
            assist = self.assist(text)

            # 计算注意力分数
            attention = torch.matmul(queries, keys.permute(0, 2, 1))
            attention = torch.matmul(attention, assist)
            attention = attention / (self.embed_size ** 0.5)
            # print(attention.shape)

            # 使用softmax函数计算注意力权重
            attention = nn.functional.softmax(attention, dim=-1)
            # 使用权重对值进行加权平均
            out = torch.matmul(attention, values)
            # 变回初始维度并组合

            outs.append(out)
        output = torch.cat(outs, dim=0)
        # print(output.shape)
        return output