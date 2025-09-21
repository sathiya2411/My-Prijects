import torch
import torch.nn as nn
from transformers import BertModel

class BertMentalHealthClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2, dropout=0.3):
        super(BertMentalHealthClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits