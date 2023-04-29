import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from kobert_tokenizer import KoBERTTokenizer
import re
from quickspacer import Spacer
from symspellpy_ko import KoSymSpell, Verbosity

device_kind = ""

domain = [
    "Happy",
    "Fear",
    "Embarrassed",
    "Sad",
    "Rage",
    "Hurt",
]

if torch.cuda.is_available():    
    device = torch.device("cuda")
    device_kind="cuda"
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    device_kind="cpu"
    print('No GPU available, using the CPU instead.')

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
batch_size = 64
max_len = 64
tok = tokenizer.tokenize

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=6,
                 dr_rate=None,
                 params=None):
        super(self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         
    def __len__(self):
        return (len(self.labels))

def predict(sentence):
    print("----------- \n Sentence : ")
    
    print(sentence)
    dataset = [[sentence, '0']]
    test = BERTDataset(dataset, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=2)
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        for logits in out:
            logits = logits.detach().cpu().numpy()
            print("Max : ", domain[np.argmax(logits)])


    print("Result : ", max(logits))
    return [logits, domain[np.argmax(logits)], logits[np.argmax(logits)] ]

model = torch.load('./SentimentAnalysisKOBert.pt', map_location=torch.device(device_kind))

def preprocessing(text):
    sym_spell = KoSymSpell()
    sym_spell.load_korean_dictionary(decompose_korean=True, load_bigrams=True)
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', '', text)
    
    processedText = ""
    words = text.split()

    for word in words:
        result = sym_spell.lookup(word, Verbosity.ALL)
        if len(result) > 0:
            processedText += result[0].term + " "
        else:
            processedText += word + " "
    
    spacer = Spacer(level=3)
    text = text.rstrip().lstrip()
    text = processedText.replace("\n", " ").replace(" ", "")
    text = spacer.space([processedText])
    
    return text[0]

def analyze_word(row):
    try:
        result = predict(preprocessing(row))
    except:
        print("Get some err")
        return
    return result
