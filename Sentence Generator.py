from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')


def get_prediction (sent):
    
    token_ids = tokenizer.encode(sent, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position ]

    with torch.no_grad():
        output = model(token_ids)

    last_hidden_state = output[0].squeeze()

    list_of_list =[]
    for index,mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=5, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        list_of_list.append(words)
    
    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess+" "+j[0]
        
    return best_guess


model_name = 'cointegrated/rubert-tiny'
model_score = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')


def get_prediction (sent):
    
    token_ids = tokenizer.encode(sent, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position ]

    with torch.no_grad():
        output = model(token_ids)

    last_hidden_state = output[0].squeeze()

    list_of_list =[]
    for index,mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=5, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        list_of_list.append(words)
    
    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess+" "+j[0]
        
    return best_guess

def GeneratorText(words):
  X=words
  Z=words.copy()
  i=0
  f=len(X)
  global_sentences = []

#go backward in the list

  while i<f:
    X.insert(i+2,"___")
    sentence=" ".join(X)
    sentence = sentence.replace("___","<mask>")
    predicted_blanks = get_prediction(sentence)
    predicted_blanks=predicted_blanks.split(" ")[1:]
    sentence=(sentence).split(" ")
    sentence=np.array(sentence)
    index, = np.where(sentence == "<mask>")
    for j in range(len(predicted_blanks)):
      sentence[index[j]]=predicted_blanks[j]
      final=" ".join(list(sentence))
      
    global_sentences.append(final)
    i=i+2
  
#go forward in the list
  i=0
  f=len(Z)
  while i+1<f:
    Z.insert(f-1,"___")
    sentence=" ".join(Z)


    sentence = sentence.replace("___","<mask>")
    predicted_blanks = get_prediction(sentence)
    predicted_blanks=predicted_blanks.split(" ")[1:]
    
    sentence=(sentence).split(" ")
    sentence=np.array(sentence)
    index, = np.where(sentence == "<mask>")

    for j in range(len(predicted_blanks)):
      sentence[index[j]]=predicted_blanks[j]
      final=" ".join(list(sentence))

    global_sentences.append(final)
    f=f-1

#remove duplicates

  global_sentences=list(set(global_sentences))
  global_scores=[]

#get scores of generated texts

  for text in global_sentences:
    s=score(sentence=text, model=model, tokenizer=tokenizer)
    global_scores.append(s)

#get max score of the predicted generated sentences

  max_index = global_scores.index(max(global_scores))

  return global_sentences[max_index]
