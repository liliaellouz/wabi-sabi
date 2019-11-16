
import re
from autocorrect import spell


def rmv_apostrophe(string):
    # specific
    phrase = re.sub(r"won\'t", "will not", string)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"\'cause'", "because", phrase)
    phrase = re.sub(r"let\s", "let us", phrase)
    phrase = re.sub(r"ma\'am", "madam", phrase)
    phrase = re.sub(r"y\'all'", "you all", phrase)
    phrase = re.sub(r"o\'clock", "of the clock", phrase)    

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    # special apostrophes
    phrase = re.sub(r"\'", " ", phrase)
    
    return phrase


def string_transformer(string, tokenizer):
    # lowercase
    # (apostrophes)
    # spaces before periods
    phrase = string.lower()
    
    phrase = rmv_apostrophe(phrase)
    
    sentence_list = [sentence.strip() for sentence in phrase.strip().split('.')]
    sentence_list = [sentence for sentence in sentence_list if sentence !='']
    sentence_list = [sentence+' .' for sentence in sentence_list]
    
    tokenized = [tokenizer.encode(sentence) for sentence in sentence_list]
    
    return [item for sublist in tokenized for item in sublist]