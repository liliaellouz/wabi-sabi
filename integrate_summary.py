import wikipedia
import nltk


def translate(word):
    """
    translates third person words into first person words
    """
    forms = {"is" : "am", 'she' : 'I', 'he' : 'I', 'her' : 'my', 'him' : 'me', 'hers' : 'mine', 'your' : 'my', 'has' : 'have'}
    if word.lower() in forms: 
        return forms[word.lower()]
    return word

person_summary = wikipedia.summary("Oprah Winfrey")
person_summary = person_summary[0:(person_summary).find('\n')]
result = ' '.join([translate(word) for word in nltk.wordpunct_tokenize(person_summary)])