concatenated = " ".join(quotes)
concatenated = concatenated[0:1600]
sentences = concatenated.split('.')
sentences = [sentence for sentence in sentences if sentence !='']
sentences = [sentence+' .' for sentence in sentences]
sentences = [sentence.strip() for sentence in sentences]