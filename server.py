from flask import Flask, json, render_template, redirect, request, abort, render_template_string, session
import re
import wikiquotes
from autocorrect import Speller

import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model



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


def string_transformer(string, tokenizer, spell_check=True):
    # handle lowercase, apostrophes, and spaces before periods
    phrase = string.lower()
    
    if spell_check:
        spell = Speller(lang='en')
        phrase = rmv_apostrophe(spell(phrase))

    sentence_list = phrase.split('.')
    sentence_list = [sentence.strip() for sentence in sentence_list]
    sentence_list = [sentence+' .' for sentence in sentence_list]

    tokenized = [tokenizer.encode(sentence) for sentence in sentence_list]
    return [item for sublist in tokenized for item in sublist]


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def init(quotes):
    global history
    global personality
    global tokenizer
    global model
    global args
    global parser
    global logger
    
    # new conversation
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
	
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    logger.info("Get personality")
    personality =  [string_transformer('my name is WabiSabi', tokenizer, False)]
    random.shuffle(quotes)
    quotes = quotes[:16]
    [personality.append(string_transformer(s, tokenizer)) for s in quotes]
    # print(personality)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    history = []



################################################################################################
#### INIT/MAIN CODE
################################################################################################


api = Flask(__name__, template_folder='static')

parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=200, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.info(pformat(args))

if args.model_checkpoint == "":
    if args.model == 'gpt2':
        raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
    else:
        args.model_checkpoint = download_pretrained_model()


if args.seed != 0:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


logger.info("Get pretrained model and tokenizer")
tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)

logger.info("Sample a personality")
personality =  [string_transformer('my name is WabiSabi', tokenizer, False)]
quotes = ['do not be afraid to ask for yourself', 
    # 'to escape fear , you must go through it', 
    ' failure is another steppingstone to greatness . ',
    'think like a queen .  queen is not afraid to fail . failure is another steppingstone to greatness . ',    
    'be thankful for what you have ; you will end up having more . if you concentrate on what you do not have, you will never, ever have enough .', 
    'surround yourself with only people who are going to lift you higher .', 
    'the biggest adventure you can ever take is to live the life of your dreams  .',
    'doing the best at this moment puts you in the best place for the next moment .', 
    'real integrity is doing the right thing , knowing that nobody is going to know whether you did it or not .', 
    'the more you praise and celebrate your life , the more there is in life to celebrate .', 
    'passion is energy . feel the power that comes from focusing on what excites you .', 
    # 'lots of people want to ride with you in the limo , but what you want is someone who will take the bus with you when the limo breaks down .',  
    'turn your wounds into wisdom . ', 
    'you can have it all . just not all at once . ', 
    'one of the hardest things in life to learn are which bridges to cross and which bridges to burn . ' ,  
    'challenges are gifts that force us to search for a new center of gravity .',
    # 'the thing you fear most has no power . your fear of it is what has the power . facing the truth really will set you free .', 
    'surround yourself only with people who are going to take you higher .', 
    'you get in life what you have the courage to ask for .', 
    'i trust that everything happens for a reason , even when we are not wise enough to see it .', 
    # 'everybody has a calling . and your real job in life is to figure out as soon as possible what that is , who you were meant to be , and to begin to honor that in the best way possible for yourself .', 
    # 'the key to realizing a dream is to focus not on success but on significance , and then even the small steps and little victories along your path will take on greater meaning .',
    'the biggest adventure you can ever take is to live the life of your dreams .',
    'self-esteem comes from being able to define the world in your own terms and refusing to abide by the judgments of others .',
    'forgiveness is giving up the hope that the past could have been any different .', 
    'luck is a matter of preparation meeting opportunity .', 
    'the whole point of being alive is to evolve into the complete person you were intended to be .', 
    'wisdom equals knowledge plus courage . you have to not only know what to do and when to do it , but you have to also be brave enough to follow through .', 
    'surround yourself with great people .', 
    'i alone cannot change the world , but i can cast a stone across the water to create many ripples .', 
    'whatever the mind of man can conceive and believe, it can achieve .', 
    'whenever you see a successful person, you only see the public glories,  never the private sacrifices to reach them .', 
    # 'at some point you are bound to stumble because if you are constantly doing what we do , raising the bar . if you are constantly pushing yourself higher, higher the law of averages not to mention the myth of icarus predicts that you will at some point fall . And when you do i want you to know this , remember this : there is no such thing as failure . failure is just life trying to move us in another direction . now when you are down there in the hole , it looks like failure .', 
    # 'and when you are down in the hole when that moment comes , it is really okay to feel bad for a little while . give yourself time to mourn what you think you may have lost but then here is the key , learn from every mistake because every experience , encounter , and particularly your mistakes are there to teach you and force you into being more who you are . and then figure out what is the next right move .', 
    'because when you inevitably stumble and find yourself stuck in a hole that is the story that will get you out : what is your true calling ? what is your dharma ? what is your purpose ?', 
    # 'i know that you all might have a little anxiety now but no matter what challenges or setbacks or disappointments you may encounter along the way , you will find true success and happiness if you have only one goal , there really is only one , and that is this : to fulfill the highest most truthful expression of yourself as a human being . you want to max out your humanity by using your energy to lift yourself up , your family and the people around you .', 
    # 'from time to time you may stumble , fall , you will for sure , you will have questions and you will have doubts about your path . but i know this , if you are willing to be guided by , that still small voice that is the gps within yourself , to find out what makes you come alive , you will be more than okay . you will be happy , you will be successful , and you will make a difference in the world .'
    ]
random.shuffle(quotes)
quotes = quotes[:16]
print(quotes)
[personality.append(string_transformer(s, tokenizer)) for s in quotes]
# print(personality)
logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

history = []


################################################################################################
##### SERVER CODE
################################################################################################

@api.route('/')
def index():
    persona = 'WabiSabi'
    if session and session['persona']:
        persona = session['persona']
    return render_template('index.html', name=persona)

@api.route('/answer', methods=['POST'])
def get_answer():
    global history
    global personality
    global tokenizer
    global model
    global args

    # get input text
    if not request.form or not 'input' in request.form:
        abort(400)
    raw_text = request.form['input']
    # print(raw_text)
    if not raw_text:
        json.dumps('Prompt should not be empty!')
    
    # get output text
    history.append(string_transformer(raw_text, tokenizer))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2*args.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    # return generated answer
    return json.dumps(out_text)

@api.route('/personas', methods=['POST'])
def get_personas():
    # get input text
    if not request.form or not 'input' in request.form:
        abort(400)
    prompt = request.form['input']
    # print(raw_text)
    if not prompt:
        json.dumps('Prompt should not be empty!')
    
    return json.dumps(wikiquotes.search(prompt, "english"))

@api.route('/change_persona', methods=['POST'])
def create_new_persona():
    # get input text
    if not request.form or not 'input' in request.form:
        abort(400)
    persona = request.form['input']
    # print(raw_text)
    if not prompt:
        json.dumps('Prompt should not be empty!')
    
    init(wikiquotes.getquotes(persona, "english"))
    session['persona'] = persona
    return redirect('/')



@api.route('/reset')
def reset():
    quotes = ['do not be afraid to ask for yourself', 
    # 'to escape fear , you must go through it', 
    ' failure is another steppingstone to greatness . ',
    'think like a queen .  queen is not afraid to fail . failure is another steppingstone to greatness . ',    
    'be thankful for what you have ; you will end up having more . if you concentrate on what you do not have, you will never, ever have enough .', 
    'surround yourself with only people who are going to lift you higher .', 
    'the biggest adventure you can ever take is to live the life of your dreams  .',
    'doing the best at this moment puts you in the best place for the next moment .', 
    'real integrity is doing the right thing , knowing that nobody is going to know whether you did it or not .', 
    'the more you praise and celebrate your life , the more there is in life to celebrate .', 
    'passion is energy . feel the power that comes from focusing on what excites you .', 
    # 'lots of people want to ride with you in the limo , but what you want is someone who will take the bus with you when the limo breaks down .',  
    'turn your wounds into wisdom . ', 
    'you can have it all . just not all at once . ', 
    'one of the hardest things in life to learn are which bridges to cross and which bridges to burn . ' ,  
    'challenges are gifts that force us to search for a new center of gravity .',
    # 'the thing you fear most has no power . your fear of it is what has the power . facing the truth really will set you free .', 
    'surround yourself only with people who are going to take you higher .', 
    'you get in life what you have the courage to ask for .', 
    'i trust that everything happens for a reason , even when we are not wise enough to see it .', 
    # 'everybody has a calling . and your real job in life is to figure out as soon as possible what that is , who you were meant to be , and to begin to honor that in the best way possible for yourself .', 
    # 'the key to realizing a dream is to focus not on success but on significance , and then even the small steps and little victories along your path will take on greater meaning .',
    'the biggest adventure you can ever take is to live the life of your dreams .',
    'self-esteem comes from being able to define the world in your own terms and refusing to abide by the judgments of others .',
    'forgiveness is giving up the hope that the past could have been any different .', 
    'luck is a matter of preparation meeting opportunity .', 
    'the whole point of being alive is to evolve into the complete person you were intended to be .', 
    'wisdom equals knowledge plus courage . you have to not only know what to do and when to do it , but you have to also be brave enough to follow through .', 
    'surround yourself with great people .', 
    'i alone cannot change the world , but i can cast a stone across the water to create many ripples .', 
    'whatever the mind of man can conceive and believe, it can achieve .', 
    'whenever you see a successful person, you only see the public glories,  never the private sacrifices to reach them .', 
    # 'at some point you are bound to stumble because if you are constantly doing what we do , raising the bar . if you are constantly pushing yourself higher, higher the law of averages not to mention the myth of icarus predicts that you will at some point fall . And when you do i want you to know this , remember this : there is no such thing as failure . failure is just life trying to move us in another direction . now when you are down there in the hole , it looks like failure .', 
    # 'and when you are down in the hole when that moment comes , it is really okay to feel bad for a little while . give yourself time to mourn what you think you may have lost but then here is the key , learn from every mistake because every experience , encounter , and particularly your mistakes are there to teach you and force you into being more who you are . and then figure out what is the next right move .', 
    'because when you inevitably stumble and find yourself stuck in a hole that is the story that will get you out : what is your true calling ? what is your dharma ? what is your purpose ?', 
    # 'i know that you all might have a little anxiety now but no matter what challenges or setbacks or disappointments you may encounter along the way , you will find true success and happiness if you have only one goal , there really is only one , and that is this : to fulfill the highest most truthful expression of yourself as a human being . you want to max out your humanity by using your energy to lift yourself up , your family and the people around you .', 
    # 'from time to time you may stumble , fall , you will for sure , you will have questions and you will have doubts about your path . but i know this , if you are willing to be guided by , that still small voice that is the gps within yourself , to find out what makes you come alive , you will be more than okay . you will be happy , you will be successful , and you will make a difference in the world .'
    ]
    init(quotes)
    return redirect('/')





api.run()


