# WabiSabi

## References
Based on the NeurIPS CAI workshop [paper](http://arxiv.org/abs/1901.08149)

## What it is
Have you ever watched an episode of Rick and Morty and wished you could experience Rick's sarcasm on a more personal level? Have you ever watched Oprah Winfrey interview someone and wished that that person was you? WabiSabi aims at making all of that come true.

Using ConvAI technology, we build personalities of different famous people through their quotes and speeches. These personalities are then used to build a chatbot that you can talk to about your hopes, dreams and worries or have a casual conversation with. WabiSabi has been created with people's wellbeing in mind. Hence, the main focus of our chatbot is to offer pep talks and advice to users when they're feeling down. For instance, if you're feeling anxious or stressed, you could chat with the Oprah Winfrey WabiSabi (doprah).

## How to talk to WabiSabi

To get WabiSabi running, you can either opt for the shell version or the web version. First download all the dependencies.
`logging, argparse, itertools, torch, pytorch_transformers, ignite, socket, json, tempfile, tarfile, autocorrect`

* For the shell version, run the following command
`python interact.py`
    
* For the web version
    * Run the following command
    
    `python server.py` 

    * Go to your favorite web browser and enter the following IP address

    `127.0.0.1:<port number>` where <port number> is the port number you see in the terminal output when you run `server.py`
