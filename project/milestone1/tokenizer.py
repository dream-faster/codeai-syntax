#!/usr/bin/python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="File containing Python code to be parsed.", required=True)
parser.add_argument(
    '--destination', help="Path to output json file of extracted tokens.", required=True)



def to_token_list(s):
	tokens = [] # list of tokens extracted from source code.
	"""
	TODO: For this task you need to implement a tokenizer that inputs python 
	code file and returns list of tokens.
	"""
	return tokens

def write_tokens(tokens):
	"""
	TODO: Implement your code to write extracted tokens in json format. For format of the 
	JSON file kindly refer to the sample_tokens.json file.
	"""

	raise Exception("Method write_tokens not implemented.")


if __name__ == "__main__":
	args = parser.parse_args()
	
	# extract tokens for the code.
	tokens = to_token_list(args.source)

	# write extracted tokens to file.
	write_tokens(tokens,args.destination)
