from datasets import Dataset
import pandas as pd
import requests
from io import StringIO

def create_examples():
    HED = [
        "(Visual-presentation,(Background-view,Black),(Foreground-view,((Center-of,Computer-screen),(Cross,White)),(Grayscale,(Face,Hair,Image))))",
        "(Foreground-view, ((Item-count, High), Ingestible-object)), (Background-view, ((Human, Body, Agent-trait/Adult), Outdoors, Furnishing, Natural-feature/Sky, Urban, Man-made-object))"
    ]

    description = [
        "The visual presentation has a black background view. In its foreground view, the center is associated with a computer screen and there's a white cross. There's also a grayscale element that includes features like a face, hair, and an image.",
        ""
    ]

    return {"HED": HED, "description": description}

def create_hugging_dataset():
    examples_dict = create_examples()
    return Dataset.from_dict(examples_dict)

def create_instructions():
    '''
    Create a list of intruction options in tuples of (<instruction>, <query>)
    '''
    options = [
        ("Translate the following tagging into sentences assuming that parentheses mean association:", "Translation:")
    ]

    return options

def get_examples_from_github():
    endpoint = "https://raw.githubusercontent.com/dungscout96/HED-LLM/main/examples.tsv"
    result = requests.get(endpoint)
    return pd.read_csv(StringIO(result.text), sep="\t")

def examples_to_tsv():
    # examples_dict = create_examples()
    examples_dict = get_examples_from_github().to_dict()
    df = pd.DataFrame.from_dict(examples_dict)
    with open('examples.tsv', 'w') as fout:
        df.to_csv(fout, index=False, sep='\t')

if __name__ == "__main__":
    examples_to_tsv()
