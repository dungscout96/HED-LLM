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
    df = get_examples_from_github()
    examples_dict = df.to_dict(orient='list')
    return Dataset.from_dict(examples_dict)

def create_instructions():
    '''
    Create a list of intruction options in tuples of (<instruction>, <query>)
    '''
    options = [
        ("Translate the following tagging into sentences assuming that parentheses mean association:", "Translation:")
    ]
    for idx, option in enumerate(options):
        print(f'''Option {idx}:\n\tInstruction: "{option[0]}"\n\tQuery: "{option[1]}"\n\n''')
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

def make_prompt(dataset, example_indices_full, example_index_to_translate, instruction, query):
    prompt = ''
    for index in example_indices_full:
        hed = dataset[index]['HED']
        desc = dataset[index]['description']
        
        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
{instruction}

{hed}

{query}
{desc}


"""
    
    hed = dataset[example_index_to_translate]['HED']
    
    prompt += f"""
{instruction}

{hed}

{query}
"""
        
    return prompt
    
if __name__ == "__main__":
    examples_to_tsv()
