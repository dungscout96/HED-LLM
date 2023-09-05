from datasets import Dataset
import pandas as pd

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

def instruction():
    return "Translate the following tagging into sentences assuming that parentheses mean association: '"


def examples_to_tsv():
    examples_dict = create_examples()
    df = pd.DataFrame.from_dict(examples_dict)
    with open('examples.tsv', 'w') as fout:
        df.to_csv(fout, index=False, sep='\t')

if __name__ == "__main__":
    examples_to_tsv()
