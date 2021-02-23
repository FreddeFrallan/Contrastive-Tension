from ContrastiveTension import Inference
import transformers

random_texts = ["This is the first sentence within this example",
         "Here is a second sentence for this example, that is a bit different",
         "Did you know that all polar bears are left handed?",
         "It is a fact that every polar bear prefers the use of their left hand",
                ]

def torchExample():
    model = transformers.AutoModel.from_pretrained('Contrastive-Tension/BERT-Large-CT-STSb')
    tokenizer = transformers.AutoTokenizer.from_pretrained('Contrastive-Tension/BERT-Large-CT-STSb')

    embeddings = Inference.torchGenerateSentenceEmbeddings(model, tokenizer, random_texts)
    print(embeddings.shape)

def tensorflowExample():
    model = transformers.TFAutoModel.from_pretrained('Contrastive-Tension/RoBerta-Large-CT-STSb')
    tokenizer = transformers.AutoTokenizer.from_pretrained('Contrastive-Tension/RoBerta-Large-CT-STSb')

    embeddings = Inference.tensorflowGenerateSentenceEmbeddings(model, tokenizer, random_texts)
    print(embeddings.shape)


if __name__ == '__main__':
    #torchExample()
    tensorflowExample()