from ContrastiveTension import Inference
import transformers

texts = ["This is the first sentence within this example",
         "Here is a second sentence for this example, that is a bit different",
         "Did you know that all polar bears are left handed?",
         "It is a fact that every polar bear prefers the use of their left hand",
         ]

def torchExample():
    model = transformers.AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

    embeddings = Inference.generateSentenceEmbeddings(model, tokenizer, texts, 'pt')
    print(embeddings.shape)

def tensorflowExample():
    model = transformers.TFAutoModel.from_pretrained('bert-base-uncased')
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

    embeddings = Inference.generateSentenceEmbeddings(model, tokenizer, texts, 'tf')
    print(embeddings.shape)


if __name__ == '__main__':
    #torchExample()
    tensorflowExample()