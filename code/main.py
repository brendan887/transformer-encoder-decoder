import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchsummary import summary
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Encoder, Decoder
from utilities import Utilities

# import nltk
# nltk.download("punkt")

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def run_encoder(tokenizer):
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    # Classifier
    encoder_model = Encoder(n_embd, n_head, n_layer, tokenizer.vocab_size, block_size, n_input, n_hidden, n_output, block_size).to(device)
    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=learning_rate)
    criterion_cls = nn.CrossEntropyLoss()

    # Train the encoder
    for epoch in range(epochs_CLS):
        total_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs, _ = encoder_model(xb)
            loss = criterion_cls(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_CLS_loader)
        accuracy = compute_classifier_accuracy(encoder_model, train_CLS_loader)
        print(f'Epoch [{epoch+1}/{epochs_CLS}], Loss: {epoch_loss}, Accuracy: {accuracy}%')

    PATH = "encoder.pt"
    torch.save(encoder_model, PATH)
    encoder_model = torch.load(PATH)
    encoder_model.eval()

    # Encoder sanity check
    utils = Utilities(tokenizer, encoder_model)
    sentence = "All should contribute, but the burden should not be excessive for any one group of programs or people."
    utils.sanity_check_encoder(sentence, block_size)

    # Encoder test
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
    encoder_test_accuracy = compute_classifier_accuracy(encoder_model, test_CLS_loader)
    print("Encoder test accuracy:", encoder_test_accuracy)

def run_decoder(tokenizer):
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    # Decoder
    decoder_model = Decoder(n_embd, n_head, n_layer, tokenizer.vocab_size, block_size, block_size).to(device)
    optimizer = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate)
    criterion_lm = nn.CrossEntropyLoss()

    # Train the decoder
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits, loss, _ = decoder_model(xb, yb)
        loss.backward()
        optimizer.step()

        if i == 0 or (i + 1) % eval_interval == 0:
            print(f'Decoder Iteration [{i+1}/{max_iters}], Loss: {loss.item()}')

    PATH = "decoder.pt"
    torch.save(decoder_model, PATH)
    decoder = torch.load(PATH)
    decoder.eval()

    # Get perplexities
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=False)
    print("Train Perplexity:", compute_perplexity(decoder, train_LM_loader, eval_iters))

    inputfile = "speechesdataset/test_LM_obama.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    test_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=False)
    print("Obama Perplexity:", compute_perplexity(decoder, test_LM_loader, eval_iters))

    inputfile = "speechesdataset/test_LM_hbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    test_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=False)
    print("H. Bush Perplexity:", compute_perplexity(decoder, test_LM_loader, eval_iters))

    inputfile = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    test_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    test_LM_loader = DataLoader(test_LM_dataset, batch_size=batch_size, shuffle=False)
    print("W. Bush Perplexity:", compute_perplexity(decoder, test_LM_loader, eval_iters))

    decoder.eval()

    # Decoder sanity check
    with torch.no_grad():
        utils = Utilities(tokenizer, decoder)
        sentence = "All should contribute, but the burden should not be excessive for any one group of programs or people."
        utils.sanity_check_decoder(sentence, block_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["encoder", "decoder"], help="Choose model: encoder or decoder")
    args = parser.parse_args()

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if args.model == "encoder":
        run_encoder(tokenizer)
    elif args.model == "decoder":
        run_decoder(tokenizer)


if __name__ == "__main__":
    main()
