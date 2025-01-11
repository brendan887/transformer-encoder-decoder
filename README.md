To run, run main.py in with argument encoder or decoder

Encoder:
1. Train the classifier
2. Print the loss and accuracy for epoch
3. Save the model as "encoder.pt"
4. Create and display 4 attention maps, one for first head on each layer. Maps are named as encoder
5. Print test accuracy on test_CLS

Decoder:
1. Train the decoder
2. Print loss for certain iterations
3. Save the model as "decoder.pt"
4. Generate perplexity scores for Train, Obama, H. Bush, and W.Bush
5. Create and display 4 attention maps, one for first head on each layer. Maps are named as decoder

Encoder and decoder architectures are implemented in transformers.py
