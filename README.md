## SageMaker_seq2seq_WordPronunciation

Sequence to Sequence modeling have seen great performance in building models where the input is a sequence of tokens (words for example) and output is also a sequence of tokens. The notebook provides an end-to-end example of training and hosting the English word pronunciation model using the Amazon SageMaker built-in Seq2Seq.  

## SageMaker-Seq2Seq-word-prpnunciation

Jupyter notebook to demonstrate an end-to-end example of training and hosting the English word pronunciation model. 

Note: The training the model with the exact same setup will take ~2 hours. 

## create_vocab_proto.py

Helper python script to generate a recordIO file from tokenized source and target sequences in numpy array. 

## record_pb2.py

Another helper python script to generate a recordIO file from tokenized source and target sequences in numpy array. 

## License

This library is licensed under the Apache 2.0 License. 
