## Structure of a particular Project

### Decide on a descriptive prefix that describes the dictums
* For example, use the prefix 'recall' where the format of the dictum is:
    
<pre>
    <|begin_dictum|>
        <|recall|>
        some_assertion_label
        some_essential_hypotheses
        $p some_assertion_statement $.
    <|end_dictum|>
</pre>

and the prompt will be

<pre>
    <|begin_dictum|> <|recall|> some_assertion_label
</pre>

and the reply will be the "recall" of the theorem (without its proof).

### Make a folder recall_gpt
* Make a folder create_files
    * Write a create_files.py program
* Make a folder create_model
    * Write a create_model.py program
* Make a folder train_model
    * Write a train_model.py program
* Make a folder validate_model
    * Write a validate_model.py program

### Make a folder main_recall
* Make a folder corpus (will contain corpus.txt and supporting files)
* Make a folder models
    * Make a folder model (will contain model.pt and supporting files)
* Write a main_train_recall.py program

You will run main_train_recall.py over and over to train the model.
