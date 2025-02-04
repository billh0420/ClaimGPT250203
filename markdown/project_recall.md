# Project recall

The model will try to recall theorems given the theorem label.

There is no corpus_validation.txt file since this is just recall.

For validation, a random dictum will be chosen from the corpus.txt file.
The model will generate a prediction from the prompt for this dictum.
The reply will be correct if it matches the theorem of the dictum.

A) corpus

- the recall dictum format is:
    <pre> <|begin_dictum|> <|recall|> some_label some_theorem  <|end_dictum|> </pre>

  where:
  <pre>
    some_theorem ::= some_essential_hypotheses $p some_wff $.
    some_essential_hypotesis ::= $e some_wff
  </pre>

- the prompt is: <pre> <|begin_dictum|> <|recall|> some_label </pre>

B) create_files

Nothing is created or changed in the corpus folder
if the file corpus.txt exists.

Delete corpus.txt (and other files in the corpus folder)
to regenerate the contents of the corpus folder.
You probably also need to delete the contents of the model folder
since the model depends upon the contents of corpus.txt
(e.g., the vocabulary size and the vocabulary tokens).

C) create_model

Nothing is created or changed in the model folder
if the file model.pt exists.

Delete model.pt (and other files in the model folder)
to regenerate the contents of the model folder.
You need to do this if you change the parameters of the model.

D) train_model

* can use log_step of AssertStepLogger during train_model
* can use plot_bucket_step_statistics to plot logged steps
* log_step creates the files errors.txt and oks.txt
  in the folder containing model.pt

E) evaluate_model

We can evaluate a model without running train_model.
* choose a random dictum from the corpus.txt file
* prompt is: <pre> <|begin_dictum|> <|recall|> some_label </pre>
* have the model generate a prediction from the prompt for this dictum
* the reply will be correct if it matches the reply of the dictum

# TODO

- Need a function or class to evaluate model for a single prompt.
  This can be shared with StepLogger and the evaluate_model function.

  Example: see RecallModelEvaluator.