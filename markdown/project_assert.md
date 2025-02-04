# project assert and project assert_logic

The model will try to generate syntactically correct wffs.

There is no need for a corpus_validation.txt file
since the reply is a random wff.

The reply will be checked for being syntactically correct.

The assert_logic project is a special case of an assert project
where we limit the metamath statements to only "logic" statements.

A) corpus

* the assert dictum format is:
    <pre> |- some_wff <|over|> </pre>
* the prompt is: <pre> |-  </pre>
* the terminal token <|over|> is radio talk for "my message is finished."
* I think <|over|> shoud be replaced with the standard <|end_dictum|>
* I think <|begin_dictum|> should also be used.
* The recommended dictum format would be:
    <pre> <|begin_dictum|> |- some_wff <|end_dictum|> </pre>

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
* prompt is: |-
* generate_prediction function generates a random assert dictum
* derive_syntax function evaluates some_wff for syntax correctness
  1. input
     * some_wff: str
     * context: str or None
     * assert_db: AssertDB
     * syntax_deriver_db: SyntaxDeriverDB (will be updated)
       1. math_statement table
       2. rule_errors table
  2. output
     * syntax_derivation: str or None
* SyntaxDeriverValidationReporter prints the full evaluation results
