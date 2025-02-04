# August 23, 2024

Let the prompt be the assertions for a claim and the target be the claim.

Run through all proofs and get all claims to generate the data.

# August 28, 2024

Format of statement: <|start claim|> premise: t1 ... premise: t2 ... etc conclusion: c1 c2 ... <|end claim|>

The prompt prefix will be: <|start claim|> premise: t1 ... premise: t2 ... etc conclusion:

The prediction will be: conclusion: c1 c2 ... <|end claim|>

The statement Dataset is obtained from the "claim" statement and its premises.

The claims are obtained from the various proof steps of proved assertions. The wffs are ignored.

# Sep 2, 2024

## claim_gpt

- Add ModelValidator class to incorporate validate(...) and keep expensive parameters
- Save GPTLanguageModel init variables when save_model (as done with epoch)

# Sep 27, 2024

## check_syntax_gpt
- prompt is like "<|start|> some_wff_statement <|check_syntax|> some_response"
where some_response is 'valid' or 'invalid'

# Oct 14, 2024

- I might prefer 'evaluate' over 'validate' in function/class names