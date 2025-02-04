# evaluate_model.py
# from claim_gpt

import random
import time

from shared import Encoder
from shared import get_encoded_statement

from shared import generate_predicted_dictum

def validate(statements, max_examples: int, encoder: Encoder, model, block_size: int):
    print(f'=== start ===')
    start_time = time.time()
    print(f'max_examples={max_examples}')
    space_token = encoder.encode('@')[0]
    conclusion_token = encoder.stoi['<|conclude|>']
    ok_counts = [0] * block_size  # ok_count by number correct
    error_counts = [0] * block_size  # error_count by number correct
    error_count = 0
    max_error_index = 0
    max_ok_error_index = 0
    for example in range(max_examples):
        random_statement = random.choice(statements)
        encoded_val_statement = get_encoded_statement(random_statement, encoder, block_size)
        val_statement = encoder.decode(encoded_val_statement)
        prompt = val_statement.split(' <|conclude|> ')[0] + ' <|conclude|>'
        terminal_token = '<|end_claim|>'
        predicted_dictum = generate_predicted_dictum(prompt=prompt, terminal_token=terminal_token, model=model)
        encoded_predicted_statement = get_encoded_statement(predicted_dictum, encoder, block_size)
        statement_prefix_index = encoded_val_statement.index(conclusion_token)
        max_prediction_count = len(encoded_val_statement) - statement_prefix_index
        for i in range(0, max_prediction_count):
            encoded_token = encoded_val_statement[statement_prefix_index + i]
            predicted_encoded_token = encoded_predicted_statement[statement_prefix_index + i]
            if predicted_encoded_token != encoded_token:
                max_error_index = max(max_error_index, i)
                error_counts[i] += 1
                error_count += 1
                if error_count <= 6:
                    print(f'===== Example {example + 1} error:{i} =====')
                    stripped_val_statement = encoder.remove_trailing_space_tokens(val_statement)
                    stripped_predicted_statement = encoder.remove_trailing_space_tokens(predicted_dictum)
                    print(stripped_val_statement)
                    print(stripped_predicted_statement)
                break
            elif encoded_token != space_token:
                max_ok_error_index = max(max_ok_error_index, i)
                ok_counts[i] += 1
    print(f'error_count={error_count}; errors={error_counts[:max_error_index + 1]}; oks={ok_counts[:max_ok_error_index + 1]}')
    print(f'error_percentage={error_count * 100 / max_examples: .2f}%')
    elapsed_time = time.time() - start_time
    print(f'elapsed_time={elapsed_time / 60: .2f} minutes')
