/* C implementation of the machine learning program shown in the Jupyter notebook https://github.com/calebnwokocha/ClaimGPT250203/blob/main/claim_gpt/claim_gpt.ipynb
 * This is a simplified version that captures the main components of the original Python program. Here are some key points about the implementation:
 *
 * Key Components:
 *
 * Model creation and management
 * Encoder for vocabulary handling
 * Training loop implementation
 * Basic file I/O for corpus handling
 * Inference engine
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <math.h>

#ifdef _WIN32
#include <direct.h>
#define realpath(a, b) _fullpath(b, a, _MAX_PATH)
#endif

// Hyperparameters and constants
#define BLOCK_SIZE 150
#define VOCAB_SIZE 666
#define MAX_TRAIN_EPOCHS 10
#define N_HEAD 10
#define N_LAYER 10
#define LEARNING_RATE 0.0001f
#define MAX_SEQUENCE_LENGTH 1024

// Structure definitions
typedef struct {
    float* weights;
    float* biases;
    int input_dim;
    int output_dim;
} Layer;

typedef struct Encoder {
    char** vocab;    // Array of token strings
    int* stoi;       // String-to-index mapping
    char** itos;     // Mapping: integer -> token string.
    int vocab_size;
} Encoder;

typedef struct {
    Layer** layers;
    int n_layers;
    float* embeddings;
    int vocab_size;
    int block_size;
    int n_head;
    int epoch;
    int step;
    int eval_mode;
    Encoder* encoder;  // Pointer to the encoder
} Model;

typedef struct {
    int** encoded_statements; // Each is a float array (encoded statement)
    int count;
} Dataset;

// Function declarations
Model* create_model(int vocab_size, int block_size, int n_head, int n_layer);
void train_model(Model* model, int max_train_epochs, const char* corpus_file_path, const char* model_file_path);
int* get_encoded_statement(const char* statement, Encoder* encoder, int block_size);
int get_space_token(Encoder* encoder);
char* generate_predicted_dictum(const char* prompt, const char* terminal_token, Model* model);
char* remove_trailing_space_tokens(Encoder* encoder, const char* statement);
char* decode(Encoder* encoder, int* tokens, int token_count);
int* generate_tokens(int max_new_tokens, int* encoded_prefix, int prefix_length, Model* model, int terminal_token_id, int* out_total_length);
int* encode_prompt(const char* prompt, Encoder* encoder, int* out_token_count);


Encoder* create_encoder(char* corpus_path);
void save_model(Model* model, const char* model_file_path);
Model* load_model(char* path);
void free_model(Model* model);

// Function definitions

Encoder* create_encoder(char* corpus_path) {
    Encoder* encoder = malloc(sizeof(Encoder));
    encoder->vocab_size = VOCAB_SIZE;

    // Allocate memory for vocab and stoi
    encoder->vocab = malloc(VOCAB_SIZE * sizeof(char*));
    encoder->stoi = malloc(VOCAB_SIZE * sizeof(int));

    // Initialize with basic tokens
    encoder->vocab[0] = strdup("<|start_claim|>");
    encoder->vocab[1] = strdup("<|given|>");
    encoder->vocab[2] = strdup("<|conclude|>");
    encoder->vocab[3] = strdup("<|end_claim|>");

    // Build vocabulary from corpus
    FILE* fp = fopen(corpus_path, "r");
    if (!fp) {
        printf("Error: Could not open corpus file\n");
        exit(1);
    }

    char line[MAX_SEQUENCE_LENGTH];
    int vocab_index = 4;  // Start after special tokens

    while (fgets(line, sizeof(line), fp) && vocab_index < VOCAB_SIZE) {
        char* token = strtok(line, " ");
        while (token) {
            // Check if token is already in vocabulary
            int found = 0;
            for (int i = 0; i < vocab_index; i++) {
                if (strcmp(encoder->vocab[i], token) == 0) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                encoder->vocab[vocab_index] = strdup(token);
                encoder->stoi[vocab_index] = vocab_index;
                vocab_index++;
            }
            token = strtok(NULL, " ");
        }
    }

    fclose(fp);
    return encoder;
}

Model* create_model(int vocab_size, int block_size, int n_head, int n_layer) {
    Model* model = malloc(sizeof(Model));
    model->vocab_size = vocab_size;
    model->block_size = block_size;
    model->n_layers = n_layer;
    model->n_head = n_head;
    model->epoch = 0;
    model->step = 0;

    // Initialize embeddings
    int embedding_dim = n_head * 64;  // Typical embedding dimension
    model->embeddings = malloc(vocab_size * embedding_dim * sizeof(float));
    for (int i = 0; i < vocab_size * embedding_dim; i++) {
        model->embeddings[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    // Initialize transformer layers
    model->layers = malloc(n_layer * sizeof(Layer*));
    for (int i = 0; i < n_layer; i++) {
        model->layers[i] = malloc(sizeof(Layer));
        model->layers[i]->input_dim = embedding_dim;
        model->layers[i]->output_dim = embedding_dim;

        // Initialize weights and biases
        int weights_size = embedding_dim * embedding_dim;
        model->layers[i]->weights = malloc(weights_size * sizeof(float));
        model->layers[i]->biases = malloc(embedding_dim * sizeof(float));
        for (int j = 0; j < weights_size; j++) {
            model->layers[i]->weights[j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
        for (int j = 0; j < embedding_dim; j++) {
            model->layers[i]->biases[j] = 0.0f;
        }
    }

    return model;
}

// Helper function to obtain the space token's integer value.
// This function searches the encoder's vocabulary for "@".
int get_space_token(Encoder* encoder) {
    for (int i = 0; i < encoder->vocab_size; i++) {
        if (strcmp(encoder->vocab[i], "@") == 0) {
            return encoder->stoi[i];
        }
    }
    // If "@" is not found, return 0 as a default.
    return 0;
}

// get_encoded_statement:
// Takes an input statement (string), an encoder, and a desired block_size.
// It splits the statement into tokens, encodes each token using the encoder,
// and returns an integer array of length block_size. If the encoded token
// sequence is shorter than block_size, it is padded with the space token.
// If it is longer, it is truncated.
int* get_encoded_statement(const char* statement, Encoder* encoder, int block_size) {
    // Allocate an array of ints for the encoded statement.
    int* encoded = malloc(block_size * sizeof(int));
    if (!encoded) {
        fprintf(stderr, "Memory allocation failed in get_encoded_statement\n");
        exit(1);
    }

    // Create a modifiable copy of the input statement.
    char* statement_copy = strdup(statement);
    if (!statement_copy) {
        fprintf(stderr, "Memory allocation failed for statement copy\n");
        exit(1);
    }

    int count = 0;
    // Tokenize the statement using whitespace as delimiter.
    char* token = strtok(statement_copy, " ");
    while (token != NULL && count < block_size) {
        int token_index = -1;
        // Search for the token in the encoder's vocabulary.
        for (int i = 0; i < encoder->vocab_size; i++) {
            if (strcmp(encoder->vocab[i], token) == 0) {
                token_index = encoder->stoi[i];
                break;
            }
        }
        // If the token is not found, assign a default value (e.g., 0).
        if (token_index == -1) {
            token_index = 0;
        }
        encoded[count++] = token_index;
        token = strtok(NULL, " ");
    }

    free(statement_copy);

    // If fewer tokens than block_size, pad the remaining entries with the space token.
    int space_token = get_space_token(encoder);
    for (int i = count; i < block_size; i++) {
        encoded[i] = space_token;
    }

    return encoded;
}

void train_model(Model* model, int max_train_epochs, const char* corpus_file_path, const char* model_file_path) {
    if (max_train_epochs == 0)
        return;

    // For this simplified C implementation we assume CPU-only.
    const char* device = "CPU";

    // Retrieve the encoder from the model.
    Encoder* encoder = model->encoder;
    printf("vocab_size=%d\n", encoder->vocab_size);

    // Print the corpus file absolute path.
    char abs_path[PATH_MAX];
#ifdef _WIN32
    _fullpath(abs_path, corpus_file_path, PATH_MAX);
#else
    if (realpath(corpus_file_path, abs_path) == NULL) {
        strncpy(abs_path, corpus_file_path, PATH_MAX);
    }
#endif
    printf("corpus_file_path=%s\n", abs_path);

    // Open the corpus file and count the number of statements (lines).
    FILE* fp = fopen(corpus_file_path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open corpus file %s\n", corpus_file_path);
        exit(1);
    }
    int corpus_statement_count = 0;
    char line[MAX_SEQUENCE_LENGTH];
    while (fgets(line, sizeof(line), fp)) {
        if (strlen(line) > 0)
            corpus_statement_count++;
    }
    printf("corpus_statement_count=%d\n", corpus_statement_count);
    rewind(fp);

    // Read all statements into memory.
    char** corpus_statements = malloc(corpus_statement_count * sizeof(char*));
    int idx = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strlen(line) > 0) {
            // Remove newline characters.
            line[strcspn(line, "\n")] = '\0';
            corpus_statements[idx] = strdup(line);
            idx++;
        }
    }
    fclose(fp);

    // Encode each training statement.
    int embedding_dim = N_HEAD * 64;
    int** encoded_train_statements = malloc(corpus_statement_count * sizeof(float*));
    for (int i = 0; i < corpus_statement_count; i++) {
        encoded_train_statements[i] = get_encoded_statement(corpus_statements[i], encoder, model->block_size);
    }
    printf("#encoded_train_statements=%d\n", corpus_statement_count);

    // Create a dataset.
    Dataset train_dataset;
    train_dataset.encoded_statements = encoded_train_statements;
    train_dataset.count = corpus_statement_count;

    // Set training parameters.
    int train_batch_size = 10;  // number of samples per batch
    int eval_interval = max_train_epochs / 10;
    if (eval_interval < 1) eval_interval = 1;
    int n_layer = model->n_layers;
    float learning_rate = LEARNING_RATE;
    printf("=== train and evaluate model ===\n");
    printf("epoch=%d; step=%d; n_head=%d; n_layer=%d\n", model->epoch, model->step, model->n_head, model->n_layers);
    printf("train_batch_size=%d\n", train_batch_size);
    printf("max_train_epochs=%d; eval_interval=%d\n", max_train_epochs, eval_interval);
    printf("block_size=%d; n_layer=%d; learning_rate=%f; device=%s\n", model->block_size, n_layer, learning_rate, device);
    printf("#train_dataset=%d\n", train_dataset.count);

    // Start timer.
    clock_t start_time = clock();

    // Training loop over epochs.
    for (int epoch = 0; epoch < max_train_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int batch_count = 0;
        // Process each encoded training statement.
        for (int i = 0; i < train_dataset.count; i++) {
            int* x = train_dataset.encoded_statements[i];  // Input vector

            // Allocate arrays for activations and pre-activation values.
            int** activations = malloc((n_layer + 1) * sizeof(float*));
            float** z_values = malloc(n_layer * sizeof(float*));
            activations[0] = x;  // Input layer activation is x

            // Forward pass through each layer.
            for (int l = 0; l < n_layer; l++) {
                activations[l + 1] = malloc(embedding_dim * sizeof(float));
                z_values[l] = malloc(embedding_dim * sizeof(float));
                Layer* layer = model->layers[l];
                for (int j = 0; j < embedding_dim; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < embedding_dim; k++) {
                        sum += layer->weights[j * embedding_dim + k] * activations[l][k];
                    }
                    sum += layer->biases[j];
                    z_values[l][j] = sum;
                    // Use ReLU for hidden layers; identity for output.
                    if (l < n_layer - 1)
                        activations[l + 1][j] = (sum > 0.0f ? sum : 0.0f);
                    else
                        activations[l + 1][j] = sum;
                }
            }

            // Compute loss: mean squared error with target equal to x.
            float loss = 0.0f;
            float* delta = malloc(embedding_dim * sizeof(float));
            for (int j = 0; j < embedding_dim; j++) {
                float diff = activations[n_layer][j] - x[j];
                loss += 0.5f * diff * diff;
                delta[j] = diff;  // derivative for identity activation.
            }
            epoch_loss += loss;

            // Backward pass: propagate error and update weights.
            float* delta_prev = malloc(embedding_dim * sizeof(float));
            for (int l = n_layer - 1; l >= 0; l--) {
                Layer* layer = model->layers[l];
                // For hidden layers, multiply delta by ReLU derivative.
                if (l < n_layer - 1) {
                    for (int j = 0; j < embedding_dim; j++) {
                        float relu_deriv = (z_values[l][j] > 0.0f ? 1.0f : 0.0f);
                        delta[j] *= relu_deriv;
                    }
                }
                // Compute delta for the previous layer.
                for (int k = 0; k < embedding_dim; k++) {
                    float sum = 0.0f;
                    for (int j = 0; j < embedding_dim; j++) {
                        sum += layer->weights[j * embedding_dim + k] * delta[j];
                    }
                    delta_prev[k] = sum;
                }
                // Update weights and biases.
                for (int j = 0; j < embedding_dim; j++) {
                    for (int k = 0; k < embedding_dim; k++) {
                        float grad = delta[j] * activations[l][k];
                        layer->weights[j * embedding_dim + k] -= learning_rate * grad;
                    }
                    layer->biases[j] -= learning_rate * delta[j];
                }
                if (l > 0) {
                    memcpy(delta, delta_prev, embedding_dim * sizeof(float));
                }
            }
            free(delta);
            free(delta_prev);
            // Free allocated activations and z_values (except the input x).
            for (int l = 1; l <= n_layer; l++) {
                free(activations[l]);
            }
            free(activations);
            for (int l = 0; l < n_layer; l++) {
                free(z_values[l]);
            }
            free(z_values);

            batch_count++;
            model->step++;  // Increment global step counter.
            if (batch_count % train_batch_size == 0) {
                printf("Epoch %d, Batch %d, Loss: %f\n", epoch, batch_count / train_batch_size, epoch_loss / train_batch_size);
                epoch_loss = 0.0f;
            }
        }
        model->epoch++;  // End-of-epoch update.

        // Optional evaluation every eval_interval epochs.
        if ((epoch + 1) % eval_interval == 0) {
            printf("Evaluation at epoch %d (evaluation not implemented in this example).\n", epoch + 1);
        }
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("elapsed_time=%.2f minutes\n", elapsed_time / 60.0);
    printf("epoch=%d; step=%d\n", model->epoch, model->step);

    // Save the model.
    printf("saving model: epoch=%d model_file_path=%s\n", model->epoch, model_file_path);
    save_model(model, model_file_path);

    // Free the allocated memory for the dataset.
    for (int i = 0; i < train_dataset.count; i++) {
        free(train_dataset.encoded_statements[i]);
        free(corpus_statements[i]);
    }
    free(train_dataset.encoded_statements);
    free(corpus_statements);
}

// encode_prompt: Tokenizes a prompt (using whitespace) and returns a dynamic array of ints.
// The number of tokens is returned via out_token_count.
// (Caller is responsible for freeing the returned array.)
int* encode_prompt(const char* prompt, Encoder* encoder, int* out_token_count) {
    int count = 0;
    char* prompt_copy = strdup(prompt);
    char* token = strtok(prompt_copy, " ");
    while (token != NULL) {
        count++;
        token = strtok(NULL, " ");
    }
    free(prompt_copy);
    *out_token_count = count;

    int* encoded = malloc(count * sizeof(int));
    int index = 0;
    prompt_copy = strdup(prompt);
    token = strtok(prompt_copy, " ");
    while (token != NULL) {
        int token_index = 0;
        int found = 0;
        for (int i = 0; i < encoder->vocab_size; i++) {
            if (strcmp(encoder->vocab[i], token) == 0) {
                token_index = encoder->stoi[i];
                found = 1;
                break;
            }
        }
        if (!found) token_index = 0; // default value if token not found
        encoded[index++] = token_index;
        token = strtok(NULL, " ");
    }
    free(prompt_copy);
    return encoded;
}

// Stub for generate_tokens: In a complete implementation, this function would run the model's\n// inference loop to generate new tokens. Here it simply returns the prefix unchanged.
int* generate_tokens(int max_new_tokens, int* encoded_prefix, int prefix_length, Model* model, int terminal_token_id, int* out_total_length) {
    // For demonstration, we do not generate new tokens and simply return the prefix.
    *out_total_length = prefix_length;
    int* result = malloc(prefix_length * sizeof(int));
    for (int i = 0; i < prefix_length; i++) {
        result[i] = encoded_prefix[i];
    }
    return result;
}

// Stub for decode: Converts an array of token ints into a string using encoder->itos mapping.
// The caller is responsible for freeing the returned string.
char* decode(Encoder* encoder, int* tokens, int token_count) {
    // Assume a maximum of 20 characters per token plus a separating space.
    int buffer_size = token_count * 21 + 1;
    char* decoded = malloc(buffer_size);
    decoded[0] = '\\0';
    for (int i = 0; i < token_count; i++) {
        strncat(decoded, encoder->itos[tokens[i]], buffer_size - strlen(decoded) - 1);
        if (i < token_count - 1) {
            strncat(decoded, " ", buffer_size - strlen(decoded) - 1);
        }
    }
    return decoded;
}

// Stub for remove_trailing_space_tokens: Trims trailing whitespace characters from the statement.
char* remove_trailing_space_tokens(Encoder* encoder, const char* statement) {
    (void)encoder;  // Unused in this simple implementation.
    int len = strlen(statement);
    while (len > 0 && (statement[len - 1] == ' ' || statement[len - 1] == '\\t' || statement[len - 1] == '\\n')) {
        len--;
    }
    char* trimmed = malloc(len + 1);
    strncpy(trimmed, statement, len);
    trimmed[len] = '\\0';
    return trimmed;
}

// --- generate_predicted_dictum Implementation ---

// generate_predicted_dictum:
//   Given a prompt and a terminal token (as strings), uses the model to generate a predicted dictum.
//   It calculates how many new tokens to generate, encodes the prompt, calls generate_tokens,
//   decodes the generated token sequence, and removes trailing space tokens.
char* generate_predicted_dictum(const char* prompt, const char* terminal_token, Model* model) {
    // Ensure terminal_token is provided.
    if (terminal_token == NULL) {
        fprintf(stderr, "Error: terminal_token is NULL\\n");
        exit(1);
    }

    Encoder* encoder = model->encoder;

    // Tokenize and encode the prompt (without padding) to determine its length.
    int token_count = 0;
    int* encoded_prefix = encode_prompt(prompt, encoder, &token_count);

    // Calculate the maximum number of new tokens to generate.
    int max_new_tokens = model->block_size - token_count;
    if (max_new_tokens < 0)
        max_new_tokens = 0;

    // Find the terminal token's integer representation.
    int terminal_token_id = -1;
    for (int i = 0; i < encoder->vocab_size; i++) {
        if (strcmp(encoder->vocab[i], terminal_token) == 0) {
            terminal_token_id = encoder->stoi[i];
            break;
        }
    }
    if (terminal_token_id == -1)
        terminal_token_id = 0; // default if not found

    // Generate tokens: this function appends new tokens to the prefix until max_new_tokens are generated or
    // the terminal token is encountered. The output length is stored in out_total_length.
    int total_length = 0;
    int* generated_tokens = generate_tokens(max_new_tokens, encoded_prefix, token_count, model, terminal_token_id, &total_length);

    // Decode the generated token sequence into a string.
    char* decoded = decode(encoder, generated_tokens, total_length);

    // Remove any trailing space tokens from the decoded string.
    char* predicted_dictum = remove_trailing_space_tokens(encoder, decoded);

    // Free temporary resources.
    free(encoded_prefix);
    free(generated_tokens);
    free(decoded);

    return predicted_dictum;
}

// Stub for save_model
void save_model(Model* model, const char* model_file_path) {
    printf("Model saved to %s (stub function).\n", model_file_path);
}

// Stub for load_model
Model* load_model(char* path) {
    printf("Loading model from %s (stub function).\n", path);
    return NULL;
}

// Stub for free_model
void free_model(Model* model) {
    if (!model) return;
    if (model->embeddings) free(model->embeddings);
    if (model->layers) {
        for (int i = 0; i < model->n_layers; i++) {
            if (model->layers[i]) {
                if (model->layers[i]->weights) free(model->layers[i]->weights);
                if (model->layers[i]->biases) free(model->layers[i]->biases);
                free(model->layers[i]);
            }
        }
        free(model->layers);
    }
    if (model->encoder) {
        if (model->encoder->vocab) {
            for (int i = 0; i < model->encoder->vocab_size; i++) {
                if (model->encoder->vocab[i]) free(model->encoder->vocab[i]);
            }
            free(model->encoder->vocab);
        }
        if (model->encoder->stoi) free(model->encoder->stoi);
        free(model->encoder);
    }
    free(model);
}

int main(int argc, char* argv[]) {
    // Verify that the inference mode is requested.
    if (argc < 2 || strcmp(argv[1], "inference") != 0) {
        printf("Usage: %s inference\n", argv[0]);
        return 1;
    }

    // Define the model file path.
    const char* model_file_path = "claim_gpt_output/main_claim/models/model.bin";

    // Load the pre-trained model.
    Model* model = load_model((char*)model_file_path);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model from %s\n", model_file_path);
        return 1;
    }

    // Inference engine prompt.
    printf("ClaimGPT Inference Engine\n");
    printf("Type a prompt and press Enter. Type 'exit' to quit.\n\n");

    char prompt[1024];
    while (1) {
        printf("Enter prompt: ");
        if (!fgets(prompt, sizeof(prompt), stdin)) {
            break;  // End-of-file or error.
        }
        // Remove trailing newline character.
        prompt[strcspn(prompt, "\n")] = '\0';

        // Allow user to exit.
        if (strcmp(prompt, "exit") == 0) {
            break;
        }

        // Generate predicted dictum using the provided terminal token "<|end_claim|>".
        char* predicted_dictum = generate_predicted_dictum(prompt, "<|end_claim|>", model);
        if (predicted_dictum) {
            printf("Predicted dictum: %s\n\n", predicted_dictum);
            free(predicted_dictum);
        } else {
            printf("Error generating dictum.\n\n");
        }
    }

    // Clean up the model resources.
    free_model(model);

    return 0;
}
