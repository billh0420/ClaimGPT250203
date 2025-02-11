/* C implementation of the machine learning program shown in the Jupyter notebook https://github.com/calebnwokocha/ClaimGPT250203/blob/main/claim_gpt/claim_gpt.ipynb
 * This is a simplified version that captures the main components of the original Python program. Here are some key points about the implementation:
 *
 * Key Components:
 *
 * Model creation and management
 * Encoder for vocabulary handling
 * Basic file I/O for corpus handling
 * Inference engine
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#include <direct.h>
#define realpath(a, b) _fullpath(b, a, _MAX_PATH)
#endif

// Hyperparameters and constants
#define BLOCK_SIZE 150
#define VOCAB_SIZE 666
#define N_HEAD 10
#define N_LAYER 10
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
    char** itos;     // Integer-to-string mapping
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

// Forward declarations for functions used in inference.
Model* create_model(int vocab_size, int block_size, int n_head, int n_layer);
Encoder* create_encoder(char* corpus_path);
char* generate_predicted_dictum(const char* prompt, const char* terminal_token, Model* model);
int* encode_prompt(const char* prompt, Encoder* encoder, int* out_token_count);
int* generate_tokens(int max_new_tokens, int* encoded_prefix, int prefix_length, Model* model, int terminal_token_id, int* out_total_length);
char* decode(Encoder* encoder, int* tokens, int token_count);
char* remove_trailing_space_tokens(Encoder* encoder, const char* statement);
Model* load_model(char* path);
void free_model(Model* model);

// ----- Model and Encoder Functions (Stubs / Simplified Implementations) -----

// Create a new model with random initialization.
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

// Create an encoder using a corpus file.
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
// Decode an array of token IDs into a string.
char* decode(Encoder* encoder, int* tokens, int token_count) {
    int buffer_size = token_count * 21 + 1;  // Assume up to 20 characters per token plus spaces.
    char* decoded = malloc(buffer_size);
    decoded[0] = '\0';
    for (int i = 0; i < token_count; i++) {
        strncat(decoded, encoder->itos[tokens[i]], buffer_size - strlen(decoded) - 1);
        if (i < token_count - 1) {
            strncat(decoded, " ", buffer_size - strlen(decoded) - 1);
        }
    }
    return decoded;
}

// Remove trailing whitespace characters from a decoded statement.
char* remove_trailing_space_tokens(Encoder* encoder, const char* statement) {
    (void)encoder;  // Not used in this simple implementation.
    int len = strlen(statement);
    while (len > 0 && (statement[len - 1] == ' ' || statement[len - 1] == '\t' || statement[len - 1] == '\n')) {
        len--;
    }
    char* trimmed = malloc(len + 1);
    strncpy(trimmed, statement, len);
    trimmed[len] = '\0';
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

// Load a pre-trained model from disk.
// In this simplified example, we simply create a new model and encoder.
Model* load_model(char* path) {
    printf("Loading model from %s...\n", path);
    Model* model = create_model(VOCAB_SIZE, BLOCK_SIZE, N_HEAD, N_LAYER);
    // Create and attach an encoder (using a dummy corpus path).
    model->encoder = create_encoder("math_gpt_output/main_claim/corpus/corpus.txt");
    return model;
}

// Free the model and its associated resources.
void free_model(Model* model) {
    if (!model) return;

    // Free each layer.
    for (int i = 0; i < model->n_layers; i++) {
        free(model->layers[i]->weights);
        free(model->layers[i]->biases);
        free(model->layers[i]);
    }
    free(model->layers);
    free(model->embeddings);

    // Free the encoder.
    if (model->encoder) {
        for (int i = 0; i < model->encoder->vocab_size; i++) {
            free(model->encoder->vocab[i]);
            free(model->encoder->itos[i]);
        }
        free(model->encoder->vocab);
        free(model->encoder->stoi);
        free(model->encoder->itos);
        free(model->encoder);
    }

    free(model);
}

// ----- Inference Engine Main Function -----

int main() {
    srand(time(NULL));

    // Define the path to the pre-trained model.
    const char* model_file_path = "claim_gpt_output/main_claim/models/model.bin";

    // Load the model.
    Model* model = load_model((char*)model_file_path);
    if (!model) {
        fprintf(stderr, "Error: Failed to load model from %s\n", model_file_path);
        return 1;
    }

    printf("Claim GPT Inference Engine\n");
    printf("Type a prompt and press Enter. Type 'exit' to quit.\n\n");

    char prompt[1024];
    while (1) {
        printf("Enter prompt: ");
        if (!fgets(prompt, sizeof(prompt), stdin)) {
            break;  // End-of-file or input error.
        }
        // Remove the trailing newline character.
        prompt[strcspn(prompt, "\n")] = '\0';

        // Check for exit command.
        if (strcmp(prompt, "exit") == 0) {
            break;
        }

        // Generate and print the predicted dictum using "<|end_claim|>" as the terminal token.
        char* predicted_dictum = generate_predicted_dictum(prompt, "<|end_claim|>", model);
        if (predicted_dictum) {
            printf("Predicted dictum: %s\n\n", predicted_dictum);
            free(predicted_dictum);
        } else {
            printf("Error generating dictum.\n\n");
        }
    }

    // Clean up model resources.
    free_model(model);

    return 0;
}
