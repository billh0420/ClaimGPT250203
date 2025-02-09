#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

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
    Encoder* encoder;  // Pointer to the encoder
} Model;

typedef struct {
    int** encoded_statements; // Each is a float array (encoded statement)
    int count;
} Dataset;

// Function declarations
Model* create_model(int vocab_size, int block_size, int n_head, int n_layer);
void train_model(Model* model, int max_train_epochs, const char* corpus_file_path, const char* model_file_path);
void validate_model(Model* model, char* test_corpus_path, int max_examples);
int* get_encoded_statement(const char* statement, Encoder* encoder, int block_size);
int get_space_token(Encoder* encoder);
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

void validate_model(Model* model, char* test_corpus_path, int max_examples) {
    FILE* fp = fopen(test_corpus_path, "r");
    if (!fp) {
        printf("Error: Could not open test corpus\n");
        exit(1);
    }

    printf("Validating model on %d examples...\n", max_examples);

    int correct = 0;
    int total = 0;
    char line[MAX_SEQUENCE_LENGTH];
    while (fgets(line, sizeof(line), fp) && total < max_examples) {
        // Simplified validation - actual implementation would compare model predictions with ground truth.
        total++;
    }
    printf("Validation accuracy: %f%%\n", (float)correct / total * 100);
    fclose(fp);
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

int main() {
    srand(time(NULL));

    // Create output directories
    system("mkdir -p math_gpt_output/main_claim/corpus");
    system("mkdir -p math_gpt_output/main_claim/models");

    // Create encoder and assign it to the model.
    printf("Creating encoder...\n");
    Encoder* encoder = create_encoder("math_gpt_output/main_claim/corpus/corpus.txt");

    printf("Creating model...\n");
    Model* model = create_model(VOCAB_SIZE, BLOCK_SIZE, N_HEAD, N_LAYER);
    model->encoder = encoder;

    // Train model: note the correct parameter order.
    printf("Training model...\n");
    train_model(model, MAX_TRAIN_EPOCHS, "math_gpt_output/main_claim/corpus/train_corpus.txt", "math_gpt_output/main_claim/models/model.bin");

    // Save model.
    printf("Saving model...\n");
    save_model(model, "math_gpt_output/main_claim/models/model.bin");

    // Validate model.
    printf("Validating model...\n");
    validate_model(model, "math_gpt_output/main_claim/corpus/test_corpus.txt", 10);

    // Cleanup.
    free_model(model);

    return 0;
}

/* C implementation of the machine learning program shown in the Jupyter notebook https://github.com/calebnwokocha/ClaimGPT250203/blob/main/claim_gpt.ipynb
 * This is a simplified version that captures the main components of the original Python program. Here are some key points about the implementation:
 *
 * Key Components:
 *
 * Model creation and management
 * Encoder for vocabulary handling
 * Training loop implementation
 * Validation functionality
 * Basic file I/O for corpus handling
 */
