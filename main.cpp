#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

std::random_device dev;
std::mt19937 rng(dev());
std::uniform_real_distribution<double> real_rand(-1.0, 1.0);

const double EulerConstant = std::exp(1.0);

class Layer {
    int input_size, output_size;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::vector<double> input_layer;
    std::vector<double> output_layer;

    public:
    Layer() : Layer(1, 1) {}

    Layer(int input_size, int output_size) {
        this->input_size = input_size;
        this->output_size = output_size;

        for(int i = 0; i < input_size; i++) {
            this->weights.push_back(std::vector<double>());

            for(int j = 0; j < output_size; j++) {
                this->weights[i].push_back(real_rand(rng));
            }
        }

        for(int i = 0; i < output_size; i++) {
            this->bias.push_back(real_rand(rng));
        }

        this->input_layer = std::vector<double>(input_size, 0);
        this->output_layer = std::vector<double>(output_size, 0);
    }

    Layer(int input_size, int output_size, std::vector<std::vector<double>> weights, std::vector<double> bias) {
        if(input_size < 1 || output_size < 1) {
            throw std::invalid_argument("Invalid size");
        }

        if(weights.size() != input_size) {
            throw std::invalid_argument("Incompatible input size with weights");
        }

        if(weights[0].size() != output_size) {
            throw std::invalid_argument("Incompatible output size with weights");
        }
        
        if(bias.size() != output_size) {
            throw std::invalid_argument("Incompatible output size with bias");
        }

        this->input_size = input_size;
        this->output_size = output_size;
        this->weights = weights;
        this->bias = bias;
        this->input_layer = std::vector<double>(input_size, 0);
        this->output_layer = std::vector<double>(output_size, 0);
    }

    double activation(double x) {
        return 1 / (1 + pow(EulerConstant, -x));
    }

    std::vector<double> cost(std::vector<double> input_layer) {
        if(input_layer.size() != this->input_size) {
            throw std::invalid_argument("aaaIncompatible input layer size");
        }

        this->input_layer = input_layer;
        this->output_layer = std::vector<double>(output_size, 0);

        for(int j = 0;j < output_size;j++) {
            output_layer[j] = 0;

            for(int i = 0;i < input_size;i++) {
                output_layer[j] += this->weights[i][j] * input_layer[i];
            }
            output_layer[j] += this->bias[j];
            output_layer[j] = activation(output_layer[j]);
        }

        return output_layer;
    }

    std::vector<double> getBias() {
        return this->bias;
    }

    void setBias(std::vector<double> bias) {
        this->bias = bias;
    }

    std::vector<std::vector<double>> getWeights() {
        return this->weights;
    }

    void setWeights(std::vector<std::vector<double>> weights) {
        this->weights = weights;
    }

    std::vector<double> getOutput() {
        return this->output_layer;
    }

    std::vector<double> getInput() {
        return this->input_layer;
    }

    int getInputSize() {
        return this->input_size;
    }

    int getOutputSize() {
        return this->output_size;
    }

    void print_weights() {
        for(auto row : weights) {
            for(auto e : row) {
                std::cout << e << " "; 
            }
            std::cout << std::endl;
        }
    }
};

class NeuralNetwork {
    int input_layer_size, hidden_layer_size, output_layer_size, layer_number;
    double learning_rate;
    std::vector<double> inputs, expected_outputs;
    std::vector<Layer> layers;

    public:
    NeuralNetwork(int input_layer_size, int output_layer_size, int layer_number, double learning_rate, std::vector<double>& inputs, std::vector<double>& expected_outputs) {
        if(layer_number != 2) {
            throw std::invalid_argument("Invalid layer number");
        }

        if(inputs.size() != input_layer_size) {
            throw std::invalid_argument("Incompatible input layer size");
        }

        if(expected_outputs.size() != output_layer_size) {
            throw std::invalid_argument("Incompatible output layer size");
        }

        this->input_layer_size = input_layer_size;
        this->hidden_layer_size = 0;
        this->output_layer_size = output_layer_size;
        this->layer_number = layer_number;
        this->learning_rate = learning_rate;

        this->inputs = inputs;
        this->expected_outputs = expected_outputs;

        this->layers.push_back(Layer(input_layer_size, output_layer_size));
        this->layers.push_back(Layer(output_layer_size, output_layer_size));
    }

    NeuralNetwork(int input_layer_size, int hidden_layer_size, int output_layer_size, int layer_number, double learning_rate, std::vector<double>& inputs, std::vector<double>& expected_outputs) {
        if(layer_number < 3) {
            throw std::invalid_argument("Invalid layer number");
        }

        if(inputs.size() != input_layer_size) {
            throw std::invalid_argument("Incompatible input layer size");
        }

        if(expected_outputs.size() != output_layer_size) {
            throw std::invalid_argument("Incompatible output layer size");
        }

        this->input_layer_size = input_layer_size;
        this->hidden_layer_size = hidden_layer_size;
        this->output_layer_size = output_layer_size;
        this->layer_number = layer_number;
        this->learning_rate = learning_rate;

        this->inputs = inputs;
        this->expected_outputs = expected_outputs;

        this->layers.push_back(Layer(input_layer_size, hidden_layer_size));
        for(int i = 0;i < layer_number-2;i++) {
            this->layers.push_back(Layer(hidden_layer_size, hidden_layer_size));
        }
        this->layers.push_back(Layer(hidden_layer_size, output_layer_size));
    }

    void setLayers(std::vector<Layer>& layers){
        this->layers = layers;
    }

    std::vector<double> errors(std::vector<double>& output_layer) {
        if(output_layer.size() != this->output_layer_size) {
            throw std::invalid_argument("Incompatible layer size");
        }

        std::vector<double> error;
        for(int i = 0;i < output_layer_size;i++) {
            double e = (output_layer[i] - this->expected_outputs[i]);
            error.push_back(0.5 * e * e);
        }

        return error;
    }

    double error_total(std::vector<double>& output_layer) {
        if(output_layer.size() != this->output_layer_size) {
            throw std::invalid_argument("Incompatible layer size");
        }

        double error = 0;
        for(int i = 0;i < output_layer_size;i++) {
            double e = (output_layer[i] - this->expected_outputs[i]);
            error += 0.5 * e * e;
        }

        return error;
    }

    std::vector<double> forward_propagation() {
        std::vector<double> cost_result = this->inputs;

        for(int i = 0;i < this->layer_number;i++) {
            cost_result = this->layers[i].cost(cost_result);
        }
        
        return cost_result;
    }

    void backpropagation(std::vector<double>& predicted) {
        int l = this->layer_number-1;
        int input_size = this->layers[l].getInputSize();
        int output_size = this->layers[l].getOutputSize();
        std::vector<double> inputs = this->layers[l].getInput();
        std::vector<double> outputs = this->layers[l].getOutput();
        std::vector<double> bias = this->layers[l].getBias();
        std::vector<std::vector<double>> weights = this->layers[l].getWeights();
        std::vector<double> gradient = std::vector<double>(output_size, 0);
        std::vector<double> new_gradient = std::vector<double>(input_size, 0);
        for(int i = 0;i < output_size;i++) {
            gradient[i] = (predicted[i] - this->expected_outputs[i]) * predicted[i] * (1 - predicted[i]);
        }

        for(int i = 0;i < input_size;i++) {
            double w = 1;
            for(int j = 0;j < output_size;j++) {
                w *= weights[i][j] * gradient[j];
                double slope = gradient[j] * inputs[i];
                // std::cout << i << " " << j << " " << weights[i][j] << " " << gradient[j] << " " << slope << " ";
                weights[i][j] = weights[i][j] - (this->learning_rate * slope);
                // std::cout << weights[i][j] << " " << (this->learning_rate * slope) << std::endl;
                bias[j] = bias[j] - (this->learning_rate * slope);
            }
            // std::cout << w << std::endl;
            new_gradient[i] = w;
            // std::cout << new_gradient[i] << std::endl;
        }

        this->layers[l].setWeights(weights);
        l--;

        for(;l >= 0;l--) {
            gradient = new_gradient;
            input_size = this->layers[l].getInputSize();
            output_size = this->layers[l].getOutputSize();
            inputs = this->layers[l].getInput();
            outputs = this->layers[l].getOutput();
            new_gradient = std::vector<double>(input_size, 0);
            weights = this->layers[l].getWeights();
            bias = this->layers[l].getBias();
        
            for(int i = 0;i < input_size;i++) {
                double w = 1;
                for(int j = 0;j < output_size;j++) {
                    w *= weights[i][j] * gradient[j];
                    double slope = inputs[i] * outputs[j] * (1 - outputs[j]) * gradient[j];
                    weights[i][j] = weights[i][j] - (this->learning_rate * slope);
                    bias[j] = bias[j] - (this->learning_rate * slope);
                }
                new_gradient[i] = w;
            }

            this->layers[l].setWeights(weights);
        }
    }

    std::vector<Layer> getLayers() {
        return this->layers;
    }

    void printLayers() {
        for(auto l : this->layers) {
            std::cout << l.getInputSize() << " " << l.getOutputSize() << std::endl;
            l.print_weights();
            std::cout << std::endl;
        }
    }

    void train(int epochs) {
        std::vector<double> result = forward_propagation();
        std::cout << error_total(result) << std::endl << std::endl;
        printLayers();
        std::cout << std::endl << std::endl;
        for(auto e : result) {
            std::cout << e << std::endl;
        }
        std::cout << std::endl << std::endl;
        backpropagation(result);

        for(int i = 1; i <= epochs;i++) {
            result = forward_propagation();
            backpropagation(result);
        }

        std::cout << error_total(result) << std::endl << std::endl;
        printLayers();
        std::cout << std::endl << std::endl;

        for(auto e : result) {
            std::cout << e << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
};

int main() {
    std::cout << "Hello World!\n";
    std::cout << std::fixed << std::setprecision(20);

    // X: 0.05
    // Y: 0.10

    // XA: 0.15   XB: 0.25  
    // YA: 0.2    YB: 0.3

    // AC: 0.4    AD: 0.5
    // BC: 0.45   BC: 0.55

    // Bias
    // First Layer 0.35
    // Second Layer 0.6
    // Layer 1
    std::vector<std::vector<double>> weights_layer1 = std::vector<std::vector<double>>({
        std::vector<double>({0.15, 0.25}),
        std::vector<double>({0.2, 0.3}),
    });
    std::vector<double> bias_layer1 = std::vector<double>({
        0.35, 0.35
    });

    Layer layer1 = Layer(2, 2, weights_layer1, bias_layer1);

    //Layer 2
    std::vector<std::vector<double>> weights_layer2 = std::vector<std::vector<double>>({
        std::vector<double>({0.4, 0.5}),
        std::vector<double>({0.45, 0.55}),
    });
    std::vector<double> bias_layer2 = std::vector<double>({
        0.6, 0.6
    });

    Layer layer2 = Layer(2, 2, weights_layer2, bias_layer2);

    std::vector<double> input({
        0.05, 0.10
    });
    std::vector<double> expected_output({
        0.01, 0.99
    });

    NeuralNetwork nn = NeuralNetwork(2, 2, 2, 0.5, input, expected_output);
    std::vector<Layer> l = std::vector<Layer>({
        layer1,
        layer2
    });

    nn.setLayers(l);

    nn.train(10000);

    return 0;
}