#! ruby

require 'matrix'
require 'json'

def debug(s)
    puts JSON.pretty_generate s
    puts "***********"
end

def sigmoid(v)
    1.0 / (1.0 + Math.exp(-v))
end

def sigmoid_prime(v)
    sigmoid(v) * (1.0 - sigmoid(v))
end

def cost(output, expected)
    Vector[*output.zip(expected).map{|o,e| e - o }].magnitude ** 2
end

class Network
    attr_accessor :sizes, :biases, :weights
    def initialize(sizes)
        @sizes = sizes
        @nb_layers = sizes.size
        @biases = sizes[1..-1].map{ |nb_neurons| 
            # each layer
            nb_neurons.times.map { 
                # each neuron 
                rand(-1.0..1.0) 
            }
        }
        @weights = sizes[0..-2].zip(sizes[1..-1]).map { |nb_inputs, nb_neurons|
            # each layer
            nb_neurons.times.map {
                # each neuron
                nb_inputs.times.map {
                    # each connection
                    rand(-1.0..1.0)
                }
            }
        }
    end

    def compute(inputs)
        @activations = [inputs]
        @weighted_sums = []
        @biases.zip(@weights).each do |b, w|
            # each layer 
            #    - b[nb_neuron]
            #    - w[nb_neuron][nb_inputs]
            @weighted_sums << w.zip(b).map { |wn, bn|
                # each neuron
                Vector[*wn].dot(Vector[*@activations[-1]]) + bn
            }
            @activations << @weighted_sums[-1].map{ |v| sigmoid(v) }
        end
        @activations[-1]
    end

    def backpropagation(input, expected_output)

        # biases gradients array 
        biases_grad = Array.new(@biases.size)
        # weights gradients array
        weights_grad = Array.new(@weights.size)

        # compute output layer 
        output = compute(input)

        # compute gradient of cost function ∇aC
        cost_gradient = output.zip(expected_output).map{ |o, e| o - e }

        # compute error on output layer
        # -> δL=∇aC⊙σ′(zL)
        error = cost_gradient.zip(@weighted_sums[-1]).map{ |c, z| c * sigmoid_prime(z) }

        biases_grad[-1] = error
        weights_grad[-1] = error.map { |e| @activations[-2].map { |a| a * e } }

        # propagate error backward on each layer
        (2...@nb_layers).map{|l| -l }.each do |layer_i|
            error = @activations[layer_i].size.times.map { |n| # each neuron of current layer

                # compute row n of new error vector (of size the number of neurons of current layer)
                #   -> δl=((wl+1)Tδl+1) ⊙ σ′(zl)
                error.map.with_index {|e,i| 
                    @weights[layer_i+1][i][n] * e 
                }.reduce(:+) * sigmoid_prime(@weighted_sums[layer_i][n])
            }
            biases_grad[layer_i] = error
            weights_grad[layer_i] = error.map { |e| @activations[layer_i-1].map { |a| a * e } }
        end
        [biases_grad, weights_grad]
    end

    def each_bias_index()
        @biases.size.times do |l| 
            @biases[l].size.times do |n|
                yield [l,n]
            end
        end
    end

    def each_weight_index()
        @weights.size.times do |l| 
            @weights[l].size.times do |n|
                @weights[l][n].size.times do |w|
                    yield [l,n,w]
                end
            end
        end
    end

    def train_batch(batch, learning_rate)
        sum_grad_b = biases.map{ |l| Array.new(l.size, 0.0) }
        sum_grad_w = weights.map{ |l| l.map {|n| Array.new(n.size, 0.0 )} }
        batch.each do |input, expected_output|
            grad_b, grad_w = backpropagation(input, expected_output)
            each_bias_index() { |l,n| sum_grad_b[l][n] += grad_b[l][n] }
            each_weight_index() { |l,n,w| sum_grad_w[l][n][w] += grad_w[l][n][w] }
        end
        each_bias_index() do |l,n| 
            @biases[l][n] = @biases[l][n] - (sum_grad_b[l][n] * learning_rate) / batch.size 
        end
        each_weight_index() do |l,n,w| 
            @weights[l][n][w] = @weights[l][n][w] - (sum_grad_w[l][n][w] * learning_rate) / batch.size 
        end
    end

    def train(train_set, nb_epoch, learning_rate, verbose = false)
        nb_epoch.times do |e|
            train_set.shuffle.each_slice(10).with_index do |batch, i|
                puts "training batch #{i.to_s.rjust(2,'0')} / #{train_set.size / 10}" if verbose
                train_batch(batch, learning_rate);
                yield [e, i]
            end
        end
    end

    def to_s
        JSON.pretty_generate({"biases" => @biases, "weights" => @weights})
    end
end
