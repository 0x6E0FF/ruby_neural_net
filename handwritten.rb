#! ruby

require 'optparse'

require_relative 'load_db'
require_relative 'neurons'

def run(network, inputs, verbose=false)
    i = 0
    nb_success = 0
    inputs.each do |img, out, label|
        output = network.compute(img)
        found = output.each_with_index.max[1]
        puts "##{i} expected=#{label}: found=#{found}  #{output.map{|e|"%.2f" % e}.inspect}" if verbose
        nb_success+=1 if label == found 
        i += 1
    end
    nb_success
end

n = nil
verbose = false

OptionParser.new do |opts|
    opts.banner = "Usage: #{File.basename(__FILE__)} [options]"

    opts.on("-v", "--[no-]verbose", "Run verbosely") do |v|
        verbose = v
    end

    opts.on("-l", "--load FILE", "load network from FILE") do |file|
        if File.extname(file) == ".json"
            # load the ouput from python program
            data = JSON.parse(File.read(file))
            n = Network.new(data["sizes"])
            n.biases = data["biases"]
            n.biases = n.sizes[1..-1].map.with_index{ |nb_neurons, l| 
                # each layer
                nb_neurons.times.map { |n|
                    # each neuron 
                    data["biases"][l][n][0]
                }
            }
            n.weights = data["weights"]
        else
            n = Marshal.load(File.open(file, "rb") { |f| f.read() })
        end
    end
end.parse!

if n == nil
    n = Network.new([28*28,30,10]) unless n

    train_set = load_training(50000)
    test_sample = load_tests(10000)

    n.train(train_set, 1, 3.0, verbose) do |e, b|
    end
    puts "#{run(n, test_sample, false)} / #{test_sample.size}"
    d = Time.now
    File.open("#{n.sizes.join('-')}__#{d.day}-#{d.month}-#{d.hour}h#{d.min}", "wb") {|f| f.write(Marshal.dump(n)) }
else
    t1 = Time.now
    test_set = load_tests(-1)
    t2 = Time.now
    puts "load time = #{t2 - t1}"
    nb_success = 0
    nb_tests  = 0
    test_set.each_slice(100) do |batch|
        nb_tests += 100
        nb_success += run(n, batch, verbose)
        puts "#{nb_success.to_s.rjust(5,'0')} / #{nb_tests.to_s.rjust(5,'0')}"
    end
    puts "compute time = #{Time.now - t2}"
end

