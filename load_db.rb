#! ruby

require 'chunky_png'
require 'FileUtils'

require 'json'

def load_training(nb_samples)
    images = []
    labels = nil
    File.open('train-images.idx3-ubyte', 'rb') do |file| 
        _,nb,r,c = file.read(4*4).unpack("NNNN")
        nb_samples.times { |i| images << file.read(r*c).unpack("C" * r * c) }
    end
    File.open('train-labels.idx1-ubyte', 'rb') do |file| 
        _,nb = file.read(4*2).unpack("NN")
        labels = file.read(nb_samples).unpack("C" * nb_samples)
    end
    images.zip(labels)
end

def save_png(images, dir)
    FileUtils.mkdir_p(dir)
    images.each_with_index do |img, i|
        png = ChunkyPNG::Image.new(28,28, ChunkyPNG::Color::BLACK)
        28.times { |y| png.replace_row!(y, img[y*28,28].map{|p| ChunkyPNG::Color::grayscale(p) })}
        png.save("#{dir}/#{i}.png")
    end
end

# puts JSON.pretty_generate (load_training(10))
# load_training(10)