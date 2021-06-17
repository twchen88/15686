load("data.mat");

mask = zeros(28,28);
mask(1:14, 1:14) = 1;
inverted_mask = ones(28,28);
inverted_mask(1:14, 1:14) = 0;


for i=1:10
    im = data_10(i, :);
    w = learn_hopfield_net(im);
    im = reshape(im, 28, 28);
    im = im.*mask;
    im = im + inverted_mask .* double(rand(size(im))>.5);
    
    hop_settle_and_show(w, reshape(im, 1, 28 * 28), 2.^[5:2:13], figure(i));
end