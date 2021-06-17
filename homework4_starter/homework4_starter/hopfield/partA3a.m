load("data.mat");

w = learn_hopfield_net(data_10(1, :));

for i=2:6
    hop_settle_and_show(w, data_10(i, :), 2.^[5:2:13], figure(i));
end



rand_data = double(rand(10, 784) > 0.5);

for i=1:10
    w = learn_hopfield_net(data_10(i, :));
    hop_settle_and_show(w, rand_data(i, :), 2.^[5:2:13], figure(i));
end
