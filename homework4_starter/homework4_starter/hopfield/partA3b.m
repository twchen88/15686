load("data.mat");


%%% 1 %%%
rand_data = double(rand(10, 784) > 0.5);
n = [4, 7];
w = learn_hopfield_net(data_10(n, :));
hop_settle_and_show(w, rand_data(n, :), 2.^[5:2:13], figure(1));

%%% 2 %%%
n = [4, 7];
w = learn_hopfield_net(data_10(n, :));
hop_settle_and_show(w, data_10(n, :), 2.^[5:2:13], figure(2));

%%% 3 %%%
n = [4, 7];
noise_data = data_10(n, :);
lvl = [round(784/10), round(784/25), round(784/50)];
w = learn_hopfield_net(data_10(n, :));

for k = 1:3
    for i=1:length(n)
        for j=1:lvl(k):784
            x = noise_data(i, j);
            if x == 1
                noise_data(i, j) = 0;
            else
                noise_data(i, j) = 1;
            end
        end
    end

    hop_settle_and_show(w, noise_data, 2.^[5:2:13], figure(2+k));
end