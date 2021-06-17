function [  ] = test_settle_hopfield_net( )
%TEST_SETTLE_HOPFIELD_NET Summary of this function goes here
%   Detailed explanation goes here

d = [-1 1 -1 1 -1 1 -1];

random_data = rand(100, size(d,2));

w = learn_hopfield_net(d);
settled = settle_hopfield_net(w, random_data, 2^11);

for i=1:100
    if(~isequal(settled(i,:),d) && ~isequal(settled(i,:), -d))
        disp 'Test failed'
        return
    end
end

disp 'Test passed'

end

