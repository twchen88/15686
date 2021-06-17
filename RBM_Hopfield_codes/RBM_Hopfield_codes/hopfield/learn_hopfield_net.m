function [ w ] = learn_hopfield_net(data)
% [w] = learn_hopfield_net(data) uses the outer product rule to learn a
% weight matrix of a Hopfield network
%
% INPUTS:
%       data ...... : (n by v) data matrix. n = data points, v = number of
%                       visible units
%
% OUTPUTS:
%       w ......... : (v by v) weight matrix. v = number of visible units

% Data will be 0s and 1s. Need to convert to -1 and 1 for hopfield.
data = convert_0_to_neg1(data);



[n,p] = size(data);
w = zeros(p,p);

%% FOR STUDENT: Compute w using the outer product rule
