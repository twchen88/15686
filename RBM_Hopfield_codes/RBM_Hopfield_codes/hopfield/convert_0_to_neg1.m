function [ m] = convert_0_to_neg1( m)
%CONVERT_0_TO_NEG1 converts the zero values in m to -1

m(m==0) = -1;
end

