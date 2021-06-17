function [m] = convert_neg1_to_0( m )
%CONVERT_NEG1_TO_0 converts -1 values in m to 0
m(m==-1) = 0;

end

