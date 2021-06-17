%Load images

a = [];
a = [a extract_patches(imread('i3.11.jpg'),8,500)];
a = [a extract_patches(imread('i3.12.jpg'),8,500)];
a = [a extract_patches(imread('i3.13.jpg'),8,500)];

b = [];
b = [b extract_patches(imread('i3.14.jpg'),8,500)];
b = [b extract_patches(imread('i3.15.jpg'),8,500)];
b = [b extract_patches(imread('i3.16.jpg'),8,500)];

c = [];
c = [c extract_patches(imread('i3.17.jpg'),8,500)];
c = [c extract_patches(imread('i3.18.jpg'),8,500)];
c = [c extract_patches(imread('i3.19.jpg'),8,500)];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%get covariance matrix (transpose matrix such that cov produces correct result)
C_a = cov(transpose(a), 1);
%singular value decomposition on covariance matrix
[U_a,S_a,V_a] = svd(C_a);

figure
e_values = sum(S_a); % 1x64 eigen values
total_var_a = sum(e_values); % total variance = trace(S)
var90_a = total_var_a * 0.9;
var99_a = total_var_a * 0.99;
v = 0;
varc = 0;
while v < var90_a
    varc = varc + 1;
    v = v + e_values(varc);
end
var_90c_a = varc;
while v < var99_a
    varc = varc + 1;
    v = v+ e_values(varc);
end
var_99c_a = varc;
gx = [0:63];
plot(gx, e_values);
title("Eigenvalues: Patterns");
xlabel("rank");
ylabel("eigenvalue");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%get covariance matrix (transpose matrix such that cov produces correct result)
C_b = cov(transpose(b), 1);
%singular value decomposition on covariance matrix
[U_b,S_b,V_b] = svd(C_b);

figure
e_values = sum(S_b); % 1x64 eigen values
total_var_b = sum(e_values); % total variance = trace(S)
var90_b = total_var_b * 0.9;
var99_b = total_var_b * 0.99;
v = 0;
varc = 0;
while v < var90_b
    varc = varc + 1;
    v = v + e_values(varc);
end
var_90c_b = varc;
while v < var99_b
    varc = varc + 1;
    v = v+ e_values(varc);
end
var_99c_b = varc;
gx = [0:63];
plot(gx, e_values);
title("Eigenvalues: People");
xlabel("rank");
ylabel("eigenvalue");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%get covariance matrix (transpose matrix such that cov produces correct result)
C_c = cov(transpose(c), 1);
%singular value decomposition on covariance matrix
[U_c,S_c,V_c] = svd(C_c);

figure
e_values = sum(S_c); % 1x64 eigen values
total_var_c = sum(e_values); % total variance = trace(S)
var90_c = total_var_c * 0.9;
var99_c = total_var_c * 0.99;
v = 0;
varc = 0;
while v < var90_c
    varc = varc + 1;
    v = v + e_values(varc);
end
var_90c_c = varc;
while v < var99_c
    varc = varc + 1;
    v = v+ e_values(varc);
end
var_99c_c = varc;
gx = [0:63];
plot(gx, e_values);
title("Eigenvalues: Abstract Art");
xlabel("rank");
ylabel("eigenvalue");