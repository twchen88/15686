Vrest = -70; % mV
I_0 = [-0.1, 0.1, 0.2, 0.3];
T = -20:1:160; % from:incre:end in msec
R = 100;
C = 100;
toff = 100;
tau = 10;
 

Vm = [];
for I = I_0
    for i = 1:length(T)
        t = T(i);
        if t < 0
            V = Vrest;
        elseif t >= toff
            V = R * I * exp(-(t-toff)/tau) + Vrest;
        else     
            V = R * I * (1 - exp(-t/tau)) + Vrest;
        end
        Vm(end+1) = V;
    end
    plot(T, Vm);
    Vm = [];
    hold on
end
legend('-0.1', '0.1', '0.2', '0.3')
title('Problem 1 Part D: Membrane potential with injected current (closed equation)');
xlabel('Time'); ylabel('Voltage');
hold off