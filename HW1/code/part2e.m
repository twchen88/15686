%% Simulation of (leaky) integrate-and-fire neuron
 clear; clf;

%% parameters of the model
 dt=0.1;       % integration time step [ms]
 tau=10;       % time constant [ms]
 E_L=-65;      % resting potential [mV]
 theta=-55;    % firing threshold [mV]
 RI_ext=14;     % constant external input [mA/Ohm]
 sigma = 1;
 
 % freq and current
 current = 0:0.5:15;
 freq = zeros(1, 31);
 
%% Integration with Euler method
for RI_ext = 0:0.5:15
 t_step=0; v=E_L;
 for t=0:dt:250;
     t_step=t_step+1;
     s=v>theta;
     v=s*E_L+(1-s).*(v-dt./tau*((v-E_L)-RI_ext + sigma * randn));
     v_rec(t_step)=v;
     t_rec(t_step)=t;
     s_rec(t_step)=s;
 end
 freq(RI_ext/0.5+1) = sum(s_rec);
end

%% Plotting results
%  subplot('position',[0.13 0.13 1-0.26 0.6])
%    plot(t_rec,v_rec);
%    hold on; plot([0 100],[-55 -55],'--');
%    axis([0 100 -66 -54]);
%    xlabel('Time [ms]'); ylabel('v [mV]')
% 
%  subplot('position',[0.13 0.8 1-0.26 0.1])
%    plot(t_rec,s_rec,'.','markersize',20);
%    axis([0 100 0.5 1.5]); 
%    set(gca,'xtick',[],'ytick',[])
%    ylabel('Spikes')

plot(current, freq);
xlabel('Current'); ylabel('Spiking Rate [Hz]')
