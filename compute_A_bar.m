%% ============================================================
%  Compute Average Defective Inventory Level (A_bar)
%  From cycle 4 to 9 using δ_i(t) = a / (g + exp(l2 * (t_dki + t)))
% ============================================================

clc; clear; close all;

% -------- Input Parameters ----------------------------------
numCycles = 9;                             % total available cycles
cyclesRange = 4:9;                         % range of interest (k=4 to 9)

tp = [8, 7, 6, 5, 5, 5, 4, 4, 3];          % production time per cycle
tc = [10, 9, 8, 7, 7, 6, 6, 5, 4];         % total cycle time (prod + idle)
tk_1 = 0;                                  % starting reference time (t0 = 0)

% Learning curve parameters for δ_i(t)
a  = 5;                                    % scaling parameter
g  = 1.2;                                  % offset parameter
l2 = -0.3;                                 % learning rate (negative for improvement)
t_dki = [0, 3, 6, 9, 12, 15, 18, 21, 24];  % offset time for each cycle

% ============================================================
A_bar = 0;  % initialize accumulated defective inventory

for k = cyclesRange
    % Time bounds for current cycle
    t_start = tk_1 + sum(tc(1:k-1));      % t_{k-1}
    t_end   = t_start + tp(k);            % t_{k-1} + tp_k

    % Defect rate δ_i(t)
    delta_fun = @(t) a ./ (g + exp(l2 .* (t_dki(k) + t)));

    % ----- (1) Integral term: defective accumulation during production -----
    integral_part = integral(@(t) delta_fun(t) .* (t - t_start), t_start, t_end);

    % ----- (2) Idle-time holding term -----
    delta_end = delta_fun(t_end);
    idle_term = delta_end * (tc(k) - tp(k));

    % ----- (3) Carry-over term for next production cycle -----
    if k < numCycles
        delta_next_start = a / (g + exp(l2 * (t_dki(k+1) + t_end)));
        carry_over = delta_next_start * tp(k+1);
    else
        carry_over = 0;  % no next cycle beyond last one
    end

    % ----- (4) Add all contributions -----
    A_bar = A_bar + integral_part + idle_term + carry_over;
end

% ============================================================
fprintf('Accumulated defective inventory (A_bar) from cycle 4–9 = %.6f\n', A_bar);

% Optional: compute average defective inventory per unit time
T_window = sum(tc(cyclesRange));
A_avg = A_bar / T_window;
fprintf('Average defective inventory per unit time (cycles 4–9) = %.6f\n', A_avg);
