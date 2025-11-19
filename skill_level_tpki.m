clc; clear; close all;

% ===============================================================
% 🔹 INPUT PARAMETERS for the i-th cycle
% ===============================================================
i = 3;                % current cycle number (example)
lambda = 1000;        % initial production rate
LR = 0.8;             % learning rate (e.g., 0.9 slow, 0.8 moderate, 0.7 fast)
l1 = -log(LR)/log(2); % learning exponent

tB = 5;               % break time between cycles
tp_prev = 4.5;        % production time in (i-1)-th cycle
tc_prev = 6.5;        % completion time in (i-1)-th cycle
alpha_prev = 2000;    % equivalent experience before (i-1)-th cycle

% ===============================================================
% 🔹 Step 1: Production rate function p1(t)
% ===============================================================
p1 = @(t) lambda * (1 + t).^l1;

% ===============================================================
% 🔹 Step 2: Compute cumulative production in (i-1)-th cycle
% ===============================================================
q_tp = integral(@(t) p1(t), 0, tp_prev);   % q_{i-1}(t_{p_{i-1}})
q_tc = integral(@(t) p1(t), 0, tc_prev);   % q_{i-1}(t_{c_{i-1}})

% ===============================================================
% 🔹 Step 3: Compute theoretical production time t_p_alpha_{i-1}
% ===============================================================
tp_alpha = ((1 + l1)/lambda * (alpha_prev + q_tp) + 1)^(1/(1 + l1)) - 1;

% ===============================================================
% 🔹 Step 4: Compute forgetting coefficient f^{p}_{i-1}
% ===============================================================
cp = tB / tp_alpha;
fp_prev = (l1 * (1 - l1) * log(q_tp + alpha_prev)) / log(cp + 1);

% ===============================================================
% 🔹 Step 5: Compute worker experience α_i
% ===============================================================
alpha_i = (alpha_prev + q_tp)^((l1 + fp_prev) / l1) *(alpha_prev + q_tc)^(-fp_prev / l1);

% ===============================================================
% 🔹 Step 6: Compute equivalent learning time t^{p}_{k_i}
% ===============================================================
tpki = ((1 + l1)/lambda * alpha_i + 1)^(1/(1 + l1)) - 1;

% ===============================================================
% 🔹 Step 7: Display results
% ===============================================================
fprintf('Cycle i = %d\n', i);
fprintf('Learning exponent (l1)        = %.4f\n', l1);
fprintf('Cumulative production q_tp    = %.4f\n', q_tp);
fprintf('Cumulative production q_tc    = %.4f\n', q_tc);
fprintf('Forgetting exponent f_p_prev  = %.4f\n', fp_prev);
fprintf('Worker experience alpha_i     = %.4f\n', alpha_i);
fprintf('Equivalent time t_p_k_i       = %.4f\n', tpki);
