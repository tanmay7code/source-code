%% Compute t_d_k_i for the i-th cycle (single-cycle routine)
% This script computes:
%  - d_{i-1}(t_p), d_{i-1}(t_c)
%  - t^beta solving integral = beta_{i-1}+d_{i-1}(t_p)
%  - f^d_{i-1}
%  - beta_i
%  - t_d_k_i solving integral = beta_i

clc;
clear; 
close all;

%% ------------------ INPUTS (edit for your case) ------------------
% Defect logistic params (from Jaber et al.)
a  = 70.067;
g  = 819.76;
l2 = 0.7932;   % learning exponent on quality (use your l2; here example ~0.7932)

% Production rate params (example: Wright form used earlier)
lambda = 1000; % initial production rate
l1 = 0.322;    % learning exponent on production (example)

% Cycle & times
i = 3;                     % cycle number (only used for label)
t_p_prev = 4.5;            % t_{p_{i-1}} (previous production duration)
t_c_prev = 6.5;            % t_{c_{i-1}} (previous completion duration)
t_B = 5;                   % break time between cycles

% Remembered defective units from previous cycle (input)
beta_prev = 150;           % β_{i-1} (given)

% Tolerances / solver options
abs_tol = 1e-8;
max_td = 1e4;              % max search limit for time (adjust if needed)

%% ------------------ Define functions ------------------
% delta1(t) : logistic defect curve for cycle 1
delta1 = @(t) a ./ (g + exp(l2 .* t));

% p1(t) : production rate in first cycle (Wright-style)
% Use p1(t) = lambda * (1 + t).^l1  (as earlier). If you have different form, replace.
p1 = @(t) lambda .* (1 + t).^l1;

% integrand for defective production: delta1(t) * p1(t)
integrand = @(t) delta1(t) .* p1(t);

%% ------------------ Step 1: compute d_{i-1}(t_p) and d_{i-1}(t_c) ---------------
% If you actually have delta_{i-1} and p_{i-1} different, replace integrand accordingly.
d_tp = integral(integrand, 0, t_p_prev, 'AbsTol',abs_tol);
d_tc = integral(integrand, 0, t_c_prev, 'AbsTol',abs_tol);

%% ------------------ Step 2: find t^beta such that integral = beta_prev + d_tp ------------
target_sum = beta_prev + d_tp;
resid_fun_tbeta = @(t) integral(integrand, 0, t, 'AbsTol',abs_tol) - target_sum;

% Provide bracket for root finding: [0, upperBound]
% Find an upper bound where residual positive (integral increases with t)
upper = 1;
while resid_fun_tbeta(upper) < 0 && upper < max_td
    upper = upper * 2;
end
if upper >= max_td
    error('Could not bracket t^beta: increase max_td or check parameters.');
end
t_beta = fzero(resid_fun_tbeta, [0, upper]);

%% ------------------ Step 3: compute forgetting exponent f^d_{i-1} -----------------------
% c^d_{i-1} = tB / t_beta
c_d = t_B / t_beta;

% ensure positivity inside log: use small eps if needed
arg_log_num = d_tp + beta_prev;
if arg_log_num <= 0
    error('Argument of log in numerator non-positive: d_tp + beta_prev <= 0');
end
arg_log_den = c_d + 1;
if arg_log_den <= 0
    error('Argument of log in denominator non-positive: c_d + 1 <= 0');
end

f_d_prev = (l2 * (1 - l2) * log(arg_log_num)) / log(arg_log_den);

%% ------------------ Step 4: compute beta_i via LFCM ------------------------------
% beta_i = (beta_prev + d_tp)^((l2+f)/l2) * (beta_prev + d_tc)^(-f/l2)
beta_i = (beta_prev + d_tp)^((l2 + f_d_prev)/l2) * (beta_prev + d_tc)^(-f_d_prev/l2);

%% ------------------ Step 5: find t_d_k_i such that integral = beta_i -------------
resid_fun_tdki = @(t) integral(integrand, 0, t, 'AbsTol',abs_tol) - beta_i;

% bracket the root
upper = 1;
while resid_fun_tdki(upper) < 0 && upper < max_td
    upper = upper * 2;
end
if upper >= max_td
    error('Could not bracket t_d_k_i: increase max_td or check parameters.');
end
t_d_k_i = fzero(resid_fun_tdki, [0, upper]);

%% ------------------ Display results ------------------------------
fprintf('----- Results for cycle i = %d (single-cycle computation) -----\n', i);
fprintf('d_{i-1}(t_p)   = %.6f\n', d_tp);
fprintf('d_{i-1}(t_c)   = %.6f\n', d_tc);
fprintf('t^beta (theoretical time) = %.6f\n', t_beta);
fprintf('c^d_{i-1}      = %.6f\n', c_d);
fprintf('f^d_{i-1}      = %.6f\n', f_d_prev);
fprintf('beta_i         = %.6f\n', beta_i);
fprintf('t_d_k_i        = %.6f time units\n', t_d_k_i);

