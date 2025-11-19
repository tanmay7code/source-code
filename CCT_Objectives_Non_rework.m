function f = CCT_Objectives_Non_rework(x)
% ============================================================
%   Sustainable Bi-Objective EPQ Model under Carbon Cap-and-Trade (CCT)
%   Decision variables: x = [tp_i, mu, G]
% ============================================================

% ===================== Extract Decision Variables =====================
tp_i = x(1); mu = x(2); G = x(3);

% ===================== Parameters =====================
D1=20000; D2=5000; s=80; cm=5; gamma=10; s_dash=2; h=6; lambda=50000; l1=0.074; l2=0.074; a=70.06; g=819.76; xi=0.2; a_e=3*10^(-7); b_e=0.0012; c_e=1.4; L=700; Ev_e=2.6; w_e=5; u=50; C=2000; tpki=0; tdki=0;

% ===================== Demand Function =====================
D=D1+mu*D2;

% ===================== q_i(t_p_i) =====================
q_i=(lambda/(1+l1))*((1+tpki+tp_i)^(1+l1)-(1+tpki)^(1+l1));

% ===================== d_i(t_p_i) =====================
d_fun=@(t)(a*lambda*(1+tpki+t).^l1)./(g+exp(l2*(tdki+t))); d_i=integral(d_fun,0,tp_i);

% ===================== Double Integral (Simplified) =====================
double_integral=integral(@(tau)(tp_i-tau).*(a*lambda*(1+tpki+tau).^l1)./(g+exp(l2*(tdki+tau))),0,tp_i);

% ===================== I_holding =====================
I_holding=(lambda/(1+l1))*((1/(2+l1))*((1+tpki+tp_i)^(2+l1)-(1+tpki)^(2+l1))-(1+tpki)^(1+l1)*tp_i)-double_integral-D*(tp_i^2)/2+(1/(2*D))*((lambda/(1+l1))*((1+tpki+tp_i)^(1+l1)-(1+tpki)^(1+l1))-d_i-D*tp_i)^2;

% ===================== π_i (Economic Objective) =====================
pi_i=(D*(s+q_i*cm+tp_i*gamma+q_i*s_dash+I_holding*h+G*mu^2))/(q_i-d_i);

% ===================== EP, ET, EH =====================
EP=(a_e*lambda^2/(1+2*l1))*((1+tpki+tp_i)^(1+2*l1)-(1+tpki)^(1+2*l1))-(b_e*lambda/(1+l1))*((1+tpki+tp_i)^(1+l1)-(1+tpki)^(1+l1))+c_e*tp_i; ET=2*L*Ev_e; EH=I_holding*w_e;

% ===================== ψ_i (Environmental Objective) =====================
psi_i=(D*(EP+ET+EH)*(1 - xi*(1 - exp(-mu*G))))/(q_i-d_i);

% ===================== CCT Policy Adjustments =====================
penalty=max(psi_i-C,0); reward=max(C-psi_i,0);

% ===================== Objective Functions =====================
f1_i=pi_i+u*penalty-u*reward; f2_i=psi_i; f=[f1_i,f2_i];
end
