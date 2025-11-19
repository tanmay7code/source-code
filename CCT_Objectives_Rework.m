function f = CCT_Objectives_Rework(x)
% ============================================================
%   Sustainable Bi-Objective EPQ Model under Carbon Cap-and-Trade (CCT)
%   with Rework Cycle and Extended Emission Term
%   Decision variables: x = [tp_i, mu, G]
% ============================================================

% ===================== Extract Decision Variables =====================
tp_i=x(1); mu=x(2); G=x(3);

% ===================== Parameters =====================
 
D1=20000; D2=5000; s=80; cm=5; gamma=10; s_dash=2; h=6; lambda=50000; l1=0.074; l2=0.074; a=70.06; g=819.76; xi=0.2; a_e=3*10^(-7); b_e=0.0012; c_e=1.4; L=700; Ev_e=2.6; w_e=5; w_e_bar=3.5; u=50; C=2000; tpki=0; tdki=0; c_r=3; epsilon=0.1; t_r=0.05; R=30000;

% ===================== Demand Function =====================
D=D1+mu*D2;

% ===================== Example sub-cycle times (for A_bar calculation) =====================
tp=[tp_i,tp_i*0.9,tp_i*1.1]; tc=[30,35,40]; tk_1=0;

% ===================== q_i(t_p_i) =====================
q_i=(lambda/(1+l1))*((1+tpki+tp_i)^(1+l1)-(1+tpki)^(1+l1));

% ===================== d_i(t_p_i) =====================
d_fun=@(t)(a*lambda*(1+tpki+t).^l1)./(g+exp(l2*(tdki+t))); d_i=integral(d_fun,0,tp_i);

% ===================== Double Integral (Simplified) =====================
double_integral=integral(@(tau)(tp_i-tau).*(a*lambda*(1+tpki+tau).^l1)./(g+exp(l2*(tdki+tau))),0,tp_i);

% ===================== Multi-Cycle \bar{A} Calculation =====================
numCycles=length(tp); d_vals=zeros(1,numCycles);
for k=1:numCycles
    tdki_k=tdki+5*(k-1);
    d_fun_k=@(t)(a*lambda*(1+tpki+t).^l1)./(g+exp(l2*(tdki_k+t)));
    d_vals(k)=integral(d_fun_k,0,tp(k));
end

% ===================== r_i (Reworked Items) =====================
r_i=(1-epsilon)*sum(d_vals);

% ===================== \overline{I}_holding =====================
I_holding_bar=(lambda/(1+l1))*((1/(2+l1))*((1+tpki+tp_i)^(2+l1)-(1+tpki)^(2+l1))-(1+tpki)^(1+l1)*tp_i)-double_integral-D*(tp_i^2)/2-(1/2)*R*(t_r^2)+R*t_r*tp_i+(1/(2*D))*((lambda/(1+l1))*((1+tpki+tp_i)^(1+l1)-(1+tpki)^(1+l1))+r_i-d_i-D*tp_i)^2;

% ===================== Compute A_bar =====================
A_bar=0;
for k=1:numCycles
    t_start=tk_1+sum(tc(1:k-1)); t_end=t_start+tp(k);
    if k==1
        A_bar=A_bar+integral(@(t)d_vals(k).*(t-tk_1),t_start,t_end)+d_vals(k)*(tc(k)-tp(k))+d_vals(k)*tp(min(k+1,numCycles));
    else
        prev_sum=sum(d_vals(1:k-1));
        A_bar=A_bar+integral(@(t)(prev_sum+d_vals(k).*(t-tk_1-sum(tc(1:k-1)))),t_start,t_end);
        A_bar=A_bar+(prev_sum+d_vals(k))*(tc(min(k+1,numCycles))-tp(min(k+1,numCycles)));
    end
end

% ===================== π_i (Economic Objective for Rework) =====================
pi_i=(D*(s+q_i*cm+tp_i*gamma+q_i*s_dash+G*mu^2+r_i*c_r+A_bar*h+I_holding_bar*h))/(q_i+r_i-d_i);

% ===================== EP, ET, and Extended EH =====================
EP=(a_e*lambda^2/(1+2*l1))*((1+tpki+tp_i)^(1+2*l1)-(1+tpki)^(1+2*l1))-(b_e*lambda/(1+l1))*((1+tpki+tp_i)^(1+l1)-(1+tpki)^(1+l1))+c_e*tp_i; ET=2*L*Ev_e; EH_bar=I_holding_bar*w_e+A_bar*w_e_bar;

% ===================== Rework Emission Term \overline{ER} =====================
t_ri=r_i/R; ER_bar=(a_e*R^2-b_e*R+c_e)*t_ri;

% ===================== ψ_i (Environmental Objective with Extended EH) =====================
psi_i=(D*(EP+ET+EH_bar+ER_bar)*(1 - xi*(1 - exp(-mu*G))))/(q_i+r_i-d_i);

% ===================== CCT Policy Adjustments =====================
penalty=max(psi_i-C,0); reward=max(C-psi_i,0);

% ===================== Objective Functions =====================
f1_i=pi_i+u*penalty-u*reward; f2_i=psi_i; f=[f1_i,f2_i];
end
