clc; clear; close all;

%% ================= SPEA2 PARAMETERS =================
N = 100;            % Population size
N_archive = 100;    % Archive size
MaxGen = 100;       % Number of generations
nVar = 3;           % Decision variables: [tp_i, mu, G]
VarMin = [1, 0.01, 0.1];    % Lower bounds
VarMax = [200, 2, 10];      % Upper bounds

pc = 0.8;           % Crossover probability
pm = 0.2;           % Mutation probability
etaC = 15;          % Crossover index
etaM = 20;          % Mutation index

%% ================ INITIALIZATION ====================
pop = repmat(struct('Position', [], 'Cost', []), N, 1);
for i = 1:N
    pop(i).Position = VarMin + rand(1, nVar) .* (VarMax - VarMin);
    pop(i).Cost = CCT_Objectives_Rework(pop(i).Position);
end
archive = [];

%% =================== MAIN LOOP ======================
for gen = 1:MaxGen
    % Combine population and archive
    union_pop = [pop; archive];
    
    % Calculate fitness
    fitness = SPEA2_Fitness(union_pop);
    
    % Environmental selection
    [~, idx] = sort(fitness);
    archive = union_pop(idx(1:min(N_archive, numel(union_pop))));
    
    % Truncate archive if needed
    if numel(archive) > N_archive
        archive = archive(1:N_archive);
    end
    
    % Binary tournament selection
    mating_pool = TournamentSelection(archive, N);
    
    % Crossover + Mutation
    new_pop = repmat(struct('Position', [], 'Cost', []), N, 1);
    for i = 1:2:N
        p1 = mating_pool(randi([1, N]));
        p2 = mating_pool(randi([1, N]));
        
        [child1, child2] = SBX_Crossover(p1.Position, p2.Position, VarMin, VarMax, etaC, pc);
        
        child1 = PolynomialMutation(child1, VarMin, VarMax, etaM, pm);
        child2 = PolynomialMutation(child2, VarMin, VarMax, etaM, pm);
        
        new_pop(i).Position = child1;
        new_pop(i).Cost = CCT_Objectives_Rework(child1);
        new_pop(i+1).Position = child2;
        new_pop(i+1).Cost = CCT_Objectives_Rework(child2);
    end
    pop = new_pop;
    
    % ======== Display Progress ========
    Costs = reshape([archive.Cost], 2, [])';
    plot(Costs(:,1), Costs(:,2), 'bo', 'MarkerFaceColor', 'r');
    title(['Generation ' num2str(gen)]);
    xlabel('Economic Objective (f1)');
    ylabel('Environmental Objective (f2)');
    grid on;
    drawnow;
end

%% ================== FINAL PARETO FRONT ====================
ParetoFront = reshape([archive.Cost], 2, [])';
figure;
plot(ParetoFront(:,1), ParetoFront(:,2), 'bo', 'MarkerFaceColor', 'g');
xlabel('Economic Objective (f1)');
ylabel('Environmental Objective (f2)');
title('Final Pareto Front - SPEA2 for CCT Rework');
grid on;

%% ================== SUPPORT FUNCTIONS =====================
function fitness = SPEA2_Fitness(pop)
    N = numel(pop);
    Costs = reshape([pop.Cost], 2, [])';
    S = zeros(N,1);
    for i = 1:N
        for j = 1:N
            if Dominates(Costs(i,:), Costs(j,:))
                S(i) = S(i) + 1;
            end
        end
    end
    R = zeros(N,1);
    for i = 1:N
        for j = 1:N
            if Dominates(Costs(j,:), Costs(i,:))
                R(i) = R(i) + S(j);
            end
        end
    end
    D = zeros(N,1);
    for i = 1:N
        dist = sqrt(sum((Costs - Costs(i,:)).^2, 2));
        dist(i) = inf;
        D(i) = 1 / (min(dist) + 2);
    end
    fitness = R + D;
end

function b = Dominates(x, y)
    b = all(x <= y) && any(x < y);
end

function mating_pool = TournamentSelection(archive, N)
    mating_pool = repmat(archive(1), N, 1);
    for i = 1:N
        a = randi([1, numel(archive)]);
        b = randi([1, numel(archive)]);
        if rand < 0.5
            mating_pool(i) = archive(a);
        else
            mating_pool(i) = archive(b);
        end
    end
end

function [y1, y2] = SBX_Crossover(x1, x2, VarMin, VarMax, eta, pc)
    nVar = numel(x1);
    y1 = zeros(size(x1));
    y2 = zeros(size(x2));
    if rand <= pc
        for i = 1:nVar
            u = rand;
            if u <= 0.5
                beta = (2*u)^(1/(eta+1));
            else
                beta = (1/(2*(1-u)))^(1/(eta+1));
            end
            y1(i) = 0.5*((1+beta)*x1(i)+(1-beta)*x2(i));
            y2(i) = 0.5*((1-beta)*x1(i)+(1+beta)*x2(i));
        end
    else
        y1 = x1; y2 = x2;
    end
    y1 = min(max(y1, VarMin), VarMax);
    y2 = min(max(y2, VarMin), VarMax);
end

function y = PolynomialMutation(x, VarMin, VarMax, eta, pm)
    nVar = numel(x);
    y = x;
    for i = 1:nVar
        if rand < pm
            u = rand;
            if u < 0.5
                delta = (2*u)^(1/(eta+1)) - 1;
            else
                delta = 1 - (2*(1-u))^(1/(eta+1));
            end
            y(i) = x(i) + delta*(VarMax(i) - VarMin(i));
            y(i) = min(max(y(i), VarMin(i)), VarMax(i));
        end
    end
end
