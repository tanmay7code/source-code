clc; clear; close all; 
% ===================== GA / NSGA-II Control Parameters =====================
nv=3; % number of decision variables: [tp_i, mu, G]
lmt=[0,100; 0.01,2; 0,300]; % bounds for tp_i, mu, G
d=[2,3,2]; % decimal accuracy for each variable
pop=50; % population size (reduce if CCT evaluation is slow)
maxgen=325; % number of generations
pc=0.75; % crossover probability
pm=0.05; % mutation probability
rng(1); % reproducible runs

% ===================== Binary representation lengths =====================
slen=zeros(nv,1); tlen=0;
for j=1:nv
    ltmp = log((lmt(j,2)-lmt(j,1))*10^d(j)+1)/log(2);
    slen(j,1)=ceil(ltmp);
    tlen=tlen+slen(j,1);
end

% ===================== Initialize binary population bp (pop x tlen) =====================
bp=zeros(pop,tlen);
for p=1:pop
    for s=1:tlen
        if rand<=0.5
            bp(p,s)=0;
        else
            bp(p,s)=1;
        end
    end
end

% ===================== Decode binary to real variables xp (pop x nv) =====================
xp=zeros(pop,nv);
for p=1:pop
    c=0;
    for j=1:nv
        sumv=0;
        for k=1:slen(j,1)
            sumv=sumv+bp(p,c+k)*(2^(slen(j,1)-k));
        end
        xp(p,j)=lmt(j,1)+sumv*((lmt(j,2)-lmt(j,1))/(2^slen(j,1)-1));
        c=c+slen(j,1);
    end
end

% ===================== Evaluate parent population using CCT_Objectives_Rework =====================
fp1=zeros(pop,1); fp2=zeros(pop,1);
for p=1:pop
    xvec = xp(p,1:nv);
    fvals = CCT_Objectives_Rework(xvec);
    fp1(p,1)=fvals(1);
    fp2(p,1)=fvals(2);
end

Iter=zeros(maxgen,1);

% ===================== Main generational loop =====================
for gen=1:maxgen

    % -------------------------
    % Non-dominated sorting for parent population
    % -------------------------
    Sp=cell(pop,1); np=zeros(pop,1); F=cell(pop,1); R=zeros(pop,1);
    F{1,1}=[];
    for p=1:pop
        Sp{p,1}=[]; np(p,1)=0;
        for q=1:pop
            if (fp1(p,1) <= fp1(q,1) && fp2(p,1) <= fp2(q,1)) && (fp1(p,1) < fp1(q,1) || fp2(p,1) < fp2(q,1))
                Sp{p,1}=[Sp{p,1}, q];
            elseif (fp1(q,1) <= fp1(p,1) && fp2(q,1) <= fp2(p,1)) && (fp1(q,1) < fp1(p,1) || fp2(q,1) < fp2(p,1))
                np(p,1)=np(p,1)+1;
            end
        end
        if np(p,1)==0
            R(p,1)=1;
            F{1,1}=[F{1,1}, p];
        end
    end

    % -------------------------
    % Find other fronts
    % -------------------------
    i=1; l1=length(F{1,1});
    while l1~=0
        Q=[];
        for k=1:l1
            idx = F{i,1}(1,k);
            l2=length(Sp{idx,1});
            for s=1:l2
                qidx = Sp{idx,1}(1,s);
                np(qidx,1)=np(qidx,1)-1;
                if np(qidx,1)==0
                    R(qidx,1)=i+1;
                    Q=[Q, qidx];
                end
            end
        end
        i=i+1;
        F{i,1}=Q;
        l1=length(F{i,1});
    end

    % -------------------------
    % Crowding-distance-like diversity measure
    % -------------------------
    div=zeros(pop,1);
    lf=zeros(pop,1);
    for i=1:pop
        lf(i,1)=length(F{i,1});
    end
    i=1;
    while lf(i,1)~=0
        S1=[]; S2=[]; T1=[];
        for k=1:lf(i,1)
            idx=F{i,1}(1,k);
            T1=[T1; idx];
            S1=[S1; fp1(idx,1)];
            S2=[S2; fp2(idx,1)];
        end
        T2=T1;
        for k=1:lf(i,1)-1
            for k1=k+1:lf(i,1)
                if S1(k,1)>S1(k1,1)
                    tmp=S1(k,1); S1(k,1)=S1(k1,1); S1(k1,1)=tmp;
                    tmp=T1(k,1); T1(k,1)=T1(k1,1); T1(k1,1)=tmp;
                end
            end
        end
        for k=1:lf(i,1)-1
            for k1=k+1:lf(i,1)
                if S2(k,1)>S2(k1,1)
                    tmp=S2(k,1); S2(k,1)=S2(k1,1); S2(k1,1)=tmp;
                    tmp=T2(k,1); T2(k,1)=T2(k1,1); T2(k1,1)=tmp;
                end
            end
        end
        if lf(i,1)>0
            div(T1(1,1),1)=inf; div(T1(lf(i,1),1),1)=inf;
            for k=2:lf(i,1)-1
                denom1 = (S1(lf(i,1),1)-S1(1,1));
                denom2 = (S2(lf(i,1),1)-S2(1,1));
                if denom1~=0
                    div(T1(k,1),1)=div(T1(k,1),1)+((S1(k+1,1)-S1(k-1,1))/denom1);
                end
                if denom2~=0
                    div(T2(k,1),1)=div(T2(k,1),1)+((S2(k+1,1)-S2(k-1,1))/denom2);
                end
            end
        end
        i=i+1;
    end

    % -------------------------
    % Binary crowded tournament selection
    % -------------------------
    id=zeros(pop,1);
    for p=1:pop
        q=p;
        while q==p
            r=rand; q=floor(r*pop);
            if q==0, q=1; end
        end
        if R(p,1)<R(q,1)
            id(p,1)=p;
        elseif R(q,1)<R(p,1)
            id(p,1)=q;
        elseif R(p,1)==R(q,1) && div(p,1)>div(q,1)
            id(p,1)=p;
        elseif R(p,1)==R(q,1) && div(p,1)==div(q,1)
            id(p,1)=p;
        else
            id(p,1)=p;
        end
    end

    % -------------------------
    % Two-point crossover to form children bc
    % -------------------------
    bc=zeros(pop,tlen);
    kcnt=0;
    while kcnt<pop
        q=1; psel=1;
        while q==psel
            r=rand; psel=floor(r*pop); if psel==0, psel=1; end
            r=rand; q=floor(r*pop); if q==0, q=1; end
        end
        psel=id(psel,1); q=id(q,1);
        r=rand;
        if r<=pc
            cs1=1; cs2=1;
            while cs1==cs2
                r=rand; cs1=floor(r*(tlen-1)); if cs1==0, cs1=1; end
                r=rand; cs2=floor(r*(tlen-1)); if cs2==0, cs2=1; end
                if cs1>cs2, tmp=cs1; cs1=cs2; cs2=tmp; end
            end
            for s=1:cs1
                bc(kcnt+1,s)=bp(psel,s); bc(kcnt+2,s)=bp(q,s);
            end
            for s=cs1+1:cs2
                bc(kcnt+1,s)=bp(q,s); bc(kcnt+2,s)=bp(psel,s);
            end
            for s=cs2+1:tlen
                bc(kcnt+1,s)=bp(psel,s); bc(kcnt+2,s)=bp(q,s);
            end
        else
            for s=1:tlen
                bc(kcnt+1,s)=bp(psel,s); bc(kcnt+2,s)=bp(q,s);
            end
        end
        kcnt=kcnt+2;
    end

    % -------------------------
    % Bit-flip mutation on children bc
    % -------------------------
    for p=1:pop
        for s=1:tlen
            if rand<=pm
                bc(p,s)=1-bc(p,s);
            end
        end
    end

    % -------------------------
    % Decode children bc -> xc
    % -------------------------
    xc=zeros(pop,nv);
    for p=1:pop
        c=0;
        for j=1:nv
            sumv=0;
            for k=1:slen(j,1)
                sumv=sumv+bc(p,c+k)*(2^(slen(j,1)-k));
            end
            xc(p,j)=lmt(j,1)+sumv*((lmt(j,2)-lmt(j,1))/(2^slen(j,1)-1));
            c=c+slen(j,1);
        end
    end

    % -------------------------
    % Evaluate child population using CCT_Objectives_Rework
    % -------------------------
    fc1=zeros(pop,1); fc2=zeros(pop,1);
    for p=1:pop
        xvec = xc(p,1:nv);
        fvals = CCT_Objectives_Rework(xvec);
        fc1(p,1)=fvals(1);
        fc2(p,1)=fvals(2);
    end

    % -------------------------
    % Form combined population (parents + children)
    % -------------------------
    b=zeros(2*pop,tlen); xcomb=zeros(2*pop,nv); f1=zeros(2*pop,1); f2=zeros(2*pop,1);
    for i=1:pop
        b(i,:)=bp(i,:); xcomb(i,:)=xp(i,:); f1(i,1)=fp1(i,1); f2(i,1)=fp2(i,1);
        b(pop+i,:)=bc(i,:); xcomb(pop+i,:)=xc(i,:); f1(pop+i,1)=fc1(i,1); f2(pop+i,1)=fc2(i,1);
    end

    % -------------------------
    % Non-dominated sorting for combined population (2*pop)
    % -------------------------
    Spc=cell(2*pop,1); npc=zeros(2*pop,1); Fc=cell(2*pop,1); Rc=zeros(2*pop,1); Fc{1,1}=[];
    for p=1:2*pop
        Spc{p,1}=[]; npc(p,1)=0;
        for q=1:2*pop
            if (f1(p,1) <= f1(q,1) && f2(p,1) <= f2(q,1)) && (f1(p,1) < f1(q,1) || f2(p,1) < f2(q,1))
                Spc{p,1}=[Spc{p,1}, q];
            elseif (f1(q,1) <= f1(p,1) && f2(q,1) <= f2(p,1)) && (f1(q,1) < f1(p,1) || f2(q,1) < f2(p,1))
                npc(p,1)=npc(p,1)+1;
            end
        end
        if npc(p,1)==0
            Rc(p,1)=1; Fc{1,1}=[Fc{1,1}, p];
        end
    end

    % -------------------------
    % Find remaining fronts for combined population
    % -------------------------
    i=1; l1c=length(Fc{1,1});
    while l1c~=0
        Qc=[];
        for k=1:l1c
            idx=Fc{i,1}(1,k);
            l2c=length(Spc{idx,1});
            for s=1:l2c
                qidx=Spc{idx,1}(1,s);
                npc(qidx,1)=npc(qidx,1)-1;
                if npc(qidx,1)==0
                    Rc(qidx,1)=i+1; Qc=[Qc, qidx];
                end
            end
        end
        i=i+1; Fc{i,1}=Qc; l1c=length(Fc{i,1});
    end

    % -------------------------
    % Crowding-distance for combined population
    % -------------------------
    divc=zeros(2*pop,1); lfc=zeros(2*pop,1);
    for i=1:2*pop, lfc(i,1)=length(Fc{i,1}); end
    i=1;
    while lfc(i,1)~=0
        S1c=[]; S2c=[]; T1c=[];
        for k=1:lfc(i,1)
            idx=Fc{i,1}(1,k);
            T1c=[T1c; idx];
            S1c=[S1c; f1(idx,1)];
            S2c=[S2c; f2(idx,1)];
        end
        T2c=T1c;
        for k=1:lfc(i,1)-1
            for k1=k+1:lfc(i,1)
                if S1c(k,1)>S1c(k1,1)
                    tmp=S1c(k,1); S1c(k,1)=S1c(k1,1); S1c(k1,1)=tmp;
                    tmp=T1c(k,1); T1c(k,1)=T1c(k1,1); T1c(k1,1)=tmp;
                end
            end
        end
        for k=1:lfc(i,1)-1
            for k1=k+1:lfc(i,1)
                if S2c(k,1)>S2c(k1,1)
                    tmp=S2c(k,1); S2c(k,1)=S2c(k1,1); S2c(k1,1)=tmp;
                    tmp=T2c(k,1); T2c(k,1)=T2c(k1,1); T2c(k1,1)=tmp;
                end
            end
        end
        if lfc(i,1)>0
            divc(T1c(1,1),1)=inf; divc(T1c(lfc(i,1),1),1)=inf;
            for k=2:lfc(i,1)-1
                denom1=(S1c(lfc(i,1),1)-S1c(1,1)); denom2=(S2c(lfc(i,1),1)-S2c(1,1));
                if denom1~=0, divc(T1c(k,1),1)=divc(T1c(k,1),1)+((S1c(k+1,1)-S1c(k-1,1))/denom1); end
                if denom2~=0, divc(T2c(k,1),1)=divc(T2c(k,1),1)+((S2c(k+1,1)-S2c(k-1,1))/denom2); end
            end
        end
        i=i+1;
    end

    % -------------------------
    % Make next generation: pick first 'pop' individuals by fronts then diversity
    % -------------------------
    U=zeros(pop,1); pcount=0; i=1;
    while pcount<pop
        lu=length(Fc{i,1});
        for j=1:lu
            if pcount+1>pop, break; end
            pcount=pcount+1; U(pcount,1)=Fc{i,1}(1,j);
        end
        i=i+1;
    end
    V=U(1:pop,1);

    % Fill bp, xp, fp1, fp2 for next generation
    for i=1:pop
        bp(i,:)=b(V(i,1),:);
        xp(i,:)=xcomb(V(i,1),:);
        fp1(i,1)=f1(V(i,1),1);
        fp2(i,1)=f2(V(i,1),1);
    end

    Iter(gen,1)=gen;
    fprintf('Generation %d done\n', gen);

end % end generation loop

% ===================== final scatter of parent front =====================
scatter(fp1,fp2);
xlabel('Objective 1 (f1)'); 
ylabel('Objective 2 (f2)'); 
title('Final population objective scatter');
