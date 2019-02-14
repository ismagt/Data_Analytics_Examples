close all;
clear;
%% Visualizar los datos y extraer subconjunto binario 
datos_caso=readtable('Frogs_MFCCs.csv');
datos_MFCCs=datos_caso(:,1:22); 
X=table2array(datos_MFCCs);
%%
Y=double(categorical(datos_caso.Genus)); 
idx= Y==3 | Y==5; % me quedo con los valores correspondientes que serian los dos tipos de genero 
% en idx tenemos un 1 cuando es lep o dendro y 0 cuando es otro tipo de genero
X=X(idx,:); 
Y=Y(idx); 
%(3) dendro y (5) lepto
%% 
y=Y==3; % de los 580 de antes que tengo me quedo con los que tienen 3 a uno los pongo. Los que son treses a uno y los 5 cero
%dibujo histograma
figure;
hist(y);
%%
for i=1:22 %i=[3,5,7]
    figure;
    histogram(X(Y==3,i),'Normalization','pdf');  % LA Y MAYUSCULA ES LA DE 3 Y 5 , LA Y MINUS SON 0 Y 1
    hold on; % filas de 3 
    histogram(X(Y==5,i),'Normalization','pdf'); % pdf que se corresponden con la verosimilitud
    hold on; % filas de 5
    histogram(X(:,i),'Normalization','pdf'); %filas totales
end

% en nombre de figura me quedo con 4, 6 y 8
% en columna serian la 3,5,7

%% Decisor ML, cada gaussiana es una probabilidad de verosimilitud y el punto medio 

media_1=mean(X(Y==3,5)); % aplico filtro cojo todas las filas que esten a 1 y la columna 5 y hago la media 
media_0=mean(X(Y==5,5));
varianza_1=var(X(Y==3,5));
varianza_0=var(X(Y==5,5));

eje_x=-1:0.01:1;

% dibujo y cojo la figura de la variable 5 que es la figura 6

figure(6); 
hold on;
plot(eje_x,normpdf(eje_x,media_1,sqrt(varianza_1)));
hold on;
plot(eje_x,normpdf(eje_x,media_0,sqrt(varianza_0)));

syms x; % lo que sería la variable a despejar 
 
umbral=eval(solve(1/sqrt(varianza_1)*exp(-(x-media_1)^2/(2*varianza_1))==1/sqrt(varianza_0)*exp(-(x-media_0)^2/(2*varianza_0)),x));

 figure(6);
 hold on;
 plot([umbral(1) umbral(1)],[0 5],'r'); % el [0 5] es para definir la altura de la recta y el umbral (1)
% % y umbral (1) es el punto donde empieza el eje x 
title('L-Decisor ML-D');

%% Decisor MAP tendremos otras dos gaussianas, donde cada una es la verosimilitud por las a priori que me dara esa multiplicacion la posteriori

p_d= sum(Y==3)/length(Y); % solo sumo los treses y lo divido entre el total
p_l= sum(Y==5)/length(Y);

syms x; % lo que sería la variable a despejar 
 
umbral_map=eval(solve(p_d/sqrt(varianza_1)*exp(-(x-media_1)^2/(2*varianza_1))==p_l/sqrt(varianza_0)*exp(-(x-media_0)^2/(2*varianza_0)),x));

 figure;
 histogram(X(Y==3,5),'Normalization','pdf');  % LA Y MAYUSCULA ES LA DE 3 Y 5 , LA Y MINUS SON 0 Y 1
 hold on;
 histogram(X(Y==5,5),'Normalization','pdf');
 hold on;
 histogram(X(:,5),'Normalization','pdf');
 eje_x=-1:0.01:1;
 hold on;
plot(eje_x,p_d*normpdf(eje_x,media_1,sqrt(varianza_1))); % multiplico la gaussiana por su apriori
hold on;
plot(eje_x,p_l*normpdf(eje_x,media_0,sqrt(varianza_0)));   % multiplico la gaussiana por su apriori
 plot([umbral_map(1) umbral_map(1)],[0 5],'r'); % el [0 5] es para definir la altura de la recta y el umbral (1)
% % y umbral (1) es el punto donde empieza el eje x 
title('L -Decisor MAP-D');

%% Algoritmo K-NN

% Cojo el 80% datos para train y el 20% de test 
X_train= X(1:round(0.8*length(X(:,5))),5); % caracteristicas lo que sería el 80% de los datos 
X_test= X(round(0.8*length(X(:,5))+1):length(X(:,5)),5); %el último 20% de los datos datos de los generos 
Y_train= Y(1:round(0.8*length(Y(:,:))));
Y_test= Y(round(0.8*length(Y(:,:))+1):length(Y(:,:)));
% la suma del x_train y del x_test son 580 que era lo que teniamos,
% viéndolo en el workspace 

Md1=fitcknn(X_train,Y_train,'NumNeighbors',6,'Standardize',1);% 6 es el numero de vecinos en los que nos fijamos creando un modelo Mdl
[label,score,cost]=predict(Md1,X_test); 

porcentaje_acierto_columna5=sum(label==Y_test)/length(Y_test); % esa prediccion que realizado la comparo con los datos reales que se tenian el Y_test
% numero de aciertos/total me da la probabilidad de aciertos 
% Ahora se realiza el knn para todas las columnas(todos los datos)
X_train_bis= X(1:round(0.8*length(X(:,:))),:); % caracteristicas; ahora lo pongo para todos por eso los :
X_test_bis= X(round(0.8*length(X(:,:))+1):length(X(:,:)),:); % datos de los generos 
Y_train_bis= Y(1:round(0.8*length(Y(:,:))));
Y_test_bis= Y(round(0.8*length(Y(:,:))+1):length(Y(:,:)));

Md2=fitcknn(X_train_bis,Y_train_bis,'NumNeighbors',6,'Standardize',1);
[label_bis,score_bis,cost_bis]=predict(Md2,X_test_bis);

porcentaje_acierto_todos=sum(label_bis==Y_test_bis)/length(Y_test_bis);

%% Cálculo de la ROC para el ML
umbrales=sort([linspace(-1,1,100) umbral(1)]); % 100 elementos que van de -1 a 1 en saltos de 0.02 de -1 a 1 hay 2 pues entre 100
for iu=1:length(umbrales)
    PFAs_ML(iu)=int(1/sqrt(2*pi*varianza_0)*exp(-(x-media_0)^2/(2*varianza_0)),umbrales(iu),inf); % pongo con media 0 xq la gaussiana es la de la derecha de dendrosophus
    PDs_ML(iu)=int(1/sqrt(2*pi*varianza_1)*exp(-(x-media_1)^2/(2*varianza_1)),umbrales(iu),inf); % pongo con media 1 xq la gaussiana es la de la izquierda de lendro
end
%calculamos la PFA y PD para 101 valores de umbral distintos 

% se calcua y dibuja la roc para el umbral bueno 
PFA_umbralML=int(1/sqrt(2*pi*varianza_0)*exp(-(x-media_0)^2/(2*varianza_0)),umbral(1),inf); % para dibujarlo luego
PD_umbralML=int(1/sqrt(2*pi*varianza_1)*exp(-(x-media_1)^2/(2*varianza_1)),umbral(1),inf); % para dibujarlo luego
% ahora con el umbral bueno estas dos lineas de arriba
figure; plot(PFAs_ML,PDs_ML,'.-');
hold on; plot(PFA_umbralML,PD_umbralML,'*'); % es el valor de la roc donde esta nuestro umbral ml 
xlabel('P_{FA}'); ylabel('P_D'); title('Curva ROC ML'); xlim([0 1]); ylim([0 1]);

%% Cálculo de la ROC para el MAP
% La ROC se calcula solo para el ML, usando las probabilidades de
% verosimilitud moviendo el umbral. 
umbrales_map=sort([linspace(-1,1,100) umbral_map(1)]); % 100 elementos que van de -1 a 1 en saltos de 0.02 de -1 a 1 hay 2 pues entre 100
for iu=1:length(umbrales_map)
    PFAs_MAP(iu)=int(1/sqrt(2*pi*varianza_0)*exp(-(x-media_0)^2/(2*varianza_0)),umbrales_map(iu),inf); % pongo con media 0 xq la gaussiana es la de la derecha de dendrosophus
    PDs_MAP(iu)=int(1/sqrt(2*pi*varianza_1)*exp(-(x-media_1)^2/(2*varianza_1)),umbrales_map(iu),inf); % pongo con media 1 xq la gaussiana es la de la izquierda de lendro
end

PFA_umbralMAP=int(1/sqrt(2*pi*varianza_0)*exp(-(x-media_0)^2/(2*varianza_0)),umbral_map(1),inf); % para dibujarlo luego
PD_umbralMAP=int(1/sqrt(2*pi*varianza_1)*exp(-(x-media_1)^2/(2*varianza_1)),umbral_map(1),inf); % para dibujarlo luego
figure; plot(PFAs_MAP,PDs_MAP,'.-');
hold on; plot(PFA_umbralMAP,PD_umbralMAP,'O'); % es el valor de la roc donde esta nuestro umbral ml 
xlabel('P_{FA}'); ylabel('P_D'); title('Curva ROC MAP'); xlim([0 1]); ylim([0 1]);

figure();
X=X(y==3 | y==5,:);
y=Y(y==3 | y==5);
N=length(y); k=3;
for i=1:N
    xtest=X(i,:); ytest=y(i);
    Xtrain=X; Xtrain(i,:)=[];
    ytrain=y; ytrain(i)=[];
    %Esto era lo que me quedaba por hacer
    d = sum((xtest - Xtrain).^2 , 2);
    [n,pos] = sort(d, 'ascend');
    yest(i,1) = mode(ytrain(pos(1:k)));
end
ytarget=y; ytarget(y==5)=1; ytarget(y==3)=0; 
youtput=yest; youtput(yest==5)=1; youtput(yest==3)=0; 
plotroc(ytarget',youtput')