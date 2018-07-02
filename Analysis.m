clear all

%导入数据
% filename = './log_2017-11-15 16-02-23-OK.txt';
% delimiterIn = ' ';
% headerlinesIn = 1;
% A = importdata(filename,delimiterIn,headerlinesIn);
% fileID = fopen('./log_2017-11-15 16-02-23-OK.txt');
fileID = fopen('./log_2017-11-17 10-54-49.txt');
InData = textscan(fileID,'%s %f %f %f %f %f %f %f %f %f %f');
fclose(fileID);

% Time = InData{:,1};
% RevolutionsL = cell2mat(InData(:,2));
% RevolutionsR = cell2mat(InData(:,3));
% RotateSpeedL = cell2mat(InData(:,4));
% RotateSpeedR = cell2mat(InData(:,5));
% TensionVL = cell2mat(InData(:,6));
% TensionVR = cell2mat(InData(:,7));
% MotorSetL = cell2mat(InData(:,8));
% MotorSetR = cell2mat(InData(:,9));
Time = InData{:,1};
RevolutionsL = InData{:,2};
RevolutionsR = InData{:,3};
RotateSpeedL = InData{:,4};
RotateSpeedR = InData{:,5};
TensionVL = InData{:,6};
TensionVR = InData{:,7};
JoyStickL = InData{:,8};
JoyStickR = InData{:,9};
MotorSetL = InData{:,10};
MotorSetR = InData{:,11};

% T=1:1:length(Time);
% T=T';
% figure;
% hold on;
% plot(T,RevolutionsL,T,RevolutionsR);
% figure;
% hold on;
% plot(T,RotateSpeedL,T,RotateSpeedR);
% figure;
% hold on;
% plot(T,TensionVL,T,TensionVR);
% figure;
% hold on;
% plot(T,MotorSetL,T,MotorSetR);
% 
% clear all
% fileID = fopen('./拉力测试.txt');
% InData = textscan(fileID,'%s %f %f %f %f %f %f %f %f');
% fclose(fileID);
% Time = InData{:,1};
% RevolutionsL = InData{:,2};
% RevolutionsR = InData{:,3};
% RotateSpeedL = InData{:,4};
% RotateSpeedR = InData{:,5};
% TensionVL = InData{:,6};
% TensionVR = InData{:,7};
% MotorSetL = InData{:,8};
% MotorSetR = InData{:,9};
% diff=TensionVR-TensionVL;
%%
rangeplot=1:1:length(Time);
rangeplot=5500:1:6500;
fac = 1;
figure; 
subplot(5,1,1);plot(rangeplot*fac, TensionVL(rangeplot));hold on ;plot(rangeplot*fac, TensionVR(rangeplot),'r');
subplot(5,1,2);plot(rangeplot*fac, RotateSpeedL(rangeplot));ylim([-3,3]);hold on;plot(rangeplot*fac, RotateSpeedR(rangeplot),'r');
subplot(5,1,3);plot(rangeplot*fac, RevolutionsL(rangeplot));hold on;plot(rangeplot*fac, RevolutionsR(rangeplot))
subplot(5,1,4);plot(rangeplot*fac, MotorSetL(rangeplot));hold on;plot(rangeplot*fac, MotorSetR(rangeplot))
subplot(5,1,5);plot(rangeplot*fac, JoyStickL(rangeplot));hold on ;plot(rangeplot*fac, JoyStickR(rangeplot),'r');

