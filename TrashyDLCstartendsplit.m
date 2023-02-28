%% VRSselectEndStartEndSplit
%First select the behavior file for the imaging session, and then select the number of planes that
%were imaged and the cell traces for each of these planes.
%HRZ Edition!!! Here alignment is made at the end of this section. If you
%do not want to do so, simply edit the code to save all variables with a u
%in front.
%1/5/2020 - added a per plane data savability in the form a for loop around
%the alignment code

%1/7/2020 - added abf data (unaligned and unnamed) to observe with vr data.
%NOTE: will need to pick a new scanstart and scanstop for abf iterations

%3/13/2021 - changed the scaling for ROE to velocity as The scaling used
%was to calculate displacement. LINE 89 now divides by dt as well


%added lickVoltage. altered the values for teleportation indicies for HRZ
%during alignment.

%NOTE: assigning values at teleportation points takes whichever position
%has more iterations during this frame of cellular activity. I.E. if they
%spent 3 points at y = 180 and 2 at y = 0 during the time in which this
%frame of celular data was recorded, y would be assined to 180.

%does this as well with trial num to ensure the first trial num is 0 and
%not 30 for the new rewlocation

%clear all;
% [filename,path]=uigetfile('*.mat','pick your behavior file');
% cd (path); %set path
% fullfilename=[path char(filename)];
% load(fullfilename);

imageSync = [];

%Find start and stop of imaging using VR


if isfield(VR,'imageSync') %makes sure VR has an imageSync variable, if not uses abf, BUT still uses VR variables later
    imageSync = VR.imageSync;
else
    [abffilename,abfpath] = uigetfile('*.abf','pick your abf file');
    abffullfilename = [abfpath char(abffilename)];
    data = abfload(abffullfilename);
    imageSync = data(:,8);
    
end

inds=find((abs(diff(imageSync))>0.3*max(abs(diff(imageSync))))==1);
meaninds=mean(diff(inds));
figure;subplot(2,1,1);hold on;plot(imageSync);plot(abs(diff(imageSync))>0.3*max(abs(diff(imageSync))),'r');
% subplot(2,1,1); hold on; scatter(1000*(VR.time),zeros(1,length(VR.time)),20,'y','filled');
subplot(2,1,2);hold on;plot(imageSync);plot(abs(diff(imageSync))>0.3*max(abs(diff(imageSync))),'r');
xlim([inds(1)-2.5*meaninds inds(1)+2.5*meaninds]);
%xlim([770 780])
[uscanstart,y]=ginput(1)
uscanstart=round(uscanstart)

figure;subplot(2,1,1);hold on;plot(imageSync);plot(abs(diff(imageSync))>0.3*max(abs(diff(imageSync))),'r');
subplot(2,1,2);hold on;plot(imageSync);plot(abs(diff(imageSync))>0.3*max(abs(diff(imageSync))),'r');
xlim([inds(end)-4*meaninds inds(end)+2*meaninds]);
[uscanstop,y]=ginput(1)
uscanstop=round(uscanstop)
disp(['Length of scan is ', num2str(uscanstop-uscanstart)])
disp(['Time of scan is ', num2str((VR.time(uscanstop)-VR.time(uscanstart)))])



close all;
if ~isfield(VR,'imageSync') %if there was no VR.imagesync, rewrites scanstart and scanstop to be in VR iteration indices
    %         buffer = diff(data(:,4))
    VRlastlick = find(VR.lick>0,1,'last');
    abflicks = findpeaks(-1*data(:,3),5);
    buffer = abflicks.loc(end)/1000-VR.time(VRlastlick);
    check_imaging_start_before = (uscanstart/1000-buffer); %there is a chance to recover imaging data from before you started VR (if you made an error) in this case so checking for that
    [trash,scanstart] = min(abs(VR.time-(uscanstart/1000-buffer)));
    [trash,scanstop] = min(abs(VR.time-(uscanstop/1000-buffer)));
else
    scanstart = uscanstart;
    scanstop = uscanstop;
    check_imaging_start_before = 0; %there is no chance to recover imaging data from before you started VR so sets to 0
    
end


%cuts all of the variables from VR
urewards=VR.reward(scanstart:scanstop); 
uimageSync=imageSync(scanstart:scanstop); 
uforwardvel=-0.013*VR.ROE(scanstart:scanstop)./diff(VR.time(scanstart-1:scanstop));  
uybinned=VR.ypos(scanstart:scanstop);   
unumframes=length(scanstart:scanstop);
uVRtimebinned = VR.time(scanstart:scanstop)- check_imaging_start_before-VR.time(scanstart);
utrialnum = VR.trialNum(scanstart:scanstop);
uchangeRewLoc = VR.changeRewLoc(scanstart:scanstop);
uchangeRewLoc(1) = VR.changeRewLoc(1);
ulicks = VR.lick(scanstart:scanstop);
ulickVoltage = VR.lickVoltage(scanstart:scanstop);

%% for loading abf data as well

addabf = input('Would you like to add abf data? (0-no,1-yes)'); %note this adds abf data as "abfdata," to your F files, cut at the imaging points but not aligned or named for generalization purposes 
if addabf
    [abffilename,abfpath] = uigetfile('*.abf','pick your abf file');
    abffullfilename = [abfpath char(abffilename)];
    data = abfload(abffullfilename);  
    imagingchannel = input('Which channel is the imaging channel?');
    inds=find((abs(diff(data(:,imagingchannel)))>0.3*max(abs(diff(data(:,5)))))==1);
    meaninds=mean(diff(inds));
    figure;subplot(2,1,1);hold on;plot(data(:,imagingchannel));plot(abs(diff(data(:,5)))>0.3*max(abs(diff(data(:,imagingchannel)))),'r');
    subplot(2,1,2);hold on;plot(data(:,imagingchannel));plot(abs(diff(data(:,imagingchannel)))>0.3*max(abs(diff(data(:,imagingchannel)))),'r');
    xlim([inds(1)-2.5*meaninds inds(1)+2.5*meaninds]);
    [abfscanstart,y] = ginput(1)
    abfscanstart = round(abfscanstart)

    
    figure;subplot(2,1,1);hold on;plot(data(:,imagingchannel));plot(abs(diff(data(:,imagingchannel)))>0.3*max(abs(diff(data(:,imagingchannel)))),'r');
    subplot(2,1,2);hold on;plot(data(:,imagingchannel));plot(abs(diff(data(:,5)))>0.3*max(abs(diff(data(:,imagingchannel)))),'r');
    xlim([inds(end)-4*meaninds inds(end)+2*meaninds]);
    [abfscanstop,y]= ginput(1)
    abfscanstop = round(abfscanstop)
    disp(['Length of scan is ', num2str(abfscanstop-abfscanstart)])
    disp(['Time of scan is ', num2str((abfscanstop-abfscanstart)/1000)])
     abfdata = data(abfscastart:abfscanstop,:);
    close all;
end

%Find start and stop of imaging

% rewards_th=1*rewards>(0.1*max(rewards));
%  rewards=double(rewards_th);
% rewards_df=diff(rewards_th);
% rewards=[rewards_df(1); rewards_df']>=1;

%bin and average both forwardvel and rotationvel
%raw data need smoothing or binning to see well on compressed x scale.
%     velbinsize=200;   %  # of frames to bin.  50 looks ok.
%     binforwardvel=reshape(forwardvel,velbinsize,(numframes/velbinsize));    %gets 50 frames and puts into column.
%                                                                             %should have (numframes/50) columns
%     meanbinforwardvel=mean(binforwardvel);  %mean of each column(bin) and turns into vector of bins
%     binrotationvel=reshape(rotationvel,velbinsize,(numframes/velbinsize));    %for rotation
%     meanbinrotationvel=mean(binrotationvel);
%     timebinx=((velbinsize/2):velbinsize:(numframes-(velbinsize/2)));    %gives x(time) value of center of bin for plotting.
%
%     figure;
%     hold on;
%     plot(ybinned);
%     plot(rewards*600,'r');

%%
numfiles=input('Number of planes: ');
Ffile{numfiles}=0;
Ffilepath{numfiles}=0;
%%
% for n=1:numfiles
%     [Ffile{n},Ffilepath{n}]=uigetfile('*.mat',['pick the F file for plane ' num2str(n)]);
% end

%% aligns structure so size is the same GM

%quick fix for suite2p making the first plane one frame longer
% load([Ffilepath{2} Ffile{2}])
% testlength = length(F(:,1));
% load([Ffilepath{1} Ffile{1}])
% if length(F(:,1))>testlength
%     F(length(F(:,1)),:) = [];
%     dFF(length(F(:,1)),:) = [];
%     Fc(length(F(:,1)),:) = [];
%     nF(length(F(:,1)),:) = [];
%     save([Ffilepath{1} Ffile{1}],'F','dFF','Fc','nF')
% end
% fullFfile = [Ffilepath{1} Ffile{1}];
% load(fullFfile);
utimedFF = linspace(0,(VR.time(scanstop)-VR.time(scanstart)),(numfiles*length(bodyparts(:,1))));
for n = 1:numfiles
%     fullFfile = [Ffilepath{n} Ffile{n}];
%     load(fullFfile);
%     
    
    clear ybinned rewards forwardvel licks changeRewLoc trialnum timedFF lickVoltage
    
    timedFF = utimedFF(n:numfiles:end);
    
    for newindx = 1:length(timedFF)
        if newindx == 1
            after = mean([timedFF(newindx) timedFF(newindx+1)]);
            rewards(newindx) = sum(urewards(find(uVRtimebinned<=after)));
            forwardvel(newindx) = mean(uforwardvel(find(uVRtimebinned<=after)));
            ybinned(newindx)= mean(uybinned(find(uVRtimebinned<=after)));
            trialnum(newindx) = max(utrialnum(find(uVRtimebinned<=after)));
            changeRewLoc(newindx) = uchangeRewLoc(newindx);
            licks(newindx) = sum(ulicks(find(uVRtimebinned<=after)))>0;
            lickVoltage(newindx) = mean(ulickVoltage(find(uVRtimebinned<=after)));
        elseif newindx == length(timedFF)
            before = mean([timedFF(newindx) timedFF(newindx-1)]);
            rewards(newindx) = sum(urewards(find(uVRtimebinned>before)));
            forwardvel(newindx) = mean(uforwardvel(find(uVRtimebinned>before)));
            ybinned(newindx)= mean(uybinned(find(uVRtimebinned>before)));
            trialnum(newindx) = max(utrialnum(find(uVRtimebinned>before)));
            changeRewLoc(newindx) = sum(uchangeRewLoc(find(uVRtimebinned>before)));
            licks(newindx) = sum(ulicks(find(uVRtimebinned>before)))>0;
            lickVoltage(newindx) = mean(ulickVoltage(find(uVRtimebinned>before)));
        else
            before = mean([timedFF(newindx) timedFF(newindx-1)]);
            after = mean([timedFF(newindx) timedFF(newindx+1)]);
            if isempty(find(uVRtimebinned>before & uVRtimebinned<=after)) && after<= check_imaging_start_before
                rewards(newindx) = urewards(1);
                licks(newindx) = ulicks(1);
                ybinned(newindx) = uybinned(1);
                forwardvel(newindx) = forwardvel(1);
                changeRewLoc(newindx) = 0;
                trialnum(newindx) = utrialnum(1);
                lickVoltage(newindx) = ulickVoltage(newindx);
            elseif isempty(find(uVRtimebinned>before & uVRtimebinned<=after)) && after > check_imaging_start_before
                rewards(newindx) = rewards(newindx-1);
                licks(newindx) = licks(newindx-1);
                ybinned(newindx) = ybinned(newindx-1);
                forwardvel(newindx) = forwardvel(newindx-1);
                changeRewLoc(newindx) = 0;
                trialnum(newindx) = trialnum(newindx-1);
                lickVoltage(newindx) = ulickVoltage(newindx-1);
                
            else
                rewards(newindx) = sum(urewards(find(uVRtimebinned>before & uVRtimebinned<=after)));
                licks(newindx) = sum(ulicks(find(uVRtimebinned>before & uVRtimebinned<=after)))>0;
                lickVoltage(newindx) = mean(ulickVoltage(find(uVRtimebinned>before & uVRtimebinned<=after))); 
                if min(diff(uybinned(find(uVRtimebinned>before & uVRtimebinned<=after)))) < -50
                    dummymin =  min(uybinned(find(uVRtimebinned>before & uVRtimebinned<=after)));
                    dummymax = max(uybinned(find(uVRtimebinned>before & uVRtimebinned<=after)));
                    dummymean = mean(uybinned(find(uVRtimebinned>before & uVRtimebinned<=after)));
                    ybinned(newindx) = ((dummymean/(dummymax-dummymin))<0.5)*dummymin+((dummymean/(dummymax-dummymin))>=0.5)*dummymax; %sets the y value in the case of teleporting to either the end or the beginning based on how many VR iterations it has at each
                    dummytrialmin =  min(utrialnum(find(uVRtimebinned>before & uVRtimebinned<=after)));
                    dummytrialmax = max(utrialnum(find(uVRtimebinned>before & uVRtimebinned<=after)));
                    dummytrialmean = mean(utrialnum(find(uVRtimebinned>before & uVRtimebinned<=after)));
                    trialnum(newindx) = ((dummytrialmean/(dummytrialmax-dummytrialmin))<0.5)*dummytrialmin+((dummytrialmean/(dummytrialmax-dummytrialmin))>=0.5)*dummytrialmax; %sets the trial value in the case of teleporting to either the end or the beginning based on how many VR iterations it has at each

                else
                    ybinned(newindx) = mean(uybinned(find(uVRtimebinned>before & uVRtimebinned<=after)));
                    trialnum(newindx) = max(utrialnum(find(uVRtimebinned>before & uVRtimebinned<=after)));
                end
                forwardvel(newindx) = mean(uforwardvel(find(uVRtimebinned>before & uVRtimebinned<=after)));
                changeRewLoc(newindx) = sum(uchangeRewLoc(find(uVRtimebinned>before & uVRtimebinned<=after)));
 
            end
        end
    end
%     fullFfile=[Ffilepath{n} Ffile{n}]
%      pause(1);
%       save(fullFfile,'ybinned','rewards','forwardvel','licks','changeRewLoc','trialnum','timedFF','lickVoltage','-append');
%       if addabf
%           save(fullFfile,'abfdata','-append');
%       end
end

%%
% for n=1:numfiles
%     fullFfile=[Ffilepath{n} Ffile{n}]
%     %     load(fullFfile);
%     %save(fullFfile,'ybinned','numframes','rewards','angle','forwardvel','rotationvel','velbinsize','meanbinforwardvel','meanbinrotationvel','timebinx','-append');
%     pause(1);
%     
%     
%     save(fullFfile,'ybinned','rewards','forwardvel','licks','changeRewLoc','trialnum','timedFF','-append'); %131018 added galvobinned
%     %     if size(data,2)>7%131018
%     %         save(fullFfile,'ch8binned','-append');
%     %     end
% end


