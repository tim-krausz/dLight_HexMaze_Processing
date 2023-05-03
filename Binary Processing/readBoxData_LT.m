function [box boxts behavData] = readboxData_LT(file)


box.file = [file '.box'];

fid = fopen(box.file,'r','b');
box.magic_key = fread(fid,1,'uint32');
box.header_Size = fread(fid,1,'int16');
box.main_version = fread(fid,1,'int16');
box.secondary_version = fread(fid,1,'int16');
box.SamplingRate = fread(fid,1,'int16');
box.BytesPerSample = fread(fid,1,'int16');
box.NumberOfChannels = fread(fid,1,'int16');
box.fileName = char(fread(fid,256,'char'));
box.Date = char(fread(fid,256,'char'));
box.Time = char(fread(fid,256,'char'));
box.Ch1_Location = char(fread(fid,256,'char'));
box.Ch2_Location = char(fread(fid,256,'char'));
box.Ch3_Location = char(fread(fid,256,'char'));
position = ftell(fid);
box.pad = fread(fid,box.header_Size-position,'uint8');
box.Data = fread(fid,'uint8');
fclose(fid)


box.Data = reshape(box.Data,box.NumberOfChannels,[]);


fid = fopen([file,'__LinearTrack.log']);
tempCellArray = textscan(fid, ' Subject:%s Date:%s Time:%s Brain Region: %s Trial:  %u16 Active FP:  %u16 Reward:  %u16 Latency1: %u32 Latency2: %u32 Latency3: %u32 FP1 prob: %u16 FP2 prob:  %u16 ITI: %u16','MultipleDelimsAsOne',1);
fclose(fid);

behavData.subject = tempCellArray{1};
behavData.date = tempCellArray{2};
behavData.time = tempCellArray{3};
behavData.brainRegion = tempCellArray{4};
behavData.trial = double(tempCellArray{5});
behavData.activeFP = double(tempCellArray{6});
behavData.reward = double(tempCellArray{7});
behavData.latency1 = double(tempCellArray{8})/1000;
behavData.latency2 = double(tempCellArray{9})/1000;
behavData.latency3 = double(tempCellArray{10})/1000;
behavData.FP1prob = double(tempCellArray{11});
behavData.FP2prob = double(tempCellArray{12});
behavData.ITI = double(tempCellArray{13});


Trial = double(1-bitget(uint8(box.Data(3,:)),4));
Trial = diff(Trial); TrialOn = find(Trial==1); TrialOff = find(Trial==-1); 

for t=1:length(TrialOn)
    fm = abs( TrialOn(t)-TrialOff);
    output = find( fm==min(fm));
    dur(t)=  1000*(TrialOff(output) - TrialOn(t))/box.SamplingRate;
end

Trial = find(Trial==1); Trial=Trial(abs(dur-10)<1); % Trial start is a 10ms pulse. Anything beyond that is perhaps an artifact


Beam1 = double(1-bitget(uint8(box.Data(1,:)),4));
Beam1 = diff(Beam1); Beam1 = (Beam1==1); Beam1 = find(Beam1==1);

Beam2 = double(1-bitget(uint8(box.Data(1,:)),3));
Beam2 = diff(Beam2); Beam2 = (Beam2==1); Beam2 = find(Beam2==1);

Beam3 = double(1-bitget(uint8(box.Data(1,:)),2));
Beam3 = diff(Beam3); Beam3 = (Beam3==1); Beam3 = find(Beam3==1);

FP1 = double(1-bitget(uint8(box.Data(1,:)),5));
FP1 = diff(FP1); FP1 = (FP1==1); FP1 = find(FP1==1);

FP2  = double(1-bitget(uint8(box.Data(1,:)),1));
FP2 = diff(FP2); FP2 = (FP2==1); FP2 = find(FP2==1);



% 
% trialLength = diff(Trial);
% trialSum = behavData.latency1 + behavData.latency2 + behavData.latency3 + behavData.ITI +10; trialSum = trialSum/1000;
% 
% figure,plot(trialLength,trialSum(2:end),'o')


boxts.event(1,:) = Trial;

    
for trial=1:length(Trial)
    activeFP = behavData.activeFP(trial);
    if activeFP==1
        dummyBeam1=Beam3;
    else
        dummyBeam1=Beam1;
    end
    dummyBeam1(dummyBeam1<Trial((trial)))=[];%dummyBeam2(dummyBeam2>5+CStime(trial))=[];
    [~,idx] = min(abs(dummyBeam1-Trial((trial))));
    if ~isempty(idx)
        boxts.event(2,trial)= dummyBeam1(idx);
    else
        boxts.event(2,trial)=nan;
    end
    
     if activeFP==1
        dummyBeam1=Beam3;
    else
        dummyBeam1=Beam1;
    end
%     if behavData.latency1(trial)==0
%        boxts.event(2,trial) =  dummyBeam1(findClosestIndex(dummyBeam1,boxts.event(1,trial)));
%     end

    
    dummyBeam2=Beam2;
    dummyBeam2(dummyBeam2<Trial((trial)))=[];%dummyBeam2(dummyBeam2>5+CStime(trial))=[];
    dummyBeam2(dummyBeam2<boxts.event(2,trial))=[];%dummyBeam2(dummyBeam2>5+CStime(trial))=[];
    [~,idx] = min(abs(dummyBeam2-Trial((trial))));
    if ~isempty(idx)
        boxts.event(3,trial)= dummyBeam2(idx);
    else
        boxts.event(3,trial)=nan;
    end
    
    if activeFP==1
        dummyBeam3=Beam1;
    else
        dummyBeam3=Beam3;
    end
    dummyBeam3(dummyBeam3<Trial((trial)))=[];%dummyBeam2(dummyBeam2>5+CStime(trial))=[];
    dummyBeam3(dummyBeam3<boxts.event(3,trial))=[];%dummyBeam2(dummyBeam2>5+CStime(trial))=[];
    [~,idx] = min(abs(dummyBeam3-Trial((trial))));
    if ~isempty(idx)
        boxts.event(4,trial)= dummyBeam3(idx);
    else
        boxts.event(4,trial)=nan;
    end
end

for trial=1:length(behavData.activeFP)
    activeFP = behavData.activeFP(trial);
    
    dummyFood=sort([FP1,FP2]);
    dummyFood(dummyFood<Trial((trial)))=[];%dummyBeam2(dummyBeam2>5+CStime(trial))=[];
    dummyFood(dummyFood<boxts.event(4,trial))=[];%dummyBeam2(dummyBeam2>5+CStime(trial))=[];
    if trial~=length(behavData.activeFP)
        dummyFood(dummyFood>boxts.event(4,(trial+1)))=[];
    end
    [~,idx] = min(abs(dummyFood-Trial((trial))));
    if ~isempty(idx)
        boxts.event(5,trial)= dummyFood(idx);
    else
        boxts.event(5,trial)=nan;
    end
    
end

% boxts.event = boxts.event/box.SamplingRate;
% 
% 
latency1Line = boxts.event(2,:)-boxts.event(1,:)-10/1000;
latency2Line = boxts.event(3,:)-boxts.event(2,:);
latency3Line = boxts.event(4,:)-boxts.event(3,:);
latency4Line = boxts.event(5,:)-boxts.event(4,:);

idx=find(~isnan(latency2Line));
d=finddelay(latency2Line(idx),behavData.latency2(idx)*1000);
boxts.event=circshift(boxts.event',d)';
% 
% clear idx
% for t=1:length(latency1Line)
%     idx(t,1) = findClosestIndex( latency2Line(t), behavData.latency2);
%     idx(t,2) = latency2Line(t)-behavData.latency2(idx(t,1))/1000;
% end
%     
% % 

latency1Line = boxts.event(2,:)-boxts.event(1,:)-10/1000;
latency2Line = boxts.event(3,:)-boxts.event(2,:);
latency3Line = boxts.event(4,:)-boxts.event(3,:);
latency4Line = boxts.event(5,:)-boxts.event(4,:);

h=figure
subplot(222),plot(latency1Line,behavData.latency1/1000,'.','MarkerSize',24);title(['R=',num2str(corr(latency1Line',behavData.latency1/1000))])
xlabel('Latency1 (Log File)');ylabel('Latency1 (Digital Lines)');set(gca,'FontSize',16)
subplot(223),plot(latency2Line,behavData.latency2/1000,'.','MarkerSize',24);title(['R=',num2str(corr(latency2Line',behavData.latency2/1000))])
xlabel('Latency2 (Log File)');ylabel('Latency2 (Digital Lines)');set(gca,'FontSize',16)
subplot(224),plot(latency3Line,behavData.latency3/1000,'.','MarkerSize',24);title(['R=',num2str(corr(latency3Line',behavData.latency3/1000))])
xlabel('Latency3 (Log File)');ylabel('Latency3 (Digital Lines)');set(gca,'FontSize',16)
close(h)
% % 
% % 




