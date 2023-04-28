clc;clear;

[file,path,FilterIndex]  = uigetfile('*.*');
file = [path filesep file];

[pathstr,nameSession,ext] = fileparts(file);
file = [pathstr nameSession];

software=1;  % Software lockin

phot=readPhotometryData(file);

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
%[box boxts behavData] = readBoxData_LT(file);

%only comment this out if analyzing noisy pulse session
start = find(diff(box.Data(3,:))<-1);
start = start(1);
pulses = find(diff(box.Data(3,:))>1); %times of pulses from arduino

visits = pulses-start;

%for noisy pulses (03242021, 03252021) use the following
%note: might not pick up initial session start pulse, but might be when
%average oscillations drop in magnitude

% sig_dips = find((diff(box.Data(3,:))<0)&(diff(box.Data(3,:))~=-8));
% start = sig_dips(1);
% visits = find((diff(box.Data(3,:))>0)&(diff(box.Data(3,:))~=8));
% visits = visits - start;

% for use before 06/2020
% pulses = find(diff(phot.Data(1,:))>5000);

%start = pulses(1);
%visits = pulses(2:end)-start;
%% Lockin Detection

tau = 10;
filterOrder = 5;

detector = phot.Data(6,:);
exc1 = phot.Data(7,:);
exc2 = phot.Data(8,:);
[sig1,ref]=lockinDetection(detector,exc1,exc2,phot.SamplingRate,'tau',tau,'filterorder',filterOrder,'detrend',false,'Full',true);

% detector = phot.Data(5,:);
% exc1 = phot.Data(4,:);
% exc2 = phot.Data(8,:);
% [sig2,~]=lockinDetection(detector,exc1,exc2,phot.SamplingRate,'tau',tau,'filterorder',filterOrder,'detrend',false,'Full',true);

detector = phot.Data(3,:);
exc1 = phot.Data(1,:);
exc2 = phot.Data(2,:);
[sig2,ref2]=lockinDetection(detector,exc1,exc2,phot.SamplingRate,'tau',tau,'filterorder',filterOrder,'detrend',false,'Full',true);

% Cut off beginning of photometry signal to match with behavioral data
sig1 = sig1(start:end);
sig2 = sig2(start:end);
ref = ref(start:end);
ref2 = ref2(start:end);

% sig1 = sig1(start:start+phot.SamplingRate*21*60);
% sig2 = sig2(start:start+phot.SamplingRate*21*60);
% ref = ref(start:start+phot.SamplingRate*21*60);

loc = phot.Channel(3).Location(1:15);
save(append(pathstr,'signals'),'sig1','sig2','ref','loc','visits');

%% Isosbestic Correction

% [dF_F1,ref_fitted1,slope1] = isosbestic_correction(sig1,ref); %green data
% 
% [dF_F2,ref_fitted2,slope2] = isosbestic_correction(sig2,ref); %red Data

% save(append(pathstr,'isosbestic'),'dF_F1','ref_fitted1','dF_F2','ref_fitted2');
% 
% %% Iso with quadratic fit
% [dF_F1,ref_fitted1] = isosbestic_correction_quad(sig1,ref); %ACh data
% 
% [dF_F2,ref_fitted2] = isosbestic_correction_quad(sig2,ref); %rdLight Data
% 
% save(append(pathstr,'isosbestic_quad'),'dF_F1','ref_fitted1','dF_F2','ref_fitted2');

%% Iso with 3rd degree polynomial fit
%  [dF_F1,ref_fitted1] = isosbestic_correction_poly(sig1,ref); %ACh data
%  
%  [dF_F2,ref_fitted2] = isosbestic_correction_poly(sig2,ref); %rdLight Data
%  
%  save(append(pathstr,'isosbestic'),'dF_F1','ref_fitted1','dF_F2','ref_fitted2');
%  
%  clear;