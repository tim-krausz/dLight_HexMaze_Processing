function phot=readPhotometryData(file)


phot.file = [file '.phot'];

fid = fopen(phot.file,'r','b');
phot.magic_key = fread(fid,1,'uint32');
phot.header_Size = fread(fid,1,'int16');
phot.main_version = fread(fid,1,'int16');
phot.secondary_version = fread(fid,1,'int16');
phot.SamplingRate = fread(fid,1,'int16');
phot.BytesPerSample = fread(fid,1,'int16');
phot.NumberOfChannels = fread(fid,1,'int16');
phot.fileName = char(fread(fid,256,'char'));
phot.Date = char(fread(fid,256,'char'));
phot.Time = char(fread(fid,256,'char'));

phot.Channel(1).Location = char(fread(fid,256,'char'));
phot.Channel(2).Location = char(fread(fid,256,'char'));
phot.Channel(3).Location = char(fread(fid,256,'char'));
phot.Channel(4).Location = char(fread(fid,256,'char'));

phot.Channel(1).Signal = char(fread(fid,256,'char'));
phot.Channel(2).Signal = char(fread(fid,256,'char'));
phot.Channel(3).Signal = char(fread(fid,256,'char'));
phot.Channel(4).Signal = char(fread(fid,256,'char'));

phot.Channel(1).Freq = fread(fid,1,'int16');
phot.Channel(2).Freq = fread(fid,1,'int16');
phot.Channel(3).Freq = fread(fid,1,'int16');
phot.Channel(4).Freq = fread(fid,1,'int16');

% Ensure values lie in [-1, 1] by dividing by by largest value
% represented in int16
phot.Channel(1).MaxV = fread(fid,1,'int16')/32767;
phot.Channel(2).MaxV = fread(fid,1,'int16')/32767;
phot.Channel(3).MaxV = fread(fid,1,'int16')/32767;
phot.Channel(4).MaxV = fread(fid,1,'int16')/32767;

phot.Channel(1).MinV = fread(fid,1,'int16')/32767;
phot.Channel(2).MinV = fread(fid,1,'int16')/32767;
phot.Channel(3).MinV = fread(fid,1,'int16')/32767;
phot.Channel(4).MinV = fread(fid,1,'int16')/32767;

for signal=1:8
phot.signal_Label(signal,:) =  char(fread(fid,256,'char'));
end

position = ftell(fid);
phot.pad = fread(fid,phot.header_Size-position,'uint8');
phot.Data = fread(fid,'int16');
fclose(fid);

phot.Data = reshape(phot.Data,phot.NumberOfChannels,[]);


end