path = 'data/outex/features/';

cd 'data/outex/features'

directory = dir;

classes = table2array(readtable('classes.csv'));

cd ../../..

info = struct('rate', cell(1,length(directory)), 'stdcv', ...
    cell(1, length(directory)), 'file_name', cell(1, length(directory)));

for k=1:length(directory)
    if contains(directory(k).name,'test3')
        features = readtable(strcat(path, directory(k).name));
        [rate,stdcv,cfMat] = classificationLDA(features, classes, 0,0);
        rate
        info(k).rate = rate;
        info(k).stdcv = stdcv;
        info(k).file_name = directory(k).name;
    end
end

save('info_outex.mat','info')

