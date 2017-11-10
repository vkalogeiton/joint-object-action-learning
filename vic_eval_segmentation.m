function vic_eval_segmentation

% -------------------------------------------------------------------------
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% email: vicky.kalogeiton@gmail.com

% If you use this software please cite our ICCV 2017 paper: 
% Joint learning of object and action detectors
% Vicky Kalogeiton, Philippe Weinzaepfel, Vittorio Ferrari, Cordelia Schmid
% In ICCV 2017

% -------------------------------------------------------------------------
% This function contains the semantic segmentation evalaution of
% object-action pairs. 
% Part of this function comes from the CVPR 2015 "Can Humans Fly? Action Understanding
% with Multiple Classes of Actors" paper.
% If you use this function please cite both papers. 
% -------------------------------------------------------------------------
    
% -------------------------------------------------------------------------
% This script is used to evaluate joint object-action semantic segmentation.
% -------------------------------------------------------------------------

csvPath = [pwd '/utils/videoset.csv'];

% Path to ground truth color images.
gtPath = pwd; % PATH to the ground truth frames

% Results 
rsltPath = [pwd '/SegmentationResults/']; % PATH to your frames

%% Evaluation

VS = readInfo(csvPath);

% Color Map for labels
CMAP = initialCMAP();
labelMap = CMAP(:,1);
validMap = logical(CMAP(:,2));
codeMap = CMAP(:,3)*1000000 + CMAP(:,4)*1000 + CMAP(:,5);

allJoint = zeros(size(labelMap));
onJoint = zeros(size(labelMap));

objectList = unique(floor(labelMap/10));
allObject = zeros(size(objectList));
onObject = zeros(size(objectList));

actionList = unique(mod(labelMap,10));
allAction = zeros(size(actionList));
onAction = zeros(size(actionList));
counting = 0; 

% confusion matrix(i,j): pixel with gt i, classified as j
confMat = zeros(length(validMap),length(validMap));  

for i=1:length(VS.vid)
    if VS.test(i) == 1        
        frameList = dir(fullfile(gtPath, VS.vid{i}, '*.png'));
        for j=1:length(frameList)
            counting = counting +1; 
            disp(['Frames processed: ' num2str(counting) ' / 2365'])
            gtIm = imread(fullfile(gtPath, VS.vid{i}, frameList(j).name));
            try
                rsltIm = imread(fullfile(rsltPath, VS.vid{i}, frameList(j).name));
            catch err
                error('Unable to read %s!', fullfile(rsltPath, frameList(j).name));
            end

            gtIm = double(gtIm);
            rsltIm = double(rsltIm);

            codeGtList = gtIm(:,:,1)*1000000 + gtIm(:,:,2)*1000 + gtIm(:,:,3);
            codeGtList2 = codeGtList(:);
            [~, idx] = ismember(codeGtList2, codeMap);
            labelGtList = labelMap(idx);

            codeRsltList = rsltIm(:,:,1)*1000000 + rsltIm(:,:,2)*1000 + rsltIm(:,:,3);
            codeRsltList = codeRsltList(:);
            [~, idx] = ismember(codeRsltList, codeMap);
            labelRsltList = labelMap(idx);

            % Joint Label
            allJoint = allJoint + histc(labelGtList, labelMap);
            idx = (labelGtList==labelRsltList);
            onJoint = onJoint + histc(labelGtList(idx), labelMap);
            
            % Object Label
            tmpLabelGtList = floor(labelGtList/10);
            tmpLabelRsltList = floor(labelRsltList/10);
            allObject = allObject + histc(tmpLabelGtList, objectList);
            idx = (tmpLabelGtList==tmpLabelRsltList);
            try
                onObject = onObject + histc(tmpLabelGtList(idx), objectList);
            catch
                 onObject = onObject + histc(tmpLabelGtList(idx), objectList)';
            end

            % Action Label
            tmpLabelGtList = mod(labelGtList,10);
            tmpLabelRsltList = mod(labelRsltList,10);
            allAction = allAction + histc(tmpLabelGtList, actionList);
            idx = (tmpLabelGtList==tmpLabelRsltList);
            onAction = onAction + histc(tmpLabelGtList(idx), actionList);
            
            % compute confusion matrix
            uniquecolors = unique(codeGtList);
            codeRsltList = rsltIm(:,:,1)*1000000 + rsltIm(:,:,2)*1000 + rsltIm(:,:,3);
            uniquecolors_seg = unique(codeRsltList);
            for gtcolor_id = 1:length(uniquecolors)
                gtcolor = uniquecolors(gtcolor_id);
                gtl = find(codeMap == gtcolor);
                for segcolor_id = 1:length(uniquecolors_seg)
                    segcolor = uniquecolors_seg(segcolor_id);
                    segl = find(codeMap == segcolor);
                    confMat(gtl, segl) = confMat(gtl, segl) + sum(sum((codeRsltList==segcolor).*(codeGtList==gtcolor)));
                end                    
            end      
        end
    end
    if mod(i,100) == 0
        fprintf('processed %f of all videos ... \n', i/length(VS.vid));
    end
end

%% joint object-action pairs
fprintf('Joint object-action\n');
iou = [];
for ii = 1:length(validMap)
    if sum(confMat(ii, :)) > 0 % make sure that there is gt label 
       iou = cat(1, iou, (confMat(ii, ii)/( sum(confMat(ii, :)) + sum(confMat(:, ii)) - confMat(ii, ii)) ));       
    end
end
disp(['Joint object-action mIoU', num2str(100*mean(iou))])

ObjActIdx = zeros(length(objectList)-1, length(actionList)-1); 
C = 1; 
for ii =1:length(objectList)-1
    for jj = 1:length(actionList)-1
        C = C + 1; 
        ObjActIdx(ii,jj) = C;         
    end
end

allJoint = allJoint(validMap);
onJoint = onJoint(validMap);

jointPerClass = onJoint./allJoint;
jointGlobal = sum(onJoint)/sum(allJoint);
fprintf('Average per-Class object-action pair Accuracy: %f \n', mean(jointPerClass));
fprintf('Average global pixel accuracy: %f \n', mean(jointGlobal));
fprintf('Joing object-action pair mIoU:  %f \n', mean(iou));

%% objects
fprintf('Object Label\n');
% indeces for object classes 
idxO = cell(length(objectList), 1);
idxO{1} = 1; % background
for ii = 2:length(objectList)  
    for jj = 1:length(ObjActIdx(ii-1, :))
        idxO{ii}(jj) = ObjActIdx(ii-1, jj); 
    end
end
ResO = zeros(length(objectList), length(objectList)); 
for ii = 1:length(objectList)
    for jj = 1:length(objectList)
        pixelsO = 0;
        for kk = 1:length(idxO{ii})
            for ll =1:length(idxO{jj})
                pixelsO = pixelsO + confMat(idxO{ii}(kk), idxO{jj}(ll));
            end
        end
        ResO(ii,jj) = pixelsO;  
    end
end
iou_o = [];
for ii = 1:length(objectList)
    if sum(ResO(ii, :)) > 0 % make sure that there is gt label 
       iou_o = cat(1, iou_o, (ResO(ii, ii)/( sum(ResO(ii, :)) + sum(ResO(:, ii)) - ResO(ii, ii)) ));       
    end
end
objectPerClass = onObject./allObject;
objectGlobal = sum(onObject)/sum(allObject);
fprintf('    Average Per-Class Accuracy: %f \n', mean(objectPerClass));

fprintf('Average per-Class object accuracy: %f \n', mean(objectPerClass));
fprintf('Average global object accuracy: %f \n', mean(objectGlobal));
fprintf('Object mIoU:  %f \n', mean(iou_o));

%% actions
% indeces for action classes 
disp('Action Labels');
actionPerClass = onAction./allAction;
actionGlobal = sum(onAction)/sum(allAction);

idxA = cell(length(actionList),1);
idxA{1} = 1; % background
for ii = 2:length(actionList)  
    for jj = 1:length(ObjActIdx(:, ii-1))
        idxA{ii}(jj) = ObjActIdx(jj,ii-1); 
    end
end
ResA = zeros(length(actionList), length(actionList)); 
for ii = 1:length(actionList)
    for jj = 1:length(actionList)
        pixelsA = 0;
        for kk = 1:length(idxA{ii})
            for ll =1:length(idxA{jj})
                pixelsA = pixelsA + confMat(idxA{ii}(kk), idxA{jj}(ll));
            end
        end
        ResA(ii,jj) = pixelsA;  
    end
end
iou_a = [];
for ii = 1:length(actionList)
    if sum(ResA(ii, :)) > 0 % make sure that there is gt label 
       iou_a = cat(1, iou_a, (ResA(ii, ii)/( sum(ResA(ii, :)) + sum(ResA(:, ii)) - ResA(ii, ii)) ));       
    end
end
fprintf('Average per-Class action accuracy: %f \n', mean(actionPerClass));
fprintf('Average global action accuracy: %f \n', mean(actionGlobal));
fprintf('Action mIoU:  %f \n', mean(iou_a));


%% results  
RES(1,1) = mean(objectPerClass)*100;
RES(1,2) = mean(objectGlobal)*100;
RES(1,3) = mean(iou_o)*100;
RES(1,4) = mean(actionPerClass)*100;
RES(1,5) = mean(actionGlobal)*100;
RES(1,6) = mean(iou_a)*100;
RES(1,7) = mean(jointPerClass)*100;
RES(1,8) = jointGlobal*100;
RES(1,9) = mean(iou)*100;

fprintf('Done.\n');
end

function VS = readInfo(csvPath)
% csv file structure:
% vid, label, timestart, timeend, height, width, frames, anno, test
% vid: 11-digit YouTube video id.
% label: single video-level label (e.g. 17 (adult-running)).
% timestart: start time to cut for a clip out of a video.
% timeend: end time for the clip.
% height: resized height.
% width: resized width.
% frames: number of frames of the cut clip.
% anno: number of VALID annotation frames.
% test: 0 - train; 1 - test.

fid = fopen(csvPath, 'r');
C = textscan(fid, '%q%d%q%q%d%d%d%d%d', 'delimiter', ',');
fclose(fid);

VS.vid = C{1};
VS.label = C{2};
VS.timeStart = C{3};
VS.timeEnd = C{4};
VS.height = C{5};
VS.width = C{6};
VS.frames = C{7};
VS.anno = C{8};
VS.test = C{9};

end

function CMAP = initialCMAP()
% This is the label id & color map.
% Columns: id, valid, R, G, B

CMAP = [0,1,0,0,0;
    11,1,52,1,1;12,1,103,1,1;13,1,154,1,1;14,0,205,1,1;15,1,255,1,1;
    16,1,255,51,51;17,1,255,103,103;18,1,255,154,154;19,1,255,205,205;
    21,1,52,46,1;22,1,103,92,1;23,0,154,138,1;24,0,205,184,1;25,0,255,230,1;
    26,1,255,235,51;27,0,255,240,103;28,1,255,245,154;29,1,255,250,205;
    31,0,11,52,1;32,0,21,103,1;33,0,31,154,1;34,1,41,205,1;35,1,52,255,1;
    36,1,92,255,51;37,0,133,255,103;38,0,174,255,154;39,1,215,255,205;
    41,1,1,52,36;42,0,1,103,72;43,1,1,154,108;44,1,1,205,143;45,1,1,255,179;
    46,1,51,255,194;47,0,103,255,210;48,1,154,255,225;49,1,205,255,240;
    51,0,1,21,52;52,0,1,41,103;53,0,1,62,154;54,1,1,82,205;55,1,1,103,255;
    56,1,51,133,255;57,1,103,164,255;58,0,154,194,255;59,1,205,225,255;
    61,1,26,1,52;62,0,52,1,103;63,1,77,1,154;64,0,103,1,205;65,1,128,1,255;
    66,1,154,51,255;67,1,179,103,255;68,1,205,154,255;69,1,230,205,255;
    71,0,52,1,31;72,1,103,1,62;73,1,154,1,92;74,0,205,1,123;75,1,255,1,153;
    76,1,255,51,174;77,1,255,103,194;78,1,255,154,215;79,1,255,205,235;
    ];

end
